"""Pipecat behind the Voice PE speaker's `/voice` WebSocket protocol (proto 2).

An experiment next to voice-assistant's own realtime bridge: same wire shape,
so pointing a speaker's `va_url` at this port is the only change. Deliberately
minimal — no follow-up window, no multi-speaker wake arbitration, no
memory/prompts DB.
"""

import asyncio
import hmac
import json
import os
from collections.abc import Awaitable, Callable

import uvicorn
from fastapi import FastAPI, WebSocket
from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioOutput,
    ClientEvent,
    InputAudioBufferClearEvent,
    ResponseCancelEvent,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.runner import WorkerRunner

PROTO_VERSION = 2
DEVICE_MIC_RATE = 16000
# OpenAI Realtime only takes/gives PCM16 @ 24 kHz, and the speaker plays 24 kHz.
PIPELINE_RATE = 24000

DEFAULT_INSTRUCTIONS = (
    "Ты голосовой ассистент умного дома. Отвечай по-русски, коротко, одним-двумя "
    "предложениями. Для управления устройствами используй инструменты Home Assistant."
)

# OpenAI caps a Realtime session at 60 min; recycle at 50, waiting up to 8 more
# for the current turn to finish.
SESSION_RECYCLE_SECS = 50 * 60
SESSION_GRACE_SECS = 8 * 60


class VoicePeSerializer(FrameSerializer):
    """Binary frames are raw PCM16 mono audio; text frames are JSON control."""

    def __init__(self):
        super().__init__()
        self._resampler = create_stream_resampler()
        self._in_rate = PIPELINE_RATE

    async def setup(self, setup: FrameProcessorSetup):
        self._in_rate = setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return None if self.should_ignore_frame(frame) else json.dumps(frame.message)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            audio = await self._resampler.resample(data, DEVICE_MIC_RATE, self._in_rate)
            return InputAudioRawFrame(audio=audio, sample_rate=self._in_rate, num_channels=1)
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            msg = None
        if not isinstance(msg, dict):
            logger.warning(f"bad device message: {data!r}")
            return None
        return InputTransportMessageFrame(message=msg)


class DeviceProtocol(FrameProcessor):
    """Device control messages → pipeline actions; turn frames → device phases.

    Sits between the LLM and transport.output(): sees the user-turn frames
    downstream and the output transport's Bot*SpeakingFrame upstream copies.
    """

    def __init__(self, send_event: Callable[[ClientEvent], Awaitable[None]]):
        super().__init__()
        # Raw OpenAI Realtime client events: with server VAD Pipecat's
        # interruption only truncates the playing item, so cancelling the
        # generation itself is on us.
        self._send_event = send_event
        self._turn: int | None = None
        # `idle` = outside a turn; `start` opens one with phase None.
        self._phase: str | None = "idle"
        self._tool_pending = False

    @property
    def in_turn(self) -> bool:
        """Between the device's `start` and our `idle`. Phases and reply audio
        outside a turn (a cancelled reply's tail, a barge-in's BotStopped) are
        dropped, not sent to the speaker."""
        return self._phase != "idle"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if not await self._handle(frame):
            await self.push_frame(frame, direction)

    async def _handle(self, frame: Frame) -> bool:
        """Returns True when the frame is consumed (device control messages,
        reply audio outside a turn)."""
        if isinstance(frame, InputTransportMessageFrame):
            await self._on_device(frame.message)
            return True
        if isinstance(frame, OutputAudioRawFrame):
            return not self.in_turn
        if isinstance(frame, FunctionCallsStartedFrame):
            self._tool_pending = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._set_phase("listening")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._set_phase("thinking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._tool_pending = False
            await self._set_phase("replying")
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._phase == "replying":
            # A spoken preamble before a tool call ("сейчас включу…") is not the
            # end of the turn — the reply after the tool result is.
            # shortcut: a tool call followed by no audio at all leaves the LED in
            # `thinking` until the firmware's own watchdog; add an
            # LLMFullResponseEndFrame fallback if that shows up in practice.
            await self._set_phase("thinking" if self._tool_pending else "idle")
        return False

    async def _on_device(self, msg: dict):
        kind = msg.get("type")
        if kind == "ping":
            await self._send({"type": "pong"})
        elif kind == "start":
            # Wake word, barge-in included: cancel any reply still playing. No
            # `listening` echo — to the firmware that means "speech heard" and
            # disarms its no-speech watchdog.
            self._turn = msg.get("turn")
            self._phase = None
            self._tool_pending = False
            await self._send_event(ResponseCancelEvent())
            await self.broadcast_interruption()
        elif kind == "interrupt":
            await self._send_event(ResponseCancelEvent())
            # The mic closed mid-segment ("стоп" included); drop it so it
            # neither becomes a reply nor prefixes the next turn's speech. Not
            # on `start`: the new turn's first audio may already be in there.
            await self._send_event(InputAudioBufferClearEvent())
            await self.broadcast_interruption()
            await self._set_phase("idle")

    async def _set_phase(self, value: str):
        if not self.in_turn or value == self._phase:
            return
        self._phase = value
        msg = {"type": "phase", "value": value}
        if self._turn is not None:
            msg["turn"] = self._turn
        await self._send(msg)

    async def _send(self, msg: dict):
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg))


def token_ok(header: str | None) -> bool:
    expected = os.environ["VA_DEVICE_TOKEN"]
    token = header.removeprefix("Bearer ").strip() if header and header.startswith("Bearer ") else ""
    return bool(token) and hmac.compare_digest(token.encode(), expected.encode())


async def ha_tools():
    ha_url, ha_token = os.environ.get("HA_URL"), os.environ.get("HA_TOKEN")
    if not (ha_url and ha_token):
        logger.warning("HA_URL/HA_TOKEN not set, running without HA tools")
        return None
    mcp = MCPClient(
        server_params=StreamableHttpParameters(
            url=f"{ha_url.rstrip('/')}/api/mcp",
            headers={"Authorization": f"Bearer {ha_token}"},
        )
    )
    try:
        return await mcp.tools()
    except Exception as e:
        # The voice path must survive HA being down — talk without tools.
        logger.error(f"HA MCP unavailable, running without tools: {e}")
        await mcp.close()
        return None


async def run_bot(websocket: WebSocket):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=VoicePeSerializer(),
        ),
    )
    model = os.environ.get("OPENAI_REALTIME_MODEL")
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            **({"model": model} if model else {}),
            system_instruction=os.environ.get("PIPECAT_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
            session_properties=SessionProperties(
                # Session-level tools go out on connect, before the first turn.
                tools=await ha_tools(),
                audio=AudioConfiguration(
                    output=AudioOutput(voice=os.environ.get("OPENAI_REALTIME_VOICE", "marin"))
                ),
            ),
        ),
    )
    context = LLMContext()
    # shortcut: private fields, re-check on a Pipecat bump. The service treats
    # its first context frame as "set up the conversation and respond", which
    # on a connection with no LLMRunFrame (we don't want the bot to talk on
    # connect) lands on the first tool result: the output never reaches OpenAI
    # and the whole history is pasted in as one user text. Pre-seeding makes
    # every later context frame take the normal tool-result path.
    llm._context = context
    llm._llm_needs_conversation_setup = False
    user_agg, assistant_agg = LLMContextAggregatorPair(context)
    device = DeviceProtocol(llm.send_client_event)
    pipeline = Pipeline(
        [transport.input(), user_agg, llm, device, transport.output(), assistant_agg]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=PIPELINE_RATE, audio_out_sample_rate=PIPELINE_RATE
        ),
        # A speaker holds its socket open for days; idling between wakes is normal.
        idle_timeout_secs=None,
        # RTVI is Pipecat's own client protocol; the speaker speaks ours.
        enable_rtvi=False,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        await worker.cancel()

    @worker.event_handler("on_pipeline_error")
    async def on_error(_worker, frame):
        # The realtime service reports a dead OpenAI session (error event,
        # dropped socket) only as a non-permanent ErrorFrame and never
        # reconnects, so the pipeline would idle on silently. Drop the speaker
        # instead: its reconnect gets a fresh session. A failing tool handler
        # (APPLICATION) is the model's problem, not the session's.
        if frame.processor is llm and frame.category is not ErrorCategory.APPLICATION:
            logger.error(f"OpenAI session failed, dropping the speaker: {frame.error}")
            await worker.cancel()

    async def recycle_before_session_cap():
        # OpenAI ends a Realtime session at 60 min and Pipecat doesn't
        # reconnect. Drop the speaker between turns before that; its reconnect
        # gets a fresh session (and re-fetches HA tools if HA was down).
        await asyncio.sleep(SESSION_RECYCLE_SECS)
        for _ in range(SESSION_GRACE_SECS):
            if not device.in_turn:
                break
            await asyncio.sleep(1)
        logger.info("recycling speaker connection ahead of OpenAI's session cap")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    recycle = asyncio.create_task(recycle_before_session_cap())
    try:
        await runner.run()
    finally:
        recycle.cancel()


app = FastAPI()


@app.websocket("/voice")
async def voice(websocket: WebSocket):
    await websocket.accept()
    if not token_ok(websocket.headers.get("authorization")):
        await websocket.close(4401, "unauthorized")
        return
    await websocket.send_text(
        json.dumps({"type": "hello", "proto": PROTO_VERSION, "audioOut": "pcm", "wakeChime": True})
    )
    logger.info("speaker connected")
    await run_bot(websocket)
    logger.info("speaker disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "3002")))
