"""Pipecat behind the Voice PE speaker's `/voice` WebSocket protocol (proto 2).

An experiment next to voice-assistant's own realtime bridge: same wire shape,
so pointing a speaker's `va_url` at this port is the only change. Deliberately
minimal — no follow-up window, no multi-speaker wake arbitration, no
memory/prompts DB.
"""

import hmac
import json
import os

import uvicorn
from fastapi import FastAPI, WebSocket
from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
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
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

PROTO_VERSION = 2
DEVICE_MIC_RATE = 16000
# OpenAI Realtime only takes/gives PCM16 @ 24 kHz, and the speaker plays 24 kHz.
PIPELINE_RATE = 24000

DEFAULT_INSTRUCTIONS = (
    "Ты голосовой ассистент умного дома. Отвечай по-русски, коротко, одним-двумя "
    "предложениями. Для управления устройствами используй инструменты Home Assistant."
)

# Pipeline turn frames → the device's LED phase.
PHASES: dict[type[Frame], str] = {
    UserStartedSpeakingFrame: "listening",
    UserStoppedSpeakingFrame: "thinking",
    BotStartedSpeakingFrame: "replying",
    BotStoppedSpeakingFrame: "idle",
}


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
            return InputTransportMessageFrame(message=json.loads(data))
        except json.JSONDecodeError:
            logger.warning(f"bad device message: {data!r}")
            return None


class DeviceProtocol(FrameProcessor):
    """Device control messages → pipeline actions; turn frames → device phases.

    Sits between the LLM and transport.output(): sees the user-turn frames
    downstream and the output transport's Bot*SpeakingFrame upstream copies.
    """

    def __init__(self):
        super().__init__()
        self._turn: int | None = None
        self._phase: str | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame):
            await self._on_device(frame.message)
            return
        phase = PHASES.get(type(frame))
        if phase:
            await self._set_phase(phase)
        await self.push_frame(frame, direction)

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
            await self.broadcast_interruption()
        elif kind == "interrupt":
            await self.broadcast_interruption()
            await self._set_phase("idle")

    async def _set_phase(self, value: str):
        if value == self._phase:
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
    user_agg, assistant_agg = LLMContextAggregatorPair(LLMContext())
    pipeline = Pipeline(
        [transport.input(), user_agg, llm, DeviceProtocol(), transport.output(), assistant_agg]
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

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


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
