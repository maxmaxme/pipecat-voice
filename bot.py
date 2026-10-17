"""Pipecat behind the Voice PE speaker's `/voice` WebSocket protocol (proto 2).

The voice front-end only: audio, turns, LED phases. Everything the assistant
knows and can do comes from voice-assistant's `/mcp`, called with the speaker's
own device token — that authenticates the speaker, and returns its tools (Home
Assistant + memory, reminders, time, …) and its system prompt. Same wire shape
as voice-assistant's realtime bridge, so pointing a speaker's `va_url` here is
the only device-side change. No multi-speaker wake arbitration.
"""

import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable

import aiohttp
import uvicorn
from fastapi import FastAPI, WebSocket
from loguru import logger
from mcp.client.session_group import StreamableHttpParameters
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.mcp_service import MCPClient
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    ClientEvent,
    InputAudioBufferClearEvent,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    Reasoning,
    ResponseCancelEvent,
    SemanticTurnDetection,
    SessionProperties,
    TurnDetection,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.errors import ErrorCategory
from pipecat.workers.runner import WorkerRunner

# Pipecat logs at DEBUG by default, which dumps the system prompt — and with it
# the speaker's memory profile — into the container log. INFO keeps turns,
# transcripts and tool results.
logger.remove()
# Every line names its speaker (the client IP, bound per connection in voice()).
logger.configure(extra={"speaker": "-"})
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "{extra[speaker]} | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)
# Own variable rather than LOG_LEVEL, which other services sharing the
# environment may set in a form loguru rejects. An unknown level falls back to
# INFO instead of failing the import.
_level = os.environ.get("PIPECAT_LOG_LEVEL", "INFO").upper()
try:
    logger.add(sys.stderr, level=_level, format=LOG_FORMAT)
except ValueError:
    logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
    logger.warning(f"unknown PIPECAT_LOG_LEVEL {_level!r}, using INFO")

PROTO_VERSION = 2
DEVICE_MIC_RATE = 16000
# OpenAI Realtime only takes/gives PCM16 @ 24 kHz, and the speaker plays 24 kHz.
PIPELINE_RATE = 24000

# voice-assistant's voice MCP; speaker-scoped by the device token we forward.
VA_MCP_URL = os.environ.get("VA_MCP_URL", "http://voice-assistant:3000/mcp")
VA_PROMPTS = ("voice-system", "tools/wait-for-user", "tools/request-follow-up")
VA_CONFIG_URI = "voice://realtime-config"

# A follow-up turn has no `start`: the device just streams once its window
# opens. Accept speech that long after our idle (window + reply drain, the
# firmware's 1.5 s echo guard and the chime).
FOLLOW_UP_SLACK_S = 5
# Question window when voice-assistant's config doesn't say (requestFollowUpMs,
# followUpChime).
QUESTION_WINDOW_MS = 4_000
QUESTION_CHIME = True

# An OpenAI session that fails this soon after connect will most likely fail
# again on reconnect (bad key, no quota, a panel setting OpenAI rejects): keep
# the dead connection that long instead of a reconnect storm.
EARLY_FAILURE_HOLD_S = 60

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


class TranscriptLog(FrameProcessor):
    """Logs what the speaker heard. Sits between the user aggregator and the LLM:
    the realtime service sends user transcripts upstream, and they arrive after
    the turn was already handed to OpenAI, so the aggregator's own events can
    miss them."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            logger.info(f"user → {frame.text}")
        await self.push_frame(frame, direction)


class DeviceProtocol(FrameProcessor):
    """Device control messages → pipeline actions; turn frames → device phases.

    Sits between the LLM and transport.output(): sees the user-turn frames
    downstream and the output transport's Bot*SpeakingFrame upstream copies.
    """

    def __init__(
        self,
        send_event: Callable[[ClientEvent], Awaitable[None]],
        follow_up_ms: int = QUESTION_WINDOW_MS,
        follow_up_chime: bool = QUESTION_CHIME,
    ):
        super().__init__()
        # Mic window after a reply that asked something (request_follow_up, or
        # a reply ending in "?"); 0 = never. The chime is the speaker's "your
        # turn" cue before it opens.
        self._follow_up_ms = follow_up_ms
        self._follow_up_chime = follow_up_chime
        # Raw OpenAI Realtime client events: with server VAD Pipecat's
        # interruption only truncates the playing item, so cancelling the
        # generation itself is on us.
        self._send_event = send_event
        self._turn: int | None = None
        # `idle` = outside a turn; `start` opens one with phase None.
        self._phase: str | None = "idle"
        self._tool_pending = False
        # The reply's spoken text since the last `thinking`, and whether the
        # model called request_follow_up — either can open a question window.
        self._reply_text = ""
        self._question = False
        # time.monotonic() until which speech without `start` continues the
        # turn (the device's follow-up window).
        self._follow_up_until = 0.0
        # Idle reset: whether this connection has had a turn, and since when
        # the device has been quiet.
        self._had_turn = False
        self._idle_since = time.monotonic()

    @property
    def in_turn(self) -> bool:
        """Between the device's `start` and our `idle`. Phases and reply audio
        outside a turn (a cancelled reply's tail, a barge-in's BotStopped) are
        dropped, not sent to the speaker."""
        return self._phase != "idle"

    @property
    def busy(self) -> bool:
        """In a turn, or the device's follow-up window may still continue one."""
        return self.in_turn or time.monotonic() < self._follow_up_until

    def idle_for(self) -> float:
        """Seconds of quiet since the last turn; 0 before any turn or while busy."""
        if not self._had_turn or self.busy:
            return 0.0
        return time.monotonic() - max(self._idle_since, self._follow_up_until)

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
        if isinstance(frame, TTSTextFrame):
            self._reply_text += frame.text
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._tool_pending = True
        elif isinstance(frame, UserStartedSpeakingFrame):
            if not self.in_turn and time.monotonic() < self._follow_up_until:
                # Speech in the follow-up window continues the same turn id.
                self._phase = None
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
            if self._tool_pending:
                await self._set_phase("thinking")
            else:
                await self._end_reply()
        return False

    async def _end_reply(self):
        asked = self._question or self._reply_text.rstrip().endswith("?")
        if asked and self._follow_up_ms > 0:
            # Sent right before the idle: the device latches it and opens the
            # mic once the reply has drained.
            await self._send_turn(
                {"type": "follow_up", "ms": self._follow_up_ms, "chime": self._follow_up_chime}
            )
            self._follow_up_until = time.monotonic() + self._follow_up_ms / 1000 + FOLLOW_UP_SLACK_S
        await self._set_phase("idle")

    async def request_follow_up(self):
        """Tool: the model is asking the user something — open the mic after."""
        self._question = True

    async def wait_for_user(self):
        """Tool: the audio was noise/echo — end the turn silently, no follow-up."""
        await self._set_phase("idle")

    async def _on_device(self, msg: dict):
        kind = msg.get("type")
        if kind != "ping":
            logger.info(f"device {kind} {msg.get('turn', '')}".rstrip())
        if kind == "ping":
            await self._send({"type": "pong"})
        elif kind == "start":
            # Wake word, barge-in included: cancel any reply still playing. No
            # `listening` echo — to the firmware that means "speech heard" and
            # disarms its no-speech watchdog.
            self._turn = msg.get("turn")
            self._had_turn = True
            self._phase = None
            self._tool_pending = False
            self._follow_up_until = 0.0
            await self._send_event(ResponseCancelEvent())
            await self.broadcast_interruption()
        elif kind == "interrupt":
            await self._send_event(ResponseCancelEvent())
            # The mic closed mid-segment ("стоп" included); drop it so it
            # neither becomes a reply nor prefixes the next turn's speech. Not
            # on `start`: the new turn's first audio may already be in there.
            await self._send_event(InputAudioBufferClearEvent())
            await self.broadcast_interruption()
            self._follow_up_until = 0.0
            await self._set_phase("idle")

    async def _set_phase(self, value: str):
        if not self.in_turn or value == self._phase:
            return
        self._phase = value
        if value == "idle":
            self._idle_since = time.monotonic()
        if value in ("listening", "thinking"):
            self._reply_text = ""
        if value == "listening":
            self._question = False
        await self._send_turn({"type": "phase", "value": value})

    async def _send_turn(self, msg: dict):
        """Turn-scoped message: stamped with the device's turn id (proto 2)."""
        if self._turn is not None:
            msg["turn"] = self._turn
        logger.info(f"→ {msg['type']} {msg.get('value', msg.get('ms', ''))}")
        await self._send(msg)

    async def _send(self, msg: dict):
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg))


class Unauthorized(Exception):
    """voice-assistant doesn't know this speaker token."""


def bearer(header: str | None) -> str | None:
    if not header or not header.startswith("Bearer "):
        return None
    return header.removeprefix("Bearer ").strip() or None


async def va_setup(token: str) -> tuple[dict[str, str], dict]:
    """The speaker's prompts and realtime config from voice-assistant; also its
    authentication.

    Plain JSON-RPC against the stateless MCP endpoint rather than the MCP SDK
    client: the SDK folds HTTP 401 into a generic error, and telling "unknown
    speaker" (close 4401) from "voice-assistant down" is the point here.
    """
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as http:

        async def call(method: str, params: dict) -> dict:
            body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            async with http.post(VA_MCP_URL, json=body) as resp:
                if resp.status == 401:
                    raise Unauthorized()
                resp.raise_for_status()
                return await resp.json()

        *prompt_replies, reply = await asyncio.gather(
            *(call("prompts/get", {"name": name}) for name in VA_PROMPTS),
            call("resources/read", {"uri": VA_CONFIG_URI}),
        )
    prompts = {}
    for name, prompt in zip(VA_PROMPTS, prompt_replies):
        if "result" not in prompt:
            raise RuntimeError(f"prompt {name!r}: {prompt.get('error')}")
        prompts[name] = prompt["result"]["messages"][0]["content"]["text"]
    if "result" not in reply:
        # An older voice-assistant without the resource: run on the defaults.
        logger.warning(f"no realtime config from voice-assistant: {reply.get('error')}")
        return prompts, {}
    return prompts, json.loads(reply["result"]["contents"][0]["text"])


def opt(cfg: dict, key: str, default):
    """A config value, or `default` when voice-assistant left it out or null."""
    value = cfg.get(key)
    return default if value is None else value


def audio_config(cfg: dict) -> AudioConfiguration:
    """The Realtime audio settings from voice-assistant's Realtime page."""
    mode = str(opt(cfg, "turnDetection", "server_vad"))
    if mode.startswith("semantic_"):
        turn = SemanticTurnDetection(eagerness=mode.removeprefix("semantic_"))
    else:
        # voice-assistant's tuned server VAD: 0.3 hears quiet far-field speech
        # the default 0.5 misses; 900 ms keeps a mid-sentence pause from
        # splitting the turn.
        turn = TurnDetection(threshold=0.3, silence_duration_ms=900)
    # Default off: the XMOS DSP already cleans the mic, and far_field on top
    # ate short phrases and word onsets.
    noise = opt(cfg, "noiseReduction", "off")
    transcription = None
    if cfg.get("transcription"):
        # A second (paid) STT pass, only for the transcript log.
        transcription = InputAudioTranscription(
            model=opt(cfg, "transcriptionModel", "gpt-4o-mini-transcribe"),
            language=cfg.get("language") or None,
        )
    return AudioConfiguration(
        input=AudioInput(
            turn_detection=turn,
            noise_reduction=None if noise == "off" else InputAudioNoiseReduction(type=noise),
            transcription=transcription,
        ),
        output=AudioOutput(voice=opt(cfg, "voice", "marin")),
    )


async def va_tools(token: str) -> list[FunctionSchema]:
    """The speaker's tools (HA + voice-assistant's own) with call handlers."""
    mcp = MCPClient(
        server_params=StreamableHttpParameters(
            url=VA_MCP_URL, headers={"Authorization": f"Bearer {token}"}
        )
    )
    try:
        return list((await mcp.tools()).standard_tools)
    except Exception as e:
        # Talking without tools beats a dead speaker.
        logger.error(f"voice-assistant tools unavailable, running without them: {e}")
        await mcp.close()
        return []


async def run_bot(websocket: WebSocket, token: str, prompts: dict[str, str], cfg: dict):
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=VoicePeSerializer(),
        ),
    )
    device = DeviceProtocol(
        lambda event: llm.send_client_event(event),
        follow_up_ms=int(opt(cfg, "requestFollowUpMs", QUESTION_WINDOW_MS)),
        follow_up_chime=bool(opt(cfg, "followUpChime", QUESTION_CHIME)),
    )

    async def wait_for_user(params: FunctionCallParams):
        await device.wait_for_user()
        # run_llm=False: a flow-control call must not trigger another response.
        await params.result_callback(None, properties=FunctionCallResultProperties(run_llm=False))

    async def request_follow_up(params: FunctionCallParams):
        await device.request_follow_up()
        await params.result_callback(None, properties=FunctionCallResultProperties(run_llm=False))

    tools = [
        FunctionSchema("wait_for_user", prompts["tools/wait-for-user"], {}, [], handler=wait_for_user),
        FunctionSchema(
            "request_follow_up", prompts["tools/request-follow-up"], {}, [], handler=request_follow_up
        ),
        *await va_tools(token),
    ]
    effort = cfg.get("reasoningEffort")
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            **({"model": cfg["model"]} if cfg.get("model") else {}),
            system_instruction=prompts["voice-system"],
            session_properties=SessionProperties(
                # Session-level tools go out on connect, before the first turn.
                tools=ToolsSchema(standard_tools=tools),
                audio=audio_config(cfg),
                reasoning=Reasoning(effort=effort) if effort else None,
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
    pipeline = Pipeline(
        [transport.input(), user_agg, TranscriptLog(), llm, device, transport.output(), assistant_agg]
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

    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_text(_aggregator, message):
        logger.info(f"assistant → {message.content}{' (interrupted)' if message.interrupted else ''}")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(_transport, _client):
        await worker.cancel()

    connected = time.monotonic()
    drop_at: float | None = None

    @worker.event_handler("on_pipeline_error")
    async def on_error(_worker, frame):
        # The realtime service reports a dead OpenAI session (error event,
        # dropped socket) only as a non-permanent ErrorFrame and never
        # reconnects, so the pipeline would idle on silently. Drop the speaker
        # instead (the recycle loop does it): its reconnect gets a fresh
        # session. A failing tool handler (APPLICATION) is the model's problem,
        # not the session's.
        nonlocal drop_at
        if frame.processor is not llm or frame.category is ErrorCategory.APPLICATION or drop_at:
            return
        early = time.monotonic() - connected < EARLY_FAILURE_HOLD_S
        drop_at = time.monotonic() + (EARLY_FAILURE_HOLD_S if early else 0)
        hold = f", holding {EARLY_FAILURE_HOLD_S} s first" if early else ""
        logger.error(f"OpenAI session failed, dropping the speaker{hold}: {frame.error}")

    idle_reset_s = int(opt(cfg, "idleResetMs", 90_000)) / 1000

    async def recycle():
        # Drop the speaker to start over; its reconnect (≈1 s) gets a fresh
        # OpenAI session and re-reads tools, prompt (with new memory) and config
        # from voice-assistant. Two reasons:
        # - idle reset: a voice conversation shouldn't carry context across a
        #   quiet stretch (the panel's idle reset, 0 = off);
        # - OpenAI ends a Realtime session at 60 min and Pipecat doesn't
        #   reconnect: go between turns at 50, mid-turn at 58 at the latest.
        while True:
            await asyncio.sleep(1)
            age = time.monotonic() - connected
            if drop_at is not None and time.monotonic() >= drop_at:
                reason = "OpenAI session failed"
            elif idle_reset_s and device.idle_for() >= idle_reset_s:
                reason = "idle reset"
            elif age >= SESSION_RECYCLE_SECS and (
                not device.busy or age >= SESSION_RECYCLE_SECS + SESSION_GRACE_SECS
            ):
                reason = "OpenAI session cap"
            else:
                continue
            logger.info(f"recycling speaker connection: {reason}")
            await worker.cancel()
            return

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    recycler = asyncio.create_task(recycle())
    try:
        await runner.run()
    finally:
        recycler.cancel()


app = FastAPI()

# Each speaker token's last good prompts + config, so a voice-assistant restart
# (or outage) doesn't silence a speaker it already authorized: it runs on these,
# without tools, until voice-assistant is back. In-memory only.
last_setup: dict[str, tuple[dict[str, str], dict]] = {}


@app.websocket("/voice")
async def voice(websocket: WebSocket):
    await websocket.accept()
    with logger.contextualize(speaker=websocket.client.host if websocket.client else "?"):
        await serve_speaker(websocket)


async def serve_speaker(websocket: WebSocket):
    token = bearer(websocket.headers.get("authorization"))
    try:
        if not token:
            raise Unauthorized()
        prompts, cfg = await va_setup(token)
        last_setup[token] = (prompts, cfg)
    except Unauthorized:
        await websocket.close(4401, "unauthorized")
        return
    except Exception as e:
        if token not in last_setup:
            logger.error(f"voice-assistant unreachable: {e!r}")
            await websocket.close(1011, "voice-assistant unavailable")
            return
        logger.warning(f"voice-assistant unreachable ({e!r}), using its last prompt and config")
        prompts, cfg = last_setup[token]
    await websocket.send_text(
        json.dumps(
            {
                "type": "hello",
                "proto": PROTO_VERSION,
                "audioOut": "pcm",
                "wakeChime": bool(opt(cfg, "wakeChime", True)),
            }
        )
    )
    logger.info("speaker connected")
    logger.info(f"realtime config: {cfg}")
    await run_bot(websocket, token, prompts, cfg)
    logger.info("speaker disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "3002")))
