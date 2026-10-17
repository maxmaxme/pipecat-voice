import asyncio

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
import json

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

import bot


def test_serializer_resamples_mic_and_passes_reply_audio():
    s = bot.VoicePeSerializer()
    mic = asyncio.run(s.deserialize(b"\x00\x00" * 1600))  # 100 ms @ 16 kHz
    assert isinstance(mic, InputAudioRawFrame) and mic.sample_rate == 24000
    out = OutputAudioRawFrame(audio=b"\x01\x02" * 10, sample_rate=24000, num_channels=1)
    assert asyncio.run(s.serialize(out)) == out.audio


def test_serializer_control_messages():
    s = bot.VoicePeSerializer()
    msg = asyncio.run(s.deserialize('{"type":"start","turn":3}'))
    assert isinstance(msg, InputTransportMessageFrame) and msg.message == {"type": "start", "turn": 3}
    assert asyncio.run(s.deserialize("not json")) is None
    assert asyncio.run(s.deserialize("[1]")) is None


def test_bearer():
    assert bot.bearer("Bearer abc") == "abc"
    assert bot.bearer("Bearer ") is None
    assert bot.bearer("abc") is None
    assert bot.bearer(None) is None


CONFIG = {"noiseReduction": "off", "requestFollowUpMs": 4000, "voice": "marin"}


def fake_va(status=200, config=True):
    """voice-assistant's /mcp (prompts/get + resources/read), as a local aiohttp server."""

    async def handle(request):
        if request.headers.get("Authorization") != "Bearer good":
            return web.json_response({"error": "Unauthorized"}, status=401)
        rpc = await request.json()
        if rpc["method"] == "resources/read":
            if not config:
                result = {"error": {"code": -32601, "message": "Method not found"}}
                return web.json_response({"jsonrpc": "2.0", "id": 1, **result})
            text = json.dumps(CONFIG)
            result = {"contents": [{"uri": rpc["params"]["uri"], "mimeType": "application/json", "text": text}]}
        else:
            name = rpc["params"]["name"]
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": f"prompt {name}"}}]}
        return web.json_response({"jsonrpc": "2.0", "id": 1, "result": result}, status=status)

    app = web.Application()
    app.router.add_post("/mcp", handle)
    return TestServer(app)


def run_va_setup(token, monkeypatch, **kwargs):
    async def go():
        async with fake_va(**kwargs) as server:
            monkeypatch.setattr(bot, "VA_MCP_URL", str(server.make_url("/mcp")))
            return await bot.va_setup(token)

    return asyncio.run(go())


def test_va_setup_reads_prompts_and_config(monkeypatch):
    prompts, cfg = run_va_setup("good", monkeypatch)
    assert prompts == {name: f"prompt {name}" for name in bot.VA_PROMPTS}
    assert cfg == CONFIG


def test_va_setup_without_the_config_resource_runs_on_defaults(monkeypatch):
    _, cfg = run_va_setup("good", monkeypatch, config=False)
    assert cfg == {}


def test_va_setup_unknown_token_is_unauthorized(monkeypatch):
    with pytest.raises(bot.Unauthorized):
        run_va_setup("bad", monkeypatch)


def test_va_setup_server_error_is_not_unauthorized(monkeypatch):
    with pytest.raises(aiohttp.ClientResponseError):
        run_va_setup("good", monkeypatch, status=500)


def test_audio_config_defaults():
    audio = bot.audio_config({})
    assert audio.input.turn_detection.threshold == 0.3
    assert audio.input.turn_detection.silence_duration_ms == 900
    assert audio.input.noise_reduction is None
    assert audio.input.transcription is None
    assert audio.output.voice == "marin"


def test_audio_config_follows_the_panel():
    audio = bot.audio_config(
        {
            "turnDetection": "semantic_low",
            "noiseReduction": "far_field",
            "transcription": True,
            "transcriptionModel": "whisper-1",
            "language": "ru",
            "voice": "cedar",
        }
    )
    assert audio.input.turn_detection.type == "semantic_vad"
    assert audio.input.turn_detection.eagerness == "low"
    assert audio.input.noise_reduction.type == "far_field"
    assert (audio.input.transcription.model, audio.input.transcription.language) == ("whisper-1", "ru")
    assert audio.output.voice == "cedar"


class Recorder(bot.DeviceProtocol):
    """DeviceProtocol with the pipeline side stubbed: records what it would send."""

    def __init__(self):
        async def send_event(evt):
            self.events.append(evt.type)

        super().__init__(send_event)
        self.events: list[str] = []
        self.interruptions = 0
        self.sent: list[dict] = []

    async def _send(self, msg):
        self.sent.append(msg)

    async def broadcast_interruption(self):
        self.interruptions += 1

    def feed(self, *frames):
        """Runs frames through the logic; returns those that would be dropped."""

        async def go():
            return [f for f in frames if await self._handle(f)]

        return asyncio.run(go())

    def phases(self):
        return [(m["value"], m.get("turn")) for m in self.sent if m["type"] == "phase"]


def start(turn):
    return InputTransportMessageFrame(message={"type": "start", "turn": turn})


def test_normal_turn():
    d = Recorder()
    d.feed(start(1), UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame(), BotStoppedSpeakingFrame())
    assert d.phases() == [("listening", 1), ("thinking", 1), ("replying", 1), ("idle", 1)]
    assert not d.in_turn


def test_no_phases_outside_a_turn():
    d = Recorder()
    d.feed(BotStartedSpeakingFrame(), BotStoppedSpeakingFrame(), UserStartedSpeakingFrame())
    assert d.sent == []


def test_barge_in_does_not_idle_the_new_turn():
    d = Recorder()
    d.feed(start(1), UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame())
    d.feed(start(2))
    assert d.events == ["response.cancel", "response.cancel"] and d.interruptions == 2
    # The interrupted reply's BotStopped arrives after the new start.
    d.feed(BotStoppedSpeakingFrame(), UserStartedSpeakingFrame())
    assert d.phases()[-1] == ("listening", 2)
    assert ("idle", 2) not in d.phases()
    assert d.in_turn


def test_tool_preamble_keeps_the_turn_open():
    d = Recorder()
    d.feed(
        start(1),
        UserStartedSpeakingFrame(),
        UserStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),  # "сейчас включу…"
        FunctionCallsStartedFrame(function_calls=[]),
        BotStoppedSpeakingFrame(),
        BotStartedSpeakingFrame(),  # reply after the tool result
        BotStoppedSpeakingFrame(),
    )
    assert [p for p, _ in d.phases()] == ["listening", "thinking", "replying", "thinking", "replying", "idle"]


def test_interrupt_cancels_and_drops_the_late_reply():
    d = Recorder()
    d.feed(start(1), UserStartedSpeakingFrame(), InputTransportMessageFrame(message={"type": "interrupt"}))
    assert d.phases()[-1] == ("idle", 1)
    assert d.events == ["response.cancel", "response.cancel", "input_audio_buffer.clear"]
    # Tail of a reply that was already in flight: no phase, no audio.
    audio = OutputAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1)
    assert d.feed(BotStartedSpeakingFrame(), audio) == [audio]
    assert d.phases()[-1] == ("idle", 1)


def test_reply_audio_passes_inside_a_turn():
    d = Recorder()
    audio = OutputAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1)
    assert d.feed(audio) == [audio]  # before any wake word
    dropped = d.feed(start(1), BotStartedSpeakingFrame(), audio)
    assert all(f is not audio for f in dropped)


def reply(text):
    return TTSTextFrame(text, aggregated_by="sentence")


def asked(d, *spoken):
    d.feed(start(1), UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame())
    d.feed(*spoken, BotStoppedSpeakingFrame())


def follow_ups(d):
    return [m for m in d.sent if m["type"] == "follow_up"]


def test_plain_reply_opens_no_window():
    d = Recorder()
    asked(d, reply("Включила свет."))
    assert follow_ups(d) == [] and d.phases()[-1] == ("idle", 1)


def test_reply_ending_in_question_opens_a_window_before_idle():
    d = Recorder()
    asked(d, reply("В какой "), reply("комнате? "))
    assert d.sent[-2] == {"type": "follow_up", "ms": 4000, "chime": True, "turn": 1}
    assert d.sent[-1] == {"type": "phase", "value": "idle", "turn": 1}


def test_question_window_disabled_by_zero_ms():
    d = Recorder()
    d._follow_up_ms = 0
    asked(d, reply("В какой комнате?"))
    assert follow_ups(d) == [] and d.phases()[-1] == ("idle", 1)


def test_request_follow_up_tool_opens_a_window():
    d = Recorder()
    d.feed(start(1), UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame())
    asyncio.run(d.request_follow_up())
    d.feed(reply("Скажите, когда будете готовы."), BotStoppedSpeakingFrame())
    assert len(follow_ups(d)) == 1


def test_speech_in_the_window_continues_the_turn_without_start():
    d = Recorder()
    asked(d, reply("В какой комнате?"))
    audio = OutputAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1)
    dropped = d.feed(UserStartedSpeakingFrame(), UserStoppedSpeakingFrame(), BotStartedSpeakingFrame(), audio)
    assert dropped == []
    assert d.phases()[-3:] == [("listening", 1), ("thinking", 1), ("replying", 1)]


def test_speech_after_the_window_is_ignored():
    d = Recorder()
    asked(d, reply("В какой комнате?"))
    d._follow_up_until = 0.0  # window over
    d.feed(UserStartedSpeakingFrame())
    assert d.phases()[-1] == ("idle", 1)
    assert not d.busy


def test_wait_for_user_ends_the_turn_silently():
    d = Recorder()
    d.feed(start(1), UserStartedSpeakingFrame(), UserStoppedSpeakingFrame())
    asyncio.run(d.wait_for_user())
    assert d.phases()[-1] == ("idle", 1) and follow_ups(d) == []


def test_ping_works_outside_a_turn():
    d = Recorder()
    d.feed(InputTransportMessageFrame(message={"type": "ping"}))
    assert d.sent == [{"type": "pong"}]
