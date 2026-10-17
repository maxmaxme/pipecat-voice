import asyncio
import json

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
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


def test_token_ok(monkeypatch):
    monkeypatch.setenv("VA_DEVICE_TOKEN", "abc")
    assert bot.token_ok("Bearer abc")
    assert not bot.token_ok("Bearer abd")
    assert not bot.token_ok("abc")
    assert not bot.token_ok(None)


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


def test_ping_works_outside_a_turn():
    d = Recorder()
    d.feed(InputTransportMessageFrame(message={"type": "ping"}))
    assert d.sent == [{"type": "pong"}]
