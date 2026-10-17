import asyncio
import json
import os

from pipecat.frames.frames import InputAudioRawFrame, InputTransportMessageFrame, OutputAudioRawFrame

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


def test_token_ok():
    os.environ["VA_DEVICE_TOKEN"] = "abc"
    assert bot.token_ok("Bearer abc")
    assert not bot.token_ok("Bearer abd")
    assert not bot.token_ok("abc")
    assert not bot.token_ok(None)
