# pipecat-voice

A [Pipecat](https://github.com/pipecat-ai/pipecat) voice front-end (OpenAI Realtime) for Voice PE speakers running the [va-direct firmware](https://github.com/maxmaxme/home-assistant-voice-pe). It speaks the same `/voice` WebSocket protocol as [voice-assistant](https://github.com/maxmaxme/voice-assistant)'s realtime bridge, so switching a speaker is just its `va_url`.

It only does the voice part: audio, turns, barge-in, LED phases, the question follow-up window. Everything the assistant knows and can do comes from voice-assistant's `/mcp`, called with the speaker's own device token: that authenticates the speaker and returns its tools (Home Assistant plus memory, reminders, time, …) and its system prompt.

Not implemented: the ambient after-every-reply follow-up window (only questions open the mic: `request_follow_up` or a reply ending in "?") and wake arbitration between several speakers.

## Env

| Var | Required | |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | |
| `VA_MCP_URL` | no | voice-assistant's voice MCP, default `http://voice-assistant:3000/mcp` |
| `OPENAI_REALTIME_MODEL` | no | defaults to Pipecat's default realtime model |
| `OPENAI_REALTIME_VOICE` | no | default `marin` |
| `PORT` | no | default `3002` |
| `PIPECAT_LOG_LEVEL` | no | default `INFO`; `DEBUG` also logs the system prompt, which carries the speaker's memory |

## Dev

```bash
uv sync
uv run pytest -q
OPENAI_API_KEY=... VA_MCP_URL=http://localhost:3000/mcp uv run python bot.py
```
