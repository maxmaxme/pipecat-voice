# pipecat-voice

A [Pipecat](https://github.com/pipecat-ai/pipecat) voice front-end (OpenAI Realtime) for Voice PE speakers running the [va-direct firmware](https://github.com/maxmaxme/home-assistant-voice-pe). It is the server side of the firmware's `/voice` WebSocket protocol; [voice-assistant](https://github.com/maxmaxme/voice-assistant) supplies the tools, prompt and settings over MCP.

It only does the voice part: audio, turns, barge-in, LED phases, the question follow-up window. Everything the assistant knows and can do comes from voice-assistant's `/mcp`, called with the speaker's own device token: that authenticates the speaker and returns its tools (Home Assistant plus memory, reminders, time, …), its system prompt and its realtime settings (the `voice://realtime-config` resource: model, voice, reasoning effort, turn detection, noise reduction, transcription, question window, wake chime), so the web panel's Realtime page drives this session too. Settings are read on each connection. The server drops the speaker (it reconnects in about a second) after the panel's idle-reset time of quiet following a conversation, so a new conversation starts clean with fresh memory, and at the latest every ~50 min because OpenAI caps a Realtime session at 60.

Not implemented: the ambient after-every-reply follow-up window (only questions open the mic: `request_follow_up` or a reply ending in "?") and wake arbitration between several speakers.

## Env

| Var | Required | |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | |
| `VA_MCP_URL` | no | voice-assistant's voice MCP, default `http://voice-assistant:3000/mcp` |
| `PORT` | no | default `3002` |
| `PIPECAT_LOG_LEVEL` | no | default `INFO`; `DEBUG` also logs the system prompt, which carries the speaker's memory |

## Dev

```bash
uv sync
uv run pytest -q
OPENAI_API_KEY=... VA_MCP_URL=http://localhost:3000/mcp uv run python bot.py
```
