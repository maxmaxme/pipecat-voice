# pipecat-voice

Experiment: a [Pipecat](https://github.com/pipecat-ai/pipecat) voice bot (OpenAI Realtime + Home Assistant MCP tools) that speaks the same `/voice` WebSocket protocol as [voice-assistant](https://github.com/maxmaxme/voice-assistant)'s realtime bridge. A Voice PE speaker running the [va-direct firmware](https://github.com/maxmaxme/home-assistant-voice-pe) can be switched to it by changing only its `va_url`.

Deliberately minimal: no follow-up listening window, no multi-speaker wake arbitration, no memory or DB-backed prompts.

## Env

| Var | Required | |
| --- | --- | --- |
| `OPENAI_API_KEY` | yes | |
| `VA_DEVICE_TOKEN` | yes | bearer token the speaker sends on the WS upgrade |
| `HA_URL`, `HA_TOKEN` | no | HA base URL + long-lived token; without them (or if HA is down) the bot runs with no tools |
| `OPENAI_REALTIME_MODEL` | no | defaults to Pipecat's default realtime model |
| `OPENAI_REALTIME_VOICE` | no | default `marin` |
| `PIPECAT_INSTRUCTIONS` | no | system prompt |
| `PORT` | no | default `3002` |

## Dev

```bash
uv sync
uv run pytest -q
VA_DEVICE_TOKEN=... OPENAI_API_KEY=... uv run python bot.py
```
