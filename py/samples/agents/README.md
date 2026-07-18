# Genkit Agents Samples

Single-file Python examples for the Genkit agents runtime, one per concept:
stores, interrupt/resume, custom state, artifacts, detach/abort, timeouts,
branching, and time-travel. Each example defines its agent inline, so you can
read one file top to bottom to learn an API.

All examples require `GEMINI_API_KEY`.

## Run

```bash
cd py/samples/agents
uv sync

# any example in the Dev UI
genkit start -- uv run basic/01_define_agent_with_store.py
```
