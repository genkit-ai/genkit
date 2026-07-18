# Genkit Agents Samples

Two sets of Python samples for the Genkit agents runtime, split by how you'd use them:

- **`basic/`** — small, single-file examples, one per concept (stores, interrupt/resume, custom state, artifacts, detach/abort, timeouts, branching, time-travel). Start here to learn the APIs.
- **`testapp/`** — a full agent app that integrates with the testapp in `js/testapps/agents`. Its `server.py` puts every agent behind an HTTP endpoint, so you can swap the Node backend out for this Python one and the existing web frontend keeps working unchanged. Use this to see the agents working end-to-end.

All examples require `GEMINI_API_KEY`.

## Run

```bash
cd py/samples/agents
uv sync

# a basic example in the Dev UI
genkit start -- uv run basic/01_define_agent_with_store.py

# the full testapp (Dev UI + HTTP API)
genkit start -- uv run testapp/server.py
```
