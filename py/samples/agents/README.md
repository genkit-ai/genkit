# Agents

One `Agent`, a `store=`, and `send` / `resume`. These samples use
`InMemorySessionStore` so they run on `GEMINI_API_KEY` alone. When you
deploy, pass `FirestoreSessionStore()` from `genkit-google-cloud` in the
same slot.

```bash
cd py/samples/agents
uv sync

genkit start -- uv run basic/01_define_agent_with_store.py
```

The numbered files in `basic/` are one idea each. Start at `01`.

`testapp/` is a FastAPI process with those agents mounted. See
[`testapp/README.md`](testapp/README.md).
