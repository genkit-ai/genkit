# Agents (experimental)

Agents are experimental. Import the instance from `genkit.exp`:

```python
from genkit.exp import Genkit, InMemorySessionStore
```

One `Agent`, a `store=`, and `send` / `resume`. These samples use
`InMemorySessionStore` so they run on `GEMINI_API_KEY` alone. When you
deploy, pass `FirestoreSessionStore()` from `genkit_google_cloud.exp` in the
same slot.

```bash
cd py/samples/agents
uv sync

genkit start -- uv run basic/01_define_agent_with_store.py
```

The numbered files in `basic/` are one idea each. Start at `01`.

`testapp/` is a FastAPI process with those agents mounted. See
[`testapp/README.md`](testapp/README.md).
