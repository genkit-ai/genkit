# Serving agents over FastAPI / HTTP

The backend examples in [`../backend/`](../backend/) call `agent.stream_bidi()` directly.
To expose the same agents over HTTP (for `GenkitChatTransport` / `useChat`), register
them on a FastAPI app — see [`../main.py`](../main.py).

## Minimal pattern

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from genkit import Genkit
from genkit.agent import InMemorySessionStore
from genkit.plugins.fastapi import genkit_fastapi_handler
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

weather_agent = ai.define_agent(
    name='weatherAgent',
    model='googleai/gemini-flash-latest',
    system='Weather assistant.',
    store=store,
)


@genkit_fastapi_handler(app, ai, path='/api/chat/weather')
async def chat_weather():
    return weather_agent
```

Run:

```bash
cd py/samples/agents
uv run uvicorn src.main:app --port 8080
```

## What the handler does (one HTTP request = one agent turn)

1. Parses `{ "data": AgentInput, "init": AgentInit }` from the POST body
2. `conn = await agent.stream_bidi(init)`
3. `await conn.send(data)` then `await conn.close()`
4. Streams SSE: `data: {"message": ...}` for each chunk, then `data: {"result": ...}`

Multi-turn over HTTP = **new POST each turn**, same `init.sessionId` (store) or
`init.state` (no store). That mirrors starting a fresh `stream_bidi` per turn in
the backend examples.

## Abort over HTTP

After a detached turn, clients call a store endpoint (see `main.py`):

```python
@app.post('/api/agents/long-task/abort')
async def abort_long_task(body: AbortBody):
    return await long_task_store.abort_snapshot(body.snapshot_id)
```

In-process equivalent: [`../backend/10_abort.py`](../backend/10_abort.py).
