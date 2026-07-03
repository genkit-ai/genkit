# Genkit FastAPI Plugin

Serve Genkit flows and agents as FastAPI endpoints.

## Installation

```bash
pip install genkit-plugin-fastapi
```

Both `serve_flow` and `serve_agent` return an `APIRouter` you mount with
`app.include_router`, so FastAPI's own `prefix` / `dependencies` / `tags` handle
the framework-level wiring and a flow and an agent read the same way.

## Serve a flow (`serve_flow`)

Turns one flow into a `POST` route. Takes `{"data": <input>}`, returns
`{"result": <output>}`.

```python
from fastapi import FastAPI
from genkit import Genkit
from genkit.plugins.fastapi import serve_flow
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
app = FastAPI()


@ai.flow()
async def chat_flow(prompt: str) -> str:
    response = await ai.generate(prompt=prompt)
    return response.text


app.include_router(serve_flow(chat_flow), prefix='/api')  # POST /api/chat_flow
```

## Serve an agent (`serve_agent`)

Builds the run-turn / getSnapshot / abort routes an agent's client (`remote_agent`
/ `AgentChat`) speaks. Pass a `context_provider` to authenticate and attach the
caller before the turn runs.

```python
from fastapi import FastAPI
from genkit.plugins.fastapi import serve_agent

app = FastAPI()
app.include_router(serve_agent(weather_agent), prefix='/api')  # POST /api/weatherAgent
```

`base_path` defaults to `/<name>`, so several agents (or flows) mount cleanly
under one prefix. Pass an explicit `base_path='/chat'` to override (or `''` for
the router root). Use FastAPI's own knobs for the rest:

```python
app.include_router(
    serve_agent(weather_agent, context_provider=auth),
    prefix='/api',
    dependencies=[Depends(rate_limit)],  # framework-level gate
    tags=['agent'],
)
```

## Running

```bash
# With Genkit Dev UI
genkit start -- uvicorn main:app --reload

# Production (no Dev UI)
uvicorn main:app
```

## Streaming

A flow route streams Server-Sent Events when the client sends `Accept: text/event-stream`:

```bash
curl -X POST http://localhost:8000/api/chat_flow \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"data": "Tell me a joke"}'
```
