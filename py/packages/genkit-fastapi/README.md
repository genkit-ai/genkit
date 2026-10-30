# Genkit FastAPI Plugin

Expose Genkit flows and agents as HTTP endpoints in a FastAPI application.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Installation

```bash
uv add genkit-fastapi genkit-google-genai
```

## Usage

### Serving Flows (`serve_flow`)

Use `serve_flow` to register a Genkit flow as a FastAPI endpoint:

```python
from fastapi import FastAPI
from genkit import Genkit
from genkit_fastapi import serve_flow
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))
app = FastAPI()


@ai.tool(description='Get current weather for a location')
async def get_weather(location: str) -> str:
    return f'Sunny in {location}'


@ai.flow()
async def chat_flow(prompt: str) -> str:
    response = await ai.generate(
        prompt=prompt,
        tools=[get_weather],
    )
    return response.text


# Mount flow endpoint at POST /api/chat_flow
app.include_router(serve_flow(chat_flow), prefix='/api')
```

### Serving Agents (`serve_agent`)

Use `serve_agent` to expose an agent as FastAPI routes (including `/getSnapshot` and `/abort` endpoints when session state storage is enabled). Agents are experimental, so `serve_agent` comes from `genkit_fastapi.exp`:

```python
from fastapi import FastAPI
from genkit.exp import Genkit
from genkit_fastapi.exp import serve_agent
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))
app = FastAPI()


@ai.tool(description='Get current weather for a location')
async def get_weather(location: str) -> str:
    return f'Sunny in {location}'


weather_agent = ai.define_agent(
    name='weatherAgent',
    model=GoogleAI.gemini('gemini-flash-latest'),
    system='You are a helpful weather assistant.',
    tools=[get_weather],
)

# Mount agent turn route at POST /api/weatherAgent (plus /getSnapshot and /abort companion endpoints)
app.include_router(serve_agent(weather_agent), prefix='/api')
```

### Custom Base Path & Dependency Injection

Customize the base path or resolve request context using FastAPI's dependency injection system:

```python
from fastapi import Depends, Header
from genkit_fastapi import serve_flow


async def user_context(authorization: str = Header(...)):
    return {'uid': parse_token(authorization)}


app.include_router(
    serve_flow(
        chat_flow,
        base_path='/chat',
        context_dependency=user_context,
    ),
    prefix='/api',
)
```

### Streaming from Custom Routes (`genkit.web.to_sse`)

`to_sse(handle)` frames any streaming handle as Genkit SSE events: a `data: {"message": ...}` per chunk, then `data: {"result": ...}`, or `data: {"error": ...}` if the run fails. It takes `flow.stream()`, `ai.generate_stream()`, `prompt.stream()` and `chat.send_stream()`, so your route keeps its own typed params and `Depends(...)` and still sends the frames `serve_flow` sends:

```python
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from genkit import FlowContext, Genkit
from genkit.web import to_sse

ai = Genkit(...)
app = FastAPI()


@ai.flow()
async def ask_chef(prompt: str, ctx: FlowContext[str]) -> str:
    stream = ai.generate_stream(prompt=prompt)
    async for chunk in stream:
        ctx.send_chunk(chunk.text)
    return (await stream.response).text


@app.post('/chat/stream')
async def chat_stream(prompt: str, user: User = Depends(current_user)):
    # 1. FastAPI owns the route: typed params, Depends, OpenAPI
    stream = ask_chef.stream(prompt, context={'user_id': user.id})

    # 2. Genkit owns the wire
    return StreamingResponse(to_sse(stream), media_type='text/event-stream')
    # => data: {"message":"Pad thai"}
    #    data: {"message":", no peanuts."}
    #    data: {"result":"Pad thai, no peanuts."}
```

> [!WARNING]
> The `result` frame is whatever the run returns, over SSE and JSON alike. Return `response.text` or `response.output` from a served flow, not the `ModelResponse`: its `request` field carries the system prompt, tool definitions, retrieved documents and history. The same goes for `to_sse(ai.generate_stream(...))`, whose result is the whole `ModelResponse`.

`to_sse(chat.send_stream(...))` frames agent turns the same way `serve_agent` does (`AgentStreamChunk` messages, then the `AgentOutput` result). Failures log on the `genkit.web._sse` logger; pass `logger=` to use your own.

### Handling Full Genkit Requests (`handle_genkit_request`)

When FastAPI owns the route signature and `Depends(...)` parameters (for authentication, rate limiting, or database sessions) and you want one endpoint to serve both JSON `POST` (`{"result": ...}`) and SSE streaming (`Accept: text/event-stream` or `?stream=true`), delegate the HTTP turn to `handle_genkit_request`:

```python
from fastapi import Depends, Request
from genkit_fastapi import handle_genkit_request


@app.post('/api/deploy', response_model=None)
async def deploy_endpoint(request: Request, user: User = Depends(current_user)):
    return await handle_genkit_request(
        request,
        action=deploy_service,
        context={'user_id': user.id},
    )
```

### Decorator Handler (`genkit_fastapi_handler`)

For custom route definitions, you can also use `@genkit_fastapi_handler`:

```python
from genkit_fastapi import genkit_fastapi_handler


@app.post('/custom-chat', response_model=None)
@genkit_fastapi_handler(ai)
@ai.flow()
async def custom_chat(prompt: str) -> str:
    response = await ai.generate(
        prompt=prompt,
        tools=[get_weather],
    )
    return response.text
```

## Running

```bash
# With Genkit Dev UI
genkit start -- uvicorn main:app --reload

# Production (no Dev UI)
uvicorn main:app
```

## Streaming

Endpoints automatically support streaming when the client sends `Accept: text/event-stream` or specifies `?stream=true`:

```bash
curl -X POST http://localhost:8000/api/chat_flow \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"data": "Tell me a joke"}'
```
