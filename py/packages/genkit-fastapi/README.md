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

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))
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

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))
app = FastAPI()


@ai.tool(description='Get current weather for a location')
async def get_weather(location: str) -> str:
    return f'Sunny in {location}'


weather_agent = ai.define_agent(
    name='weatherAgent',
    model=GoogleAI.gemini_model('gemini-flash-latest'),
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

### Streaming from Custom Routes (`stream.to_sse()`)

Call `.to_sse()` on any `StreamResponse` (`flow.stream()`, `ai.generate_stream()`) or `AgentTurn` (`chat.send_stream()`) to frame chunks, the final result, and errors as Genkit SSE events (`data: {"message": ...}`, `data: {"result": ...}`, or `data: {"error": ...}`) inside your own FastAPI route:

```python
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from genkit import FlowContext, Genkit

ai = Genkit(...)
app = FastAPI()


@ai.flow
async def deploy_service(service: str, ctx: FlowContext[str]) -> str:
    user_id = ctx.context['user_id']
    ctx.send_chunk(f'[{user_id}] building {service}')
    ctx.send_chunk(f'[{user_id}] rolling out {service}')
    return f'https://{service}.run.app'


@app.post('/deploy/stream')
async def deploy_stream(service: str, user: User = Depends(current_user)):
    stream = deploy_service.stream(service, context={'user_id': user.id})
    return StreamingResponse(stream.to_sse(), media_type='text/event-stream')
    # => data: {"message": "[u_42] building payments"}
    # => data: {"message": "[u_42] rolling out payments"}
    # => data: {"result": "https://payments.run.app"}


@app.post('/chat/stream')
async def chat_stream(prompt: str, user: User = Depends(current_user)):
    stream = ai.generate_stream(prompt=prompt, context={'user_id': user.id})
    return StreamingResponse(stream.to_sse(), media_type='text/event-stream')
```

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
