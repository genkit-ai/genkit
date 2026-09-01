# Genkit FastAPI Plugin

Expose Genkit flows and agents as HTTP endpoints in a FastAPI application.

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

Use `serve_agent` to expose an agent as FastAPI routes (including `/getSnapshot` and `/abort` endpoints when session state storage is enabled):

```python
from fastapi import FastAPI
from genkit import Genkit
from genkit_fastapi import serve_agent
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
