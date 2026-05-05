# Genkit Python SDK — Framework Integration (FastAPI)

Two styles are supported side-by-side:

1. **Plain FastAPI endpoints** that call flows directly. Genkit was designed to work naturally in a FastAPI style async-first server environment.
2. **`@genkit_fastapi_handler`** for exposing flows through the Genkit protocol (used by the Dev UI). It hides the tricky parts — SSE framing, chunk-vs-final-frame encoding, error propagation, and `ActionRunContext` wiring — so your flow just `send_chunk`s and returns.

Define the flow with `@ai.flow()` so it's registered on the `Genkit` instance (tracing, telemetry, Dev UI visibility). The FastAPI route is just the transport.

```python
from fastapi import FastAPI
from genkit import ActionRunContext, Genkit
from genkit.plugins.fastapi import genkit_fastapi_handler
from genkit.plugins.google_genai import GoogleAI

app = FastAPI()
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.0-flash')

# Style 1: plain FastAPI endpoint, calls a Genkit flow
@ai.flow()
async def review_code(input: CodeInput) -> Analysis:
    response = await ai.generate(prompt=f'Review this code:\n{input.code}', output_schema=Analysis)
    return response.output

@app.post('/review')
async def review(input: CodeInput) -> Analysis:
    return await review_code(input)

# Style 2: stack the decorators — Dev UI compatible, streaming built in
@app.post('/flow/chat', response_model=None)
@genkit_fastapi_handler(ai)
@ai.flow()
async def chat(input: ChatInput, ctx: ActionRunContext) -> str:
    """Answer a question, streaming tokens as they arrive."""
    sr = ai.generate_stream(prompt=input.question)
    async for chunk in sr.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)   # emitted as an SSE frame to the client
    final = await sr.response            # aggregated ModelResponse
    return final.text
```

The decorator stack reads bottom-up: `@ai.flow()` registers the function as a flow (returns an `Action`), `@genkit_fastapi_handler(ai)` wraps it into a FastAPI handler that understands the Genkit protocol + SSE, and `@app.post(...)` mounts it on the route. 

Full sample: [`py/samples/fastapi-bugbot/src/main.py`](https://github.com/genkit-ai/genkit/blob/main/py/samples/fastapi-bugbot/src/main.py)

## Auth via `context_provider`

`genkit_fastapi_handler` takes an optional `context_provider` — a callable that inspects the raw request, returns a dict, and that dict becomes `ctx.context` inside the flow. Raise to reject the request (no flow invocation).

```python
from genkit import PublicError
from genkit.plugin_api import RequestData

async def auth_context(req: RequestData) -> dict[str, object]:
    """Verify Bearer token; attach user_id to ctx.context; reject on failure."""
    token = req.headers.get('Authorization', '').removeprefix('Bearer ').strip()
    if not token:
        raise PublicError(status='UNAUTHENTICATED', message='Missing Bearer token')
    try:
        claims = verify_jwt(token)        # your own JWT verification
    except InvalidTokenError as e:
        raise PublicError(status='UNAUTHENTICATED', message=str(e)) from e
    return {'user_id': claims['sub']}

@app.post('/flow/chat', response_model=None)
@genkit_fastapi_handler(ai, context_provider=auth_context)
@ai.flow()
async def chat(input: ChatInput, ctx: ActionRunContext) -> str:
    user_id = ctx.context['user_id']
    resp = await ai.generate(prompt=f'Reply to {user_id}: {input.question}')
    return resp.text
```

A few things worth knowing:

- **Raising `PublicError`** gives the caller a structured `{status, message}` error with a proper HTTP status. Raising a plain exception works too but surfaces as a generic 500.
- **The context flows transitively.** Anything you put in `ctx.context` is visible to tools the flow calls, to middleware, and in tracing — so `user_id` scope naturally through the call tree.
- **Async or sync** — both work. The handler awaits the result if you `return` a coroutine.

## Streaming

The client opts in via either:

- `Accept: text/event-stream` header, or
- `?stream=true` query param.

When set, the handler calls `flow.stream(...)` instead of `flow.run(...)` and returns a `StreamingResponse` over Server-Sent Events. Wire format is one SSE frame per chunk plus one final frame with the completed result:

```text
data: {"message":{"text":"Hello"}}

data: {"message":{"text":" world"}}

data: {"result":{"text":"Hello world","usage":{...}}}

```

If the flow errors, the last frame is `error: {"error":{...}}` instead.

### Frontend integration

Using `streamFlow` from the Genkit JS client directly:

```ts
import { streamFlow } from 'genkit/beta/client';

const { stream, output } = streamFlow<typeof reviewFlow>({
  url: '/flow/review',
  input: { code: source },
});

for await (const chunk of stream) {
  setText(prev => prev + chunk.text);
}
const final = await output;   // Analysis — resolves after the last SSE frame
```