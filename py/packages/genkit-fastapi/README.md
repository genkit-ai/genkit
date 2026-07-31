# Genkit FastAPI Plugin

Serve Genkit flows as FastAPI endpoints.

## Installation

```bash
uv add genkit-fastapi genkit-google-genai
```

## Usage

```python
from fastapi import FastAPI
from genkit import Genkit
from genkit_fastapi import genkit_fastapi_handler
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')
app = FastAPI()


@app.post('/chat', response_model=None)
@genkit_fastapi_handler(ai)
@ai.flow()
async def chat_flow(prompt: str) -> str:
    response = await ai.generate(prompt=prompt)
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

The handler automatically supports streaming when the client sends `Accept: text/event-stream`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"data": "Tell me a joke"}'
```
