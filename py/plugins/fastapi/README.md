# Genkit FastAPI Plugin

Serve Genkit flows as FastAPI endpoints.

## Installation

```bash
pip install genkit-plugin-fastapi
```

## Usage

from fastapi import FastAPI
from genkit import Genkit
from genkit.plugins.fastapi import genkit_fastapi_handler, genkit_lifespan
from genkit.plugins.google_genai import GoogleAI

app = FastAPI(lifespan=genkit_lifespan(ai))
ai = Genkit(plugins=[GoogleAI()])


@ai.flow()
async def chat_flow(prompt: str) -> str:
    response = await ai.generate(prompt=prompt)
    return response.text


@app.post('/chat')
@genkit_fastapi_handler(ai)
async def chat():
    return chat_flow

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
