# Genkit Django Plugin

`genkit-plugin-django` exposes Genkit flows as HTTP endpoints in a Django application. It mirrors `genkit-plugin-flask` and `genkit-plugin-fastapi`: one decorator turns a `@ai.flow()` into a Django view that speaks the Genkit HTTP protocol (JSON envelope, optional SSE streaming, structured error responses).

## Install

```bash
pip install genkit-plugin-django
```

## Usage

```python
# myapp/views.py
from genkit import Genkit
from genkit.plugins.django import genkit_django_handler

ai = Genkit(plugins=[...])


@genkit_django_handler(ai)
@ai.flow()
async def chat(prompt: str) -> str:
    response = await ai.generate(prompt=prompt)
    return response.text
```

```python
# myproject/urls.py
from django.urls import path
from myapp.views import chat

urlpatterns = [
    path('chat/', chat),
]
```

The view requires Django's ASGI server (Django 4.1+):

```bash
uvicorn myproject.asgi:application
```

## Wire protocol

- Body: `{"data": <flow_input>}`. Missing `data` → 400.
- Streaming: `Accept: text/event-stream` or `?stream=true`. Each chunk emits `data: {"message": ...}\n\n`; completion emits `data: {"result": ...}\n\n`; on exception `error: {"error": ...}`.
- Non-stream: `{"result": <flow_output>}` on success; 500 with `HttpErrorWireFormat` JSON on exception.

## Context provider

```python
async def auth(request_data):
    return {'username': request_data.headers.get('authorization')}


@genkit_django_handler(ai, context_provider=auth)
@ai.flow()
async def chat(prompt, ctx):
    user = ctx.context.get('username')
    ...
```
