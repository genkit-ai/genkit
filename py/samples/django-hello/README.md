# Django Hello

Serve a Genkit flow through Django and stream the model response back to the client. Mirrors `flask-hello` and `fastapi-bugbot` but uses Django's ASGI server and the `genkit-plugin-django` adaptor.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run uvicorn myproject.asgi:application --port 8080
```

Then call it:

```bash
curl -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'Authorization: beginner-demo' \
  -d '{"data":{"name":"Mittens"}}'
```

To inspect the flow in Dev UI instead:

```bash
genkit start -- uv run uvicorn myproject.asgi:application --port 8080
```

## Streaming

Pass `Accept: text/event-stream` to consume the response chunk-by-chunk:

```bash
curl -N -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'Authorization: beginner-demo' \
  -d '{"data":{"name":"Mittens"}}'
```
