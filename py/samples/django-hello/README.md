# Django

A flow as a Django view.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run uvicorn myproject.asgi:application --port 8080
```

```bash
curl -N -X POST http://localhost:8080/chat \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -H 'Authorization: beginner-demo' \
  -d '{"data":{"name":"Mittens"}}'
```
