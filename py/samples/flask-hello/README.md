# Flask Hello

Serve a Genkit flow through Flask and stream the model response back to the client.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
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
genkit start -- uv run src/main.py
```
