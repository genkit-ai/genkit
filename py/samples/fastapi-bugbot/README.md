# FastAPI BugBot

A small FastAPI app that reviews code for security, bug, and style issues.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

- API: http://localhost:8080
- Swagger: http://localhost:8080/docs

```bash
curl -X POST http://localhost:8080/review \
  -H "Content-Type: application/json" \
  -d '{"code":"eval(user_input)","language":"python"}'
```

To inspect the underlying flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

- Dev UI: http://localhost:4000
