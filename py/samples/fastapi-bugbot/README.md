# BugBot

A FastAPI app that reviews a snippet for security, bugs, and style.
Three Gemini calls in parallel, structured JSON back.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

```bash
curl -X POST http://localhost:8080/review \
  -H "Content-Type: application/json" \
  -d '{"data":{"code":"eval(user_input)","language":"python"}}'
```

Swagger is http://localhost:8080/docs.
