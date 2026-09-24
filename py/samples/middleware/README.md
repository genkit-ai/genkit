# Middleware

One class on `ai.generate(..., use=[...])`. This one redacts emails
before the model sees the ticket.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```
