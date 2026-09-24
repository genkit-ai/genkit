# Multipart tools

A tool either returns a value (we wrap it) or returns `response(...)` (the action result, with optional media).

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

In Dev UI:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and run `ask_about_the_lab`.
