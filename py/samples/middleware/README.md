# Middleware

Intercept or modify model requests with `use=` on `ai.generate()`.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To inspect the flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Try `logging_demo` and `request_modifier_demo`.
