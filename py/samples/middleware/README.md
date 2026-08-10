# Middleware

Intercept or modify model requests with `use=` on `ai.generate()`.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
genkit start -- uv run src/main.py
```

That command should provide a link to the Dev UI, where you can manually trigger the sample's flows.

Try `logging_demo`, `request_modifier_demo`, and the `middleware_demo` prompt (also via `middleware_prompt_demo`).
