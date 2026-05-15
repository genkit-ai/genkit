# Tracing

Spans show up in Dev UI as they start, not when they finish. For long flows with many steps.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To watch it in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Run `trace_steps_live` and watch the Traces tab.
