# Flows

A flow is typed, traced, and streamable. No model, no API key.

```bash
uv sync
uv run src/main.py
```

That runs two named steps and a short stream. Hang the same flows on
FastAPI with `serve_flow` (see `src/main.py`), or open Dev UI:

```bash
genkit start -- uv run src/main.py
```
