# Google Deep Research

Start a background Deep Research job with `generate_operation()`, poll with `check_operation()`, and read the finished report from `operation.output`.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore the flow in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Flow: `deep_research`.

Supported models (set `model` in flow input):

- `googleai/deep-research-preview-04-2026`
- `googleai/deep-research-max-preview-04-2026`
- `googleai/deep-research-pro-preview-12-2025`

Deep Research runs as a long-running background model — expect several minutes before `operation.done` is true.
