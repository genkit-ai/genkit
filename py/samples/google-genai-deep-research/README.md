# Google Deep Research

Start a background Deep Research job with `generate_operation()`. Poll with
`check_operation()` from Dev UI — `uv run src/main.py` only starts the job so
it returns immediately.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore the flow (and poll) in Dev UI:

```bash
genkit start -- uv run src/main.py
```

Flow: `deep_research`.

Supported models (set `model` in flow input):

- `googleai/deep-research-preview-04-2026`
- `googleai/deep-research-max-preview-04-2026`
- `googleai/deep-research-pro-preview-12-2025`

This sample talks to Google AI Deep Research, not Vertex. A finished report
can take several minutes; use Dev UI to watch `operation.done`.
