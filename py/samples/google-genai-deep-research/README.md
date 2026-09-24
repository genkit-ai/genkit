# Google Deep Research

Start a background Deep Research job with `generate_operation()`, then
`check_operation()` once. A finished report can take several minutes —
keep polling from Dev UI.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore the flow (and keep polling) in Dev UI:

```bash
genkit start -- uv run src/main.py
```

Flow: `deep_research`.

Supported models (set `model` in flow input):

- `googleai/deep-research-preview-04-2026`
- `googleai/deep-research-max-preview-04-2026`
- `googleai/deep-research-pro-preview-12-2025`

This sample talks to Google AI Deep Research, not Vertex. Antigravity and
Lyria 3 are ordinary `generate` models on the same plugin (`GoogleAI.antigravity_model`,
`GoogleAI.lyria_model`); they are not in this sample.
