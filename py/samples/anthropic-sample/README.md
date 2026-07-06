# Anthropic sample

Demonstrates generating plain-text and structured/JSON responses using Anthropic's Claude Opus models (`claude-opus-4-7` and `claude-opus-4-8`).

Requires `ANTHROPIC_API_KEY`.

## Setup

```bash
cd py/samples/anthropic-sample
uv sync
```

## Running

To run the flows in Dev UI:

```bash
genkit start -- uv run src/main.py
```

To run the CLI smoke test once:

```bash
uv run src/main.py
```
