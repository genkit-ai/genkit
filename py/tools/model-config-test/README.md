# Model Config Testing Tool

A tool to test model performance across different models and configuration variations.

## Setup

1.  Ensure you have `uv` installed.
2.  Set your API keys (e.g., `GEMINI_API_KEY`).

## Usage

Run the tool:

```bash
uv run tools/model-config-test/model_performance_test.py --models googleai/gemini-2.0-flash
```

Or run the web interface:

```bash
cd py
uv run tools/model-config-test/server.py
```

## Features

-   **Model Discovery**: Automatically finds registered models.
-   **Config Discovery**: Inspects model schema to find parameters.
-   **Variations**: Tests min, max, midpoint, and default values.
-   **Report**: Generates a Markdown report with pass/fail stats and detailed results.
