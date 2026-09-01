# Samples

Snippets you can run tonight. Most need `GEMINI_API_KEY`.

```bash
cd py/samples/<name>
uv sync
uv run src/main.py
```

For traces while it runs:

```bash
genkit start -- uv run src/main.py
```

If you live in FastAPI, start with `fastapi-bugbot`. The other folders are
named after the feature they show.
