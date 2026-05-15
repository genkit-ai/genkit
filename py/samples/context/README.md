# Context Sample

Learn how to pass request-scoped data like user info through `ai.generate()`, flows, and tools without threading extra parameters everywhere.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore the flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and try:

- `context_in_generate`
- `context_in_flow`
- `context_propagation_chain`
