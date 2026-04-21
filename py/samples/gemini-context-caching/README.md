# Google Context Caching

Cache large docs so follow-up queries reuse context. Saves latency and tokens for RAG/summarization.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To explore it in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Try `ask_about_cached_document`.
