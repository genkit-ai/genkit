# Valkey RAG Sample

Indexes documents into Valkey, retrieves the top matches for a query via KNN
vector search, then uses an Ollama LLM to generate an answer grounded in the
retrieved context.

## Prerequisites

1. Valkey with the valkey-search module:

   ```bash
   docker run -d --name valkey-search -p 6379:6379 valkey/valkey-search:latest
   ```

2. Ollama with the required models:

   ```bash
   ollama pull nomic-embed-text
   ollama pull gemma4:e2b
   ```

## Run

```bash
uv sync
uv run src/main.py
```
