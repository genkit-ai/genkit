# Ollama

Chat, streaming, tools, and embeddings on your laptop. Same `generate()`
as the cloud plugins.

Install Ollama from [ollama.com/download](https://ollama.com/download),
then:

```bash
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text
uv sync
uv run src/main.py
```

Set `OLLAMA_HOST` if the server is not on `127.0.0.1:11434`.
