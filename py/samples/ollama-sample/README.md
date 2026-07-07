# Ollama

Run local LLM chat, streaming, tools, and embeddings through Genkit with Ollama.

Install Ollama from [ollama.com/download](https://ollama.com/download). Start the
server if it is not already running — it stays in the foreground, so use a
separate terminal (or rely on the Ollama app):

```bash
ollama serve
```

Then pull the sample models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Run the quick smoke test:

```bash
uv sync
uv run src/main.py
```

To explore all flows in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Then open [http://localhost:4000](http://localhost:4000) and try:

- `chat`
- `chat_stream`
- `tool_assistant`
- `embed_text`

The sample uses the default Ollama server at `http://127.0.0.1:11434`. To use a
different server, set `OLLAMA_HOST`. To use different local models, set
`OLLAMA_CHAT_MODEL` or `OLLAMA_EMBEDDER_MODEL`.
