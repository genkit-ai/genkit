# Ollama

Run local LLM chat, streaming, tools, and embeddings through Genkit with Ollama.

Install Ollama from [ollama.com/download](https://ollama.com/download), then
start the server and pull the sample models:

```bash
ollama serve
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
