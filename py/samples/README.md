# Genkit Samples

These samples are intentionally small and beginner-oriented. Each one tries to show one idea clearly instead of packing in every possible feature.

To run the default example once:

```bash
cd py/samples/<sample-name>
uv sync
uv run src/main.py
```

To open the Dev UI and run flows interactively:

```bash
cd py/samples/<sample-name>
uv sync
genkit start -- uv run src/main.py
```

Dev UI: http://localhost:4000. Most samples need `GEMINI_API_KEY`. See [plugins/README.md](../plugins/README.md) for provider setup.

## Samples

| Sample | What it shows |
|--------|----------------|
| `context` | Pass context through `generate()`, flows, and tools |
| `dynamic-tools` | Create a tool at runtime and trace plain functions |
| `evaluators` | Run simple custom evaluators with `genkit eval:run` |
| `fastapi-bugbot` | A small FastAPI app that reviews code |
| `flask-hello` | Expose Genkit flows through Flask |
| `gemini-code-execution` | Ask Gemini to write and run code |
| `gemini-context-caching` | Cache a large source document for follow-up prompts |
| `google-genai-media` | Speech, image, and video generation |
| `middleware` | Observe or modify model requests |
| `output-formats` | Text, enum, JSON, array, and JSONL outputs |
| `prompts` | `.prompt` files, variants, helpers, and streaming |
| `tool-interrupts` | Trivia (`respond_example.py`) and bank approval (`approval_example.py`) — interrupt + resume |
| `tracing` | Watch spans appear in real time |
| `vertexai-imagen` | Generate an image with Vertex AI Imagen |
