# Ollama Hello World

Local LLM inference, tools, vision, and embeddings with Genkit — all running
privately on your machine via [Ollama](https://ollama.com/).

## Prerequisites

- **Ollama** installed and running locally.

## Quick Start

The `run.sh` script handles everything — it installs dependencies, pulls
required models, and starts the Dev UI:

```bash
./run.sh
```

## Manual Setup

1. **Install Ollama**: Download from [ollama.com](https://ollama.com/download).
2. **Start the server**:
   ```bash
   ollama serve
   ```
3. **Pull models**:
   ```bash
   ollama pull gemma3:latest        # General generation
   ollama pull mistral-nemo:latest  # Tool calling
   ollama pull llava:latest         # Vision / image description
   ollama pull nomic-embed-text     # Embeddings for RAG
   ```
4. **Run the sample**:
   ```bash
   genkit start -- uv run src/main.py
   ```

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows** (uses gemma3):
   - [ ] `say_hi` — Simple generation
   - [ ] `say_hi_stream` — Streaming response
   - [ ] `say_hi_constrained` — Structured output (HelloSchema)
   - [ ] `generate_character` — Structured output (RpgCharacter)
   - [ ] `generate_code` — Code generation

3. **Test tools** (uses mistral-nemo):
   - [ ] `calculate_gablorken` — Tool calling demo
   - [ ] `currency_exchange` — Currency conversion tool
   - [ ] `weather_flow` — Weather tool

4. **Test vision** (uses llava):
   - [ ] `describe_image` — Image description from URL

5. **Test embeddings & RAG** (uses nomic-embed-text + gemma3):
   - [ ] `Pokedex` — Ask questions about Pokemon using local RAG

6. **Example queries for Pokedex**:
   - "Tell me about fire-type Pokemon"
   - "Which Pokemon can fly?"
   - "What's the strongest water Pokemon?"

7. **Notes**:
   - gemma2:latest does NOT support tool calling; use mistral-nemo
   - Vision requires `ollama pull llava` before use
   - First run may be slow (model loading into memory)
   - All processing happens locally — no API calls

8. **Expected behavior**:
   - Responses generated locally (no external requests)
   - Streaming shows incremental output
   - Tools work with compatible models only
   - Vision describes images accurately
   - Embeddings computed locally, similarity search finds relevant Pokemon
   - RAG combines retrieval with generation
