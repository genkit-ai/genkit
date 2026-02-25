# Hello Google GenAI - Vertex AI

Demonstrates using Google Cloud Vertex AI with Genkit for text generation.

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with Gemini |
| Streaming | `say_hi_stream` | Token-by-token streaming response |
| System Prompts | `system_prompt` | Persona control (pirate captain) |
| Multi-turn Chat | `multi_turn_chat` | Context-preserving conversations |
| Tool Calling | `tool_calling` | Weather + Fahrenheit conversion |
| Structured Output | `generate_character` | RPG character as JSON (Pydantic) |
| Streaming Structured | `streaming_structured_output` | Progressive JSON parsing |
| Generation Config | `generate_with_config` | Custom temperature, max_output_tokens |
| Multimodal (Image) | `describe_image` | Image description with Gemini |
| Code Generation | `generate_code` | Code generation from descriptions |
| Embeddings | `embed_docs` | Text embeddings with Vertex AI |

## Quick Start

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

That's it! The script will:

1. ✓ Prompt for your project ID if not set
2. ✓ Check gcloud authentication (and help you authenticate if needed)
3. ✓ Enable Vertex AI API (with your permission)
4. ✓ Install dependencies
5. ✓ Start the demo and open your browser

## Manual Setup (if needed)

If you prefer manual setup or the automatic setup fails:

### 1. Install gcloud CLI

Download from: https://cloud.google.com/sdk/docs/install

### 2. Authentication

```bash
gcloud auth application-default login
```

### 3. Enable Vertex AI API

```bash
gcloud services enable aiplatform.googleapis.com --project=$GOOGLE_CLOUD_PROJECT
```

### 4. Run the Demo

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

Then open the Dev UI at http://localhost:4000

## Development

The `run.sh` script uses `watchmedo` to monitor changes in:
- `src/` (Python logic)
- `../../packages` (Genkit core)
- `../../plugins` (Genkit plugins)
- File patterns: `*.py`, `*.prompt`, `*.json`

Changes will automatically trigger a restart of the sample.
