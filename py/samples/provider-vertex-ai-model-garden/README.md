# Vertex AI Model Garden Sample

Access third-party models (Claude, Llama, Mistral) through Google Cloud's
infrastructure — one auth method, no separate API keys.

## How Model Garden Works

```
┌─────────────────────────────────────────────────────────────────┐
│                   VERTEX AI MODEL GARDEN                         │
│                                                                  │
│   Your Code                Google Cloud                          │
│   ─────────                ────────────                          │
│   ┌──────────┐             ┌──────────────────────┐             │
│   │  Genkit   │────────────►  Model Garden         │             │
│   │  + Model  │  GCP Auth   │                      │             │
│   │  Garden   │  (ADC)      │  ┌────────────────┐  │             │
│   │  Plugin   │             │  │ Claude (Anthro) │  │             │
│   └──────────┘             │  │ Llama (Meta)    │  │             │
│                             │  │ Mistral         │  │             │
│                             │  │ Gemini          │  │             │
│                             │  └────────────────┘  │             │
│                             └──────────────────────┘             │
│                                                                  │
│   One login (gcloud auth) → access to all models                │
└─────────────────────────────────────────────────────────────────┘
```

## Features Demonstrated

| Feature | Flow | Description |
|---------|------|-------------|
| Simple Generation | `say_hi` | Basic text generation with Claude |
| Streaming | `say_hi_stream` | Token-by-token streaming response |
| System Prompts | `system_prompt` | Persona control (pirate captain) |
| Multi-turn Chat | `multi_turn_chat` | Context-preserving conversations |
| Tool Calling | `weather_flow` | Function calling with tools |
| Multi-tool Chains | `claude-sonnet-4 - tool_calling_flow` | Weather + unit conversion |
| Structured Output | `generate_character` | JSON output with Pydantic schema |
| Streaming Structured | `streaming_structured_output` | Progressive JSON parsing |
| Generation Config | `jokes_flow` | Custom temperature, max_output_tokens |
| Cross-model | `gemini-2.5-flash - tool_flow` | Gemini via same project |
| Llama Models | `llama-3.2 - basic_flow` | Meta Llama via Model Garden |

## ELI5: Key Concepts

| Concept | ELI5 |
|---------|------|
| **Model Garden** | Google Cloud's model marketplace — access Claude, Llama, Mistral through one platform |
| **Third-party Models** | Models from other companies (Anthropic, Meta) running on Google's infrastructure |
| **GCP Credentials** | Your Google Cloud login — one auth method for all models, no separate API keys |
| **`ModelGardenPlugin`** | The plugin that connects Genkit to Model Garden |
| **`model_garden_name()`** | Helper to create model references — `"anthropic/claude-3-5-sonnet"` becomes the full path |

## Quick Start

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./run.sh
```

## Setup

### 1. Authenticate with Google Cloud

```bash
gcloud auth application-default login
```

### 2. Enable Model Garden Access

1. Go to [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) in GCP Console
2. Enable access to desired models (e.g., Claude Sonnet, Llama)
3. Note the **location** where each model is available (e.g., `us-east5` for Claude)

### 3. Set Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
# Location is prompted interactively (default: us-central1)
```

### 4. Run the Sample

```bash
./run.sh
```

Or manually:

```bash
genkit start -- uv run src/main.py
```

Then open the Dev UI at http://localhost:4000.

## Available Models

| Provider | Model | Location |
|----------|-------|----------|
| Anthropic | `claude-sonnet-4@20250514` | us-east5 |
| Anthropic | `claude-3-5-sonnet-v2@20241022` | us-east5 |
| Meta | `llama-3.2-90b-vision-instruct-maas` | us-central1 |
| Mistral | `ministral-3-14b-instruct-2512` | us-central1 |
| Google | `gemini-2.5-flash` (via VertexAI plugin) | project default |

## Testing This Demo

1. **Open DevUI** at http://localhost:4000

2. **Test basic flows**:
   - [ ] `say_hi` — Greeting with Claude
   - [ ] `say_hi_stream` — Streaming response
   - [ ] `system_prompt` — Pirate captain persona
   - [ ] `multi_turn_chat` — Travel assistant multi-turn

3. **Test tools**:
   - [ ] `weather_flow` — Tool calling with Claude
   - [ ] `claude-sonnet-4 - tool_calling_flow` — Weather + Fahrenheit conversion
   - [ ] `gemini-2.5-flash - tool_flow` — Same tools with Gemini

4. **Test structured output**:
   - [ ] `generate_character` — RPG character as JSON
   - [ ] `streaming_structured_output` — Progressive JSON streaming
   - [ ] `jokes_flow` — Custom temperature config

5. **Test cross-model**:
   - [ ] `llama-3.2 - basic_flow` — Meta Llama via Model Garden

6. **Expected behavior**:
   - Models respond via Vertex AI infrastructure
   - No direct API keys needed (uses GCP auth)
   - Enterprise features (logging, quotas) available

## Development

The `run.sh` script uses `watchmedo` for hot reloading on file changes.
