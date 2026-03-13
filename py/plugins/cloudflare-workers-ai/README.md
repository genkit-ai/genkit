# Genkit Cloudflare Workers AI Plugin

> **Community Plugin** – This plugin is maintained by the community and is supported on a best-effort basis. It is not an official Cloudflare product.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

This plugin provides access to [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) models and OTLP telemetry export for the Genkit framework. Cloudflare Workers AI runs AI models at the edge, close to users, providing low-latency inference with global availability.

## Installation

```bash
pip install genkit-plugin-cloudflare-workers-ai
```

## Setup

1. Get your Cloudflare Account ID and API Token from the [Cloudflare Dashboard](https://dash.cloudflare.com/).
2. Set environment variables:

```bash
export CLOUDFLARE_ACCOUNT_ID=your_account_id
export CLOUDFLARE_API_TOKEN=your_api_token
```

## Usage

### Basic Text Generation

```python
from genkit import Genkit
from genkit.plugins.cloudflare_workers_ai import CloudflareWorkersAI, cloudflare_model

ai = Genkit(
    plugins=[CloudflareWorkersAI()],
    model=cloudflare_model('@cf/meta/llama-3.1-8b-instruct'),
)

@ai.flow()
async def say_hello(name: str) -> str:
    response = await ai.generate(prompt=f'Say hello to {name}!')
    return response.text
```

### Streaming

```python
@ai.flow()
async def stream_story() -> str:
    chunks = []
    async for chunk in ai.generate_stream(prompt='Write a short story.'):
        print(chunk.text, end='', flush=True)
        chunks.append(chunk.text)
    return ''.join(chunks)
```

### Embeddings

```python
from genkit.plugins.cloudflare_workers_ai import bge_base_en

@ai.flow()
async def get_embeddings(text: str) -> list[float]:
    response = await ai.embed(content=text, embedder=bge_base_en)
    return response.embeddings[0].embedding
```

### OTLP Telemetry Export

Enable trace export to any OTLP-compatible backend (Grafana, Honeycomb, etc.):

```python
from genkit.plugins.cloudflare_workers_ai import add_cloudflare_telemetry

# Set CF_OTLP_ENDPOINT environment variable, then:
add_cloudflare_telemetry()
```

## Supported Models

### Text Generation (Chat)

| Model ID | Description |
|----------|-------------|
| `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | Meta Llama 3.3 70B |
| `@cf/meta/llama-3.1-8b-instruct` | Meta Llama 3.1 8B |
| `@cf/meta/llama-3.1-8b-instruct-fast` | Meta Llama 3.1 8B (Fast) |
| `@cf/meta/llama-4-scout-17b-16e-instruct` | Meta Llama 4 Scout 17B (Multimodal) |
| `@cf/mistral/mistral-7b-instruct-v0.2` | Mistral 7B |
| `@cf/qwen/qwen1.5-14b-chat-awq` | Qwen 1.5 14B |

### Embeddings

| Model ID | Dimensions |
|----------|------------|
| `@cf/baai/bge-base-en-v1.5` | 768 |
| `@cf/baai/bge-large-en-v1.5` | 1024 |
| `@cf/baai/bge-small-en-v1.5` | 384 |

## Features

- **Text generation** with streaming support (SSE)
- **Tool/function calling** for agentic workflows
- **Text embeddings** for semantic search and RAG
- **Multimodal inputs** (with Llama 4 Scout)
- **Edge inference** - low latency globally
- **OTLP telemetry** - export traces to any compatible backend

## Testing

Run the sample application:

```bash
cd py/samples/provider-cloudflare-workers-ai-hello
./run.sh
```

Then open the Genkit Dev UI at http://localhost:4000 to test the flows.

## Documentation

- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/)
- [Models Catalog](https://developers.cloudflare.com/workers-ai/models/)
- [REST API](https://developers.cloudflare.com/workers-ai/get-started/rest-api/)
- [Workers Observability](https://developers.cloudflare.com/workers/observability/)

## Disclaimer

This is a community plugin and is not officially supported or endorsed by Cloudflare, Inc.
