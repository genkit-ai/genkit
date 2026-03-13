# Cloudflare Workers AI Hello World Sample

> **Community Plugin** – This plugin is maintained by the community and is supported on a best-effort basis. It is not an official Cloudflare product.

This sample demonstrates how to use the [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) plugin for Genkit.

## Prerequisites

1. A [Cloudflare account](https://dash.cloudflare.com/) with Workers AI enabled
2. Your Cloudflare Account ID
3. An API token with Workers AI permissions

## Setup

1. Export your credentials:

```bash
export CLOUDFLARE_ACCOUNT_ID=your_account_id
export CLOUDFLARE_API_TOKEN=your_api_token
```

2. Run the sample:

```bash
./run.sh
```

## Features Demonstrated

### Text Generation

The `say_hello` flow uses Llama 3.1 8B to generate a friendly greeting:

```python
response = await ai.generate(prompt=f'Say hello to {input.name}!')
```

### Streaming

The `streaming_demo` flow demonstrates token-by-token streaming:

```python
async for chunk in ai.generate_stream(prompt='Tell me a short joke.'):
    print(chunk.text, end='')
```

### Tool Calling

The `tool_demo` flow shows how to use tools with Cloudflare models:

```python
@ai.tool()
async def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny, 72°F."
```

### Embeddings

The `embedding_demo` flow generates text embeddings:

```python
embeddings = await ai.embed(embedder=bge_base_en, documents=['Hello world'])
```

### OTLP Telemetry (Optional)

Enable trace export to any OTLP-compatible backend:

```python
from genkit.plugins.cloudflare_workers_ai import add_cloudflare_telemetry

# Set CF_OTLP_ENDPOINT environment variable, then:
add_cloudflare_telemetry()
```

## Testing

After starting the sample, open the Genkit DevUI at http://localhost:4000 and:

1. **say_hello**: Enter a name to get a personalized greeting
2. **streaming_demo**: Watch tokens stream in real-time
3. **tool_demo**: See tool calling in action
4. **embedding_demo**: Generate and view embedding vectors
5. **generate_character**: Generate a structured RPG character as JSON

## Supported Models

### Text Generation
- `@cf/meta/llama-3.1-8b-instruct` - Default model
- `@cf/meta/llama-3.3-70b-instruct-fp8-fast` - Larger, more capable
- `@hf/mistral/mistral-7b-instruct-v0.2` - Mistral alternative (HuggingFace hosted)

### Embeddings
- `@cf/baai/bge-base-en-v1.5` - 768 dimensions
- `@cf/baai/bge-large-en-v1.5` - 1024 dimensions
- `@cf/baai/bge-small-en-v1.5` - 384 dimensions
