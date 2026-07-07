# Genkit Ollama Plugin

This Genkit plugin connects Python apps to locally running Ollama models for
chat, streaming, tool calling, multimodal prompts, and embeddings.

## Installation

```bash
uv add genkit genkit-plugin-ollama
```

Install Ollama from [ollama.com/download](https://ollama.com/download), then
start the local server:

```bash
ollama serve
```

Ollama serves `http://127.0.0.1:11434` by default. Pull the models your app will
use before running Genkit:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Usage

```python
from genkit import Genkit
from genkit.plugins.ollama import EmbeddingDefinition, ModelDefinition, Ollama

ai = Genkit(
    plugins=[
        Ollama(
            models=[ModelDefinition(name='llama3.2')],
            embedders=[EmbeddingDefinition(name='nomic-embed-text')],
        )
    ],
    model='ollama/llama3.2',
)

response = await ai.generate(prompt='Write a haiku about local models.')
print(response.text)

embeddings = await ai.embed(embedder='ollama/nomic-embed-text', content='local inference')
print(len(embeddings[0].embedding))
```

These snippets assume an async context (`await` inside an `async def`); pasting
them at module top level raises `SyntaxError: 'await' outside function`. See the
[runnable sample](../../samples/ollama-sample) for a complete `async def main()`
plus `ai.run_main(...)` entry point.

### Streaming

```python
stream_response = ai.generate_stream(prompt='Stream a haiku about Ollama.')
async for chunk in stream_response.stream:
    print(chunk.text, end='', flush=True)
final = await stream_response.response
```

### Tool calling

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(description='City to look up')

@ai.tool()
async def current_weather(input: WeatherInput) -> str:
    return f'{input.city} is 18°C and partly cloudy.'

response = await ai.generate(
    prompt='What is the weather in London?',
    tools=['current_weather'],
)
print(response.text)
```

Ollama tool inputs are object schemas, so wrap primitive inputs in a Pydantic
model as above. When a tool's schema declares `properties` but omits an explicit
`type`, the plugin infers an object schema rather than dropping the tool.

### JSON / schema-constrained output

```python
from pydantic import BaseModel

class Haiku(BaseModel):
    line_one: str
    line_two: str
    line_three: str

response = await ai.generate(
    prompt='Write a haiku about local models.',
    output_schema=Haiku,
)
print(response.output)
```

### Ollama-specific config (`OllamaConfig`)

`OllamaConfig` extends the common Genkit `ModelConfig` with Ollama-only
knobs (`think`, `keep_alive`, `num_ctx`, `min_p`, `seed`, `num_predict`):

```python
from genkit.plugins.ollama import OllamaConfig

# Reasoning model with a 32k context window kept warm for an hour
response = await ai.generate(
    model='ollama/deepseek-r1',
    prompt='Plan a small REST API.',
    config=OllamaConfig(
        think=True,
        num_ctx=32_000,
        keep_alive='1h',
        temperature=0.2,
    ),
)
```

### Remote server, headers, and timeouts

```python
Ollama(server_address='http://ollama.example.com:11434')

# Static headers
Ollama(request_headers={'Authorization': 'Bearer <token>'})

# Async-resolved headers, re-evaluated per request (e.g. minting a short-lived token)
from genkit.plugins.ollama import RequestHeaderParams

async def auth_headers(params: RequestHeaderParams) -> dict[str, str]:
    return {'Authorization': f'Bearer {await mint_token(params.server_address)}'}

Ollama(request_headers=auth_headers, timeout=60.0)
```

Callable headers are re-evaluated on every request, so short-lived tokens refresh
automatically. A static dict is applied once to a cached client.

### Vision models

```python
from genkit.plugins.ollama import ModelDefinition, Ollama, OllamaSupports

Ollama(models=[ModelDefinition(name='llava', supports=OllamaSupports(media=True))])
```

Media support is opt-in per model to avoid advertising a capability the
underlying model does not actually have.

### Troubleshooting

If the plugin can't reach the server it raises `OllamaConnectionError`
with the URL it tried. Start the daemon (`ollama serve`) or set
`server_address` to a reachable host.

## Sample

See [`py/samples/ollama-sample`](../../samples/ollama-sample) for a runnable sample covering
chat, streaming, tool calling, and embeddings with a local Ollama server.

## Notes

Ollama is open-source software under the
[MIT License](https://github.com/ollama/ollama/blob/main/LICENSE). Individual
models pulled through Ollama have their own licenses; review model cards
before production use. Models run locally on your hardware by default — no
data leaves the machine unless you point the plugin at a remote Ollama
server.

## Acknowledgements

Thanks to the community contributors who built and maintained the original
community version of this plugin.

## License

Apache-2.0
