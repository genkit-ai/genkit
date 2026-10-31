# Genkit Ollama Plugin

This Genkit plugin connects Python apps to locally running Ollama models for
chat, streaming, tool calling, multimodal prompts, and embeddings.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Installation

```bash
uv add genkit genkit-ollama
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
ollama pull llava
ollama pull nomic-embed-text
```

## Usage

```python
from genkit import Genkit
from genkit_ollama import Ollama

ai = Genkit(
    plugins=[Ollama()],
    model='ollama/llama3.2',
)

response = await ai.generate(prompt='Write a haiku about local models.')
print(response.text)

embeddings = await ai.embed(embedder='ollama/nomic-embed-text', content='local inference')
print(len(embeddings[0].embedding))
```

These snippets assume an async context (`await` inside an `async def`); pasting
them at module top level raises `SyntaxError: 'await' outside function`. See the
[runnable sample](https://github.com/genkit-ai/genkit/tree/main/py/samples/ollama-sample)
for a complete `async def main()` plus `ai.run_main(...)` entry point.

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

`OllamaConfig` extends the common Genkit `ModelConfig` with `think`,
`keep_alive`, and every sampler option the installed `ollama` client knows
(`num_ctx`, `min_p`, `seed`, `num_predict`, `repeat_penalty`, `mirostat`, ...).
Unknown keys raise `INVALID_ARGUMENT`, so a misspelled `num_ctx` fails instead
of running with the default context window.

```python
from genkit_ollama import OllamaConfig

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

A sampler option this class doesn't declare yet goes in `extra`. It is merged
into the request's `options` as written and wins over a declared field with the
same name:

```python
config={'temperature': 0.2, 'extra': {'new_sampler_option': 0.5}}
# => options={'temperature': 0.2, 'new_sampler_option': 0.5}
```

`stop_sequences` maps to Ollama's `stop`. When both are set, `stop` wins, the
same way `num_predict` wins over `max_output_tokens`.

### Remote server, headers, and timeouts

`Ollama()` reads `OLLAMA_HOST`, then falls back to `http://127.0.0.1:11434`.
`base_url` is the server root, with no `/v1` or `/api` suffix; the plugin calls
`{base_url}/api/chat`.

```python
Ollama(base_url='http://ollama.example.com:11434')

# Static headers
Ollama(request_headers={'Authorization': 'Bearer <token>'})

# Headers computed per request: Ollama on Cloud Run behind IAM needs a fresh ID token
import asyncio

import google.auth.transport.requests
from google.oauth2 import id_token

OLLAMA_URL = 'https://ollama-gpu-abc123-uc.a.run.app'
creds = id_token.fetch_id_token_credentials(OLLAMA_URL)


async def cloud_run_auth() -> dict[str, str]:
    # Refresh only when the cached token has expired, off the event loop.
    if not creds.valid:
        await asyncio.to_thread(creds.refresh, google.auth.transport.requests.Request())
    return {'Authorization': f'Bearer {creds.token}'}


Ollama(base_url=OLLAMA_URL, request_headers=cloud_run_auth, timeout=60.0)
```

A dict is sent as-is. A callable takes no arguments, may be sync or async, and
runs before every HTTP request on the plugin's shared connection pool, so it
should cache its token and only refresh when it expires.

### Per-request API key (hosted Ollama)

A multi-tenant app can bill each call to the caller's own key. Pass it on the
call's context, not the config, so it never lands in a trace:

```python
# 1. Plugin runs on OLLAMA_API_KEY (or no key) by default
ai = Genkit(plugins=[Ollama(base_url='https://ollama.com')])

# 2. This call authenticates as the tenant
response = await ai.generate(
    model='ollama/gpt-oss:120b',
    prompt='Summarize the order history.',
    context={'secrets': {'api_key': tenant_key}},
)
# => sent with Authorization: Bearer <tenant_key>
```

- **Slot**: `context.secrets.api_key` (`apiKey` works too). It applies to that
  call only; concurrent tenants each send their own key.
- **No key in secrets**: `secrets` without `api_key`, or keys elsewhere in the
  context (an auth provider's `context['api_key']`, another provider's key),
  are ignored and the call runs on the plugin's headers.
- **Bad key**: a key that is blank, not a string, or contains whitespace fails
  the call with `INVALID_ARGUMENT`.
- **Config**: `config={'api_key': ...}` or `config.extra.api_key` raises
  `INVALID_ARGUMENT` before anything is sent.
- **Proxy auth**: if `request_headers` sets `Authorization` (a dict, or a
  callable that returns one), a per-request key fails with
  `FAILED_PRECONDITION` rather than replacing the proxy's token.
- **Embedders**: `embed()` takes no context, so embedders always run on the
  plugin's headers.

### Model capabilities

There is nothing to configure. The Dev UI lists the models the server has pulled
(`/api/tags`), and any `ollama/<name>` resolves on first use. Every model goes
through `/api/chat`. The plugin asks `/api/show` once per model: `vision` turns
on media input and `tools` turns on tool calling. If the probe fails (server
down, model not pulled, an Ollama too old to report `capabilities`), the model
advertises both, as JS and Go do. Names containing `embed` are listed as
embedders.

### Troubleshooting

If the plugin can't reach the server it raises a `GenkitError` with status
`UNAVAILABLE` and the URL it tried (`DEADLINE_EXCEEDED` if the request timed
out). Start the daemon (`ollama serve`) or set `base_url` to a reachable
host.

## Sample

See [py/samples/ollama-sample](https://github.com/genkit-ai/genkit/tree/main/py/samples/ollama-sample) for a runnable sample covering
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
