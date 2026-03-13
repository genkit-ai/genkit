# Genkit Cohere AI Plugin (Community)

> **Community Plugin** — This plugin is community-maintained and is not an
> official Google or Cohere product. It is provided on an "as-is" basis.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

The Cohere AI plugin for [Genkit](https://github.com/firebase/genkit)
provides integration with [Cohere's](https://cohere.com/) AI models,
including:

- **Chat models** — Command A, Command A Vision, Command A Reasoning,
  Command R+, Command R, and more for text generation, tool calling,
  and structured output via the V2 API.
- **Embedding models** — Embed v4.0, Embed English/Multilingual v3.0
  for semantic search, RAG, and clustering.

## Installation

```bash
pip install genkit-plugin-cohere
```

## Quick Start

```python
from genkit.ai import Genkit
from genkit.plugins.cohere import Cohere, cohere_name

ai = Genkit(
    plugins=[Cohere()],
    model=cohere_name('command-a-03-2025'),
)

response = await ai.generate(prompt='Hello, Cohere!')
print(response.text)
```

## Configuration

Set your Cohere API key via the `COHERE_API_KEY` environment variable
(or `CO_API_KEY`), or pass it directly:

```python
ai = Genkit(plugins=[Cohere(api_key='your-api-key')])
```

Get an API key from: https://dashboard.cohere.com/api-keys

## Supported Models

### Chat Models

| Model                         | Tool Calling | Structured Output | Media | Streaming |
|-------------------------------|:------------:|:-----------------:|:-----:|:---------:|
| `command-a-03-2025`           | ✅           | ✅                | ❌    | ✅        |
| `command-a-reasoning-08-2025` | ✅           | ✅                | ❌    | ✅        |
| `command-a-translate-08-2025` | ✅           | ✅                | ❌    | ✅        |
| `command-a-vision-07-2025`    | ❌           | ✅                | ✅    | ✅        |
| `command-r7b-12-2024`         | ✅           | ✅                | ❌    | ✅        |
| `command-r-plus-08-2024`      | ✅           | ✅                | ❌    | ✅        |
| `command-r-plus-04-2024`      | ✅           | ✅                | ❌    | ✅        |
| `command-r-plus`              | ✅           | ✅                | ❌    | ✅        |
| `command-r-08-2024`           | ✅           | ✅                | ❌    | ✅        |
| `command-r-03-2024`           | ✅           | ✅                | ❌    | ✅        |
| `command-r`                   | ✅           | ✅                | ❌    | ✅        |

> **Note:** `command` and `command-light` were removed by Cohere on
> September 15, 2025 and are not supported.

### Embedding Models

| Model                           | Dimensions | Languages    |
|---------------------------------|-----------:|--------------|
| `embed-v4.0`                    | 1024       | Multilingual |
| `embed-english-v3.0`            | 1024       | English      |
| `embed-english-light-v3.0`      | 384        | English      |
| `embed-multilingual-v3.0`       | 1024       | 100+         |
| `embed-multilingual-light-v3.0` | 384        | 100+         |

## Features

### Tool Calling

```python
@ai.tool()
async def get_weather(location: str) -> str:
    \"\"\"Get weather for a location.\"\"\"
    return f'Sunny and 72°F in {location}'

response = await ai.generate(
    model=cohere_name('command-a-03-2025'),
    prompt='What is the weather in San Francisco?',
    tools=[get_weather],
)
```

### Structured Output

```python
from pydantic import BaseModel
from genkit.ai import Output

class Character(BaseModel):
    name: str
    backstory: str

response = await ai.generate(
    prompt='Create an RPG character',
    output=Output(schema=Character),
)
character = response.output  # typed as Character
```

### Embeddings

```python
from genkit.blocks.document import Document

doc = Document.from_text('Hello world')
embeddings = await ai.embed(
    embedder=cohere_name('embed-v4.0'),
    content=doc,
)
```

## Disclaimer

This is a **community-maintained** plugin and is not officially supported by
Google or Cohere. Use of Cohere's API is subject to
[Cohere's Terms of Use](https://cohere.com/terms-of-use) and
[Privacy Policy](https://cohere.com/privacy). You are responsible for
complying with all applicable terms when using this plugin.

- **API Key Security** — Never commit your Cohere API key to version control.
  Use environment variables or a secrets manager.
- **Usage Limits** — Be aware of your Cohere plan's rate limits and token
  quotas. See [Cohere Pricing](https://cohere.com/pricing).
- **Data Handling** — Review Cohere's data processing practices before
  sending sensitive or personally identifiable information.

## License

Apache-2.0
