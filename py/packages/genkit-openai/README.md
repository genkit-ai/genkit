# Genkit OpenAI Plugin

> **Preview** — This plugin is in preview and may have API changes in future releases.

OpenAI-compatible model provider for Genkit (OpenAI, Azure OpenAI, and other
compatible endpoints).

## Installation

```bash
uv add genkit-openai
```

## Usage

```python
from genkit import Genkit
from genkit_openai import OpenAI

ai = Genkit(plugins=[OpenAI()])

res = await ai.generate(
    model='openai/gpt-5.2',
    prompt='Suggest 2 catchy names for an AI newsletter.',
)
print(res.text)
```

Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAI()`.
