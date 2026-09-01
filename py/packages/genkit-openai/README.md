# Genkit OpenAI Plugin

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
    model=OpenAI.gpt_model('gpt-5.2'),
    prompt='Suggest 2 catchy names for an AI newsletter.',
)
print(res.text)
```

Set `OPENAI_API_KEY` in the environment, or pass `api_key=` to `OpenAI()`.
