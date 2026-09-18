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

## xAI (Grok)

The package also includes a dedicated xAI provider for Grok chat, vision, and
image-generation models. Set `XAI_API_KEY` in the environment, or pass
`api_key=` to `XAI()`.

```python
from genkit import Genkit
from genkit_openai import XAI

ai = Genkit(plugins=[XAI()])

res = await ai.generate(
    model='xai/grok-3',
    prompt='Explain why the sky is blue in one paragraph.',
)
print(res.text)
```

Use `xai/grok-2-vision-1212` for image understanding and
`xai/grok-2-image-1212` for image generation.
