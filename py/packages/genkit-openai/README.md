# Genkit OpenAI Plugin

OpenAI-compatible model provider for Genkit (OpenAI, Azure OpenAI, and other
compatible endpoints).

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

## Config

`config` takes the settings `OpenAIConfig` declares. Any other key raises
before the request is sent, so a typo fails by name:

```python
res = await ai.generate(
    model='openai/gpt-4o',
    prompt='Write a haiku about the sea.',
    config={
        'temperature': 0.2,
        'max_tokens': 200,  # caps reply length
        'extra': {'reasoning': {'effort': 'low'}},  # fields OpenAIConfig doesn't declare
    },
)
```

- `max_tokens` caps reply length. Reasoning models get it as
  `max_completion_tokens`. `max_output_tokens` is accepted but sends no cap.
- `extra` goes out as top-level request body fields. A key in `extra`
  replaces a setting of the same name, and nothing inside it is checked, so
  don't put API keys there.

## Per-request API key

To bill a call to someone else's OpenAI account, pass their key in
`context.secrets`. It's used for that call only, on chat, image,
text-to-speech, and transcription models:

```python
res = await ai.generate(
    model='openai/gpt-4o',
    prompt='hi',
    context={'secrets': {'api_key': tenant_key}},
)
```

The plugin doesn't need a key of its own for this: `OpenAI()` with no
`OPENAI_API_KEY` serves every call that brings a key in `context.secrets`,
and a call without one fails with `FAILED_PRECONDITION`.
