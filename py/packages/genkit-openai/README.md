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

- `max_tokens` caps reply length. Genkit's cross-model `max_output_tokens`
  (`maxOutputTokens` in the Dev UI) sends the same cap when `max_tokens` is
  unset. Reasoning models get the cap as `max_completion_tokens`.
- `extra` goes out as top-level request body fields. A key in `extra`
  replaces a setting of the same name. An API key in `config` or `extra`
  raises `INVALID_ARGUMENT`; it belongs in `context.secrets`.

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

- `apiKey` works as well as `api_key`. Other entries in `context.secrets`
  are left alone, so a call whose secrets hold no key runs on the plugin's
  key. A key that is blank or not a string raises `INVALID_ARGUMENT`.
- The tenant call sends only the tenant's key. The plugin's `organization`,
  `project`, `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` are left off; `base_url`,
  timeouts and other `default_headers` carry over. A plugin whose
  `default_headers` pin `Authorization` refuses a tenant key with
  `FAILED_PRECONDITION`.
- The plugin doesn't need a key of its own for chat, image, text-to-speech,
  and transcription models: `OpenAI()` with no `OPENAI_API_KEY` serves every
  call that brings a key in `context.secrets`, and a call without one fails
  with `FAILED_PRECONDITION`.
- Embedders run on the plugin's key only, since `ai.embed()` takes no
  `context`. On a plugin without a key they fail with `FAILED_PRECONDITION`.
