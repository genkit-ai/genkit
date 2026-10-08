# Google GenAI Plugin

This Genkit plugin provides a unified interface for Google AI (Gemini) and Vertex AI models, embedding, and other services.

> **Building with a coding agent? Install the Genkit Python skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-python
> ```
>
> It teaches your agent the current Genkit Python APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

## Setup environment

```bash
uv add genkit genkit-google-genai
```

## Configuration

### Google AI (AI Studio)

To use Google AI models, obtain an API key from [Google AI Studio](https://aistudio.google.com/) and set it in your environment:

```bash
export GEMINI_API_KEY='<your-api-key>'
```

### Vertex AI (Google Cloud)

To use Vertex AI models, ensure you have a Google Cloud project and Application Default Credentials (ADC) set up:

```bash
gcloud auth application-default login
```

## Quickstart

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


@ai.flow()
async def greet(name: str) -> str:
    res = await ai.generate(prompt=f'Say hello to {name}.')
    return res.text
```

## Features

### Dynamic Models

The plugin automatically discovers available models from the API upon initialization. You can use any model name supported by the API (e.g., `GoogleAI.gemini_model('gemini-flash-latest')`, `VertexAI.gemini_model('gemini-3.1-pro-preview')`).

### Typed config

Config classes and their nested settings take snake_case keyword arguments,
so pyright, pyrefly and ty check every key. All of them import from the
package root. The Dev UI and JSON configs use the camelCase names
(`thinkingConfig`, `thinkingBudget`); both spellings validate at runtime.

```python
from genkit_google_genai import GeminiConfig, GeminiTtsConfig, PrebuiltVoiceConfig, SpeechConfig, ThinkingConfig, VoiceConfig

# 1. Cap thinking on a text model
config = GeminiConfig(temperature=0.4, thinking_config=ThinkingConfig(thinking_budget=1024))

# 2. Pick a prebuilt voice on a TTS model
tts_config = GeminiTtsConfig(
    speech_config=SpeechConfig(voice_config=VoiceConfig(prebuilt_voice_config=PrebuiltVoiceConfig(voice_name='Kore')))
)

# 3. Dump the config the way the Dev UI shows it
print(config.model_dump(by_alias=True, exclude_none=True))
# => {'temperature': 0.4, 'thinkingConfig': {'thinkingBudget': 1024}}
```

### Provider fields the plugin doesn't declare

Gemini, Gemini image, and Veo configs reject keys they don't declare, so a
typo like `temprature` fails before the request is sent. A request field the
schema doesn't declare yet goes in `extra` under its REST wire name. It's
merged over the top of the request body, recursing into nested objects:

```python
from genkit_google_genai import GeminiConfig

config = GeminiConfig.model_validate({
    'temperature': 1.0,
    'extra': {
        'generationConfig': {'newKnob': 1},  # adds one key to generationConfig
        'labels': {'team': 'search'},        # Vertex AI request labels
    },
})
```

`extra` can't set the fields Genkit builds from the request (`contents`,
`systemInstruction`, `tools`, `toolConfig`, the structured-output fields of
`generationConfig`, `cachedContent`); those raise `INVALID_ARGUMENT`.

### Video generation (Veo)

Video is a job, not a round-trip. `generate_operation` hands back a ticket;
`check_operation` is how you find out when the video is ready. When the job
finishes, `operation.output` has a playable `media.url` — Studio sends a
download URL, Vertex often sends the mp4 inline.

**With `GoogleAI`:**

```python
from genkit import Genkit
from genkit_google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

operation = await ai.generate_operation(
    model=GoogleAI.veo_model('veo-3.1-fast-generate-preview'),
    prompt='A paper airplane gliding through a bright classroom',
)
while not operation.done:
    operation = await ai.check_operation(operation)
print(operation.output)
```

**With `VertexAI`:**

```python
from genkit import Genkit
from genkit_google_genai import VertexAI

ai = Genkit(plugins=[VertexAI()])

operation = await ai.generate_operation(
    model=VertexAI.veo_model('veo-3.1-generate-001'),
    prompt='A paper airplane gliding through a bright classroom',
)
while not operation.done:
    operation = await ai.check_operation(operation)
print(operation.output)
```

Runnable version: [google-genai-media](https://github.com/genkit-ai/genkit/tree/main/py/samples/google-genai-media).

### Vertex AI Evaluators

Built-in evaluators for assessing model output quality, accessed via `ai.evaluate()`. The VertexAI plugin registers them only when a project resolves (`VertexAI(project=...)` or `GOOGLE_CLOUD_PROJECT`); `VertexAI(api_key=...)` without a project lists none. They authenticate with Application Default Credentials, not `api_key` or `credentials`:

```python
from genkit import BaseDataPoint, Genkit
from genkit_google_genai import VertexAI

ai = Genkit(plugins=[VertexAI(project='my-project')])

# Prepare test dataset
dataset = [
    BaseDataPoint(
        input='Write about AI.',
        output='AI is transforming industries through intelligent automation.',
    ),
]

# Evaluate fluency (scores 1-5)
results = await ai.evaluate(
    evaluator='vertexai/fluency',
    dataset=dataset,
)

for result in results:
    print(f'Score: {result.evaluation[0].score}')
```

Runnable snippets are in [`py/samples`](../../samples).
