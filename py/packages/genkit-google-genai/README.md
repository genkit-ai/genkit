# Google GenAI Plugin

This Genkit plugin provides a unified interface for Google AI (Gemini) and Vertex AI models, embedding, and other services.

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

The plugin automatically discovers available models from the API upon initialization. You can use any model name supported by the API (e.g., `GoogleAI.gemini_model('gemini-flash-latest')`, `VertexAI.gemini_model('gemini-2.5-pro')`).

### Dynamic Configuration

Unrecognized provider parameters on the family config are forwarded to the API:

```python
from genkit_google_genai import GeminiConfigSchema

config = GeminiConfigSchema.model_validate({
    'temperature': 1.0,
    'response_modalities': ['TEXT', 'IMAGE'],
})
```

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

Built-in evaluators for assessing model output quality. Evaluators are automatically registered when using the VertexAI plugin and are accessed via `ai.evaluate()`:

```python
from genkit import Genkit
from genkit.evaluator import BaseDataPoint
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

for result in results.root:
    print(f'Score: {result.evaluation.score}')
```

Runnable snippets are in [`py/samples`](../../samples).
