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

## Features

### Dynamic Models

The plugin automatically discovers available models from the API upon initialization. You can use any model name supported by the API (e.g., `googleai/gemini-flash-latest`, `vertexai/gemini-2.5-pro`).

### Dynamic Configuration

New or experimental parameters can be passed on the family config and ride through to the API:

```python
from genkit_google_genai import GeminiConfigSchema

config = GeminiConfigSchema.model_validate({
    'temperature': 1.0,
    'response_modalities': ['TEXT', 'IMAGE'],
})
```

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


**Supported evaluators:**

| Evaluator | Description |
|-----------|-------------|
| `vertexai/bleu` | Translation quality (compare to reference) |
| `vertexai/rouge` | Summarization quality |
| `vertexai/fluency` | Language mastery and readability |
| `vertexai/safety` | Harmful/inappropriate content detection |
| `vertexai/groundedness` | Hallucination detection |
| `vertexai/summarization_quality` | Overall summarization ability |

## Examples

For comprehensive usage examples, see:

- [google-genai-media](https://github.com/genkit-ai/genkit/tree/main/py/samples/google-genai-media) - Speech, image, and video generation
- [gemini-code-execution](https://github.com/genkit-ai/genkit/tree/main/py/samples/gemini-code-execution) - Gemini code execution
- [gemini-context-caching](https://github.com/genkit-ai/genkit/tree/main/py/samples/gemini-context-caching) - Context caching for large prompts
- [vertexai-imagen](https://github.com/genkit-ai/genkit/tree/main/py/samples/vertexai-imagen) - Vertex AI Imagen generation
