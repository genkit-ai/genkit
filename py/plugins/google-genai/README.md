# Google GenAI Plugin

This Genkit plugin provides a unified interface for Google AI (Gemini) and Vertex AI models, embedding, and other services.

## Setup environment

```bash
uv venv
source .venv/bin/activate
pip install genkit-plugins-google-genai
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

The plugin automatically discovers available models from the API upon initialization. You can use any model name supported by the API (e.g., `googleai/gemini-2.0-flash-exp`, `vertexai/gemini-1.5-pro`).

### Dynamic Configuration

New or experimental parameters can be passed flexibly using `model_validate` to bypass strict schema checks:

```python
from genkit.plugins.google_genai import GeminiConfigSchema

config = GeminiConfigSchema.model_validate({
    'temperature': 1.0,
    'response_modalities': ['TEXT', 'IMAGE'],
})
```

### Vertex AI Rerankers

The VertexAI plugin provides semantic rerankers for improving RAG quality by re-scoring documents based on relevance:

```python
from genkit import Genkit
from genkit.plugins.google_genai import VertexAI

ai = Genkit(plugins=[VertexAI(project='my-project')])

# Rerank documents after retrieval
ranked_docs = await ai.rerank(
    reranker='vertexai/semantic-ranker-default@latest',
    query='What is machine learning?',
    documents=retrieved_docs,
    options={'top_n': 5},
)
```

**Supported Models:**

| Model | Description |
|-------|-------------|
| `semantic-ranker-default@latest` | Latest default semantic ranker |
| `semantic-ranker-default-004` | Semantic ranker version 004 |
| `semantic-ranker-fast-004` | Fast variant (lower latency) |

### Vertex AI Evaluators

Built-in evaluators for assessing model output quality. Evaluators are automatically registered when using the VertexAI plugin and are accessed via `ai.evaluate()`:

```python
from genkit import Genkit
from genkit.core.typing import BaseDataPoint
from genkit.plugins.google_genai import VertexAI

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


**Supported Metrics:**

| Metric | Description |
|--------|-------------|
| `BLEU` | Translation quality (compare to reference) |
| `ROUGE` | Summarization quality |
| `FLUENCY` | Language mastery and readability |
| `SAFETY` | Harmful/inappropriate content detection |
| `GROUNDEDNESS` | Hallucination detection |
| `SUMMARIZATION_QUALITY` | Overall summarization ability |

## Examples

For comprehensive usage examples, see:

- [`py/samples/provider-google-genai-hello/README.md`](../../samples/provider-google-genai-hello/README.md) - Basic Gemini usage
- [`py/samples/provider-vertex-ai-rerank-eval/README.md`](../../samples/provider-vertex-ai-rerank-eval/README.md) - Rerankers and evaluators
