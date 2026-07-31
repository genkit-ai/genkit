# Genkit Vertex AI Plugin

Integrate Genkit with Google Cloud Vertex AI Model Garden.

## Installation

```bash
uv add genkit-vertexai
```

## Usage

```python
from genkit import Genkit
from genkit_vertexai.model_garden import ModelGarden

ai = Genkit(
    plugins=[ModelGarden(project_id='my-project', location='us-central1')],
)

res = await ai.generate(
    model='modelgarden/anthropic/claude-3-5-sonnet-v2@20241022',
    prompt='Explain recursion in 10 words.',
)
print(res.text)
```

Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.
