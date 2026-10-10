# Genkit Vertex AI Plugin

Integrate Genkit with Google Cloud Vertex AI Model Garden.

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

Install the extra for the Model Garden publishers you call:

```bash
uv add 'genkit-vertexai[anthropic]'          # Claude
uv add 'genkit-vertexai[openai]'             # Llama, Mistral, and other OpenAI-compatible models
uv add 'genkit-vertexai[anthropic,openai]'   # both
```

The package without an extra serves no models. Calling a model without its
extra raises an error naming the command to run.

## Usage

```python
from genkit import Genkit
from genkit_vertexai import ModelGarden

ai = Genkit(
    plugins=[ModelGarden(project='my-project', location='us-central1')],
)

res = await ai.generate(
    model='modelgarden/anthropic/claude-3-5-sonnet-v2@20241022',
    prompt='Explain recursion in 10 words.',
)
print(res.text)
```

Requires Google Cloud Application Default Credentials (ADC) or explicit credentials.
