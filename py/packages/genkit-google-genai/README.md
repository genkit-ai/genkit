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

### Dynamic Models & Specifying Model References

Any model ID served by the Gemini API (including newly released models) is resolved lazily on demand when `ai.generate()` is called, without needing pre-registration or SDK updates. You can pass models to `Genkit(model=...)` or `ai.generate(model=...)` in two ways:

- **Typed `ModelRef` (recommended)**: `GoogleAI.gemini_model('gemini-flash-latest')` or `VertexAI.gemini_model('gemini-4-argon')` — pass the bare model ID; returns a `ModelRef[GeminiConfig]` with the `'googleai/'` or `'vertexai/'` namespace stamped automatically and static config type-checking.
- **Prefixed string shorthand**: `'googleai/gemini-flash-latest'` or `'vertexai/gemini-4-argon'` — when passing a plain `str`, the `'googleai/'` or `'vertexai/'` prefix is required; passing an unprefixed string raises `NOT_FOUND`.

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

### Structured output

Set the output schema on the request; the plugin sends it to Gemini as
`responseJsonSchema`, so `anyOf`, `$ref`, `additionalProperties` and numeric
bounds reach the model as written.

```python
from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str
    population: int


response = await ai.generate(prompt='Give quick facts about Japan.', output_schema=Country)
print(response.output)
```

To send the schema as the older OpenAPI-style `responseSchema` field instead,
construct the plugin with `GoogleAI(legacy_response_schema=True)` or
`VertexAI(legacy_response_schema=True)`. The schema is converted to that shape.
Keywords the conversion does not map, such as `pattern`, `minLength`,
`additionalProperties` and an `anyOf` or `oneOf` next to a `type`, are left
out, as are a `$ref` back to a model that contains it, an array with no
convertible `items` and a map (an object with `additionalProperties` and no
`properties`, such as `dict[str, int]`). It raises `INVALID_ARGUMENT` for a constraint with no
`type` (an `enum` of strings counts as a string), for a `type` list, `anyOf`
or `oneOf` that would lose some but not all of its non-null branches, for a
required property or whole schema that cannot be converted, and for an
unresolvable `$ref`, an unsupported `type` or a keyword value it cannot read.
Constrained enum output always goes out this way, as `text/x.enum` with a
converted `responseSchema`.

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

### Virtual Try-On (Vertex AI)

`virtual-try-on-001` dresses a person in a product and returns the composited
images. The inputs are media parts on the last message, tagged so the model
knows which is which: `metadata={'type': 'personImage'}` for the person and
`metadata={'type': 'productImage'}` for the product. Each `url` is a `data:`
url, bare base64 image data, or a `gs://` path; http(s) urls are not fetched.

```python
from genkit import Genkit, Message, Part, Role
from genkit_google_genai import VertexAI

ai = Genkit(plugins=[VertexAI()])

response = await ai.generate(
    model=VertexAI.virtual_try_on_model('virtual-try-on-001'),
    messages=[
        Message(
            role=Role.USER,
            content=[
                Part.from_media(
                    'gs://my-bucket/person.png', content_type='image/png', metadata={'type': 'personImage'}
                ),
                Part.from_media(
                    'gs://my-bucket/shirt.png', content_type='image/png', metadata={'type': 'productImage'}
                ),
            ],
        )
    ],
)
for part in response.message.content:
    print(part.media.url)
```

### Vertex AI Evaluators

Built-in evaluators for assessing model output quality, accessed via `ai.evaluate()`. The VertexAI plugin registers them only when a project resolves (`VertexAI(project=...)` or `GOOGLE_CLOUD_PROJECT`); `VertexAI(api_key=...)` without a project lists none. They authenticate with Application Default Credentials, not `api_key` or `credentials`:

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

for result in results:
    print(f'Score: {result.evaluation[0].score}')
```

Runnable snippets are in [`py/samples`](../../samples).
