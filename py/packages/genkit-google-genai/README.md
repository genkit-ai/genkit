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

# 1. Initialize Genkit with plugin and default model reference
ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini('gemini-flash-latest'))


# 2. Define a typed flow
@ai.flow()
async def greet(name: str) -> str:
    res = await ai.generate(prompt=f'Say hello to {name}.')
    return res.text


# 3. Execute and inspect output
print(await greet('Grace'))
# => Hello Grace! How can I help you today?
```

## Features

### Dynamic Models

The plugin automatically discovers available models from the API upon initialization. You can use any model name supported by the API via model reference constructors (e.g., `GoogleAI.gemini('gemini-flash-latest')`, `VertexAI.gemini('gemini-3.1-pro-preview')`).

### Typed config

Config classes take flat snake_case keyword arguments, so pyright, pyrefly
and ty check every key. The plugin builds Gemini's nested request objects
(`thinkingConfig`, `speechConfig`, `tools`) when it sends the request. The Dev
UI and `.prompt` files use the same flat fields in camelCase
(`thinkingLevel`); both spellings validate at runtime.

#### Text generation with GeminiConfig and tool_choice

```python
from genkit import Genkit
from genkit_google_genai import GeminiConfig, GoogleAI

# 1. Initialize plugin and client
ai = Genkit(plugins=[GoogleAI()])


# 2. Define a function tool
@ai.tool()
async def check_inventory(ingredient: str) -> dict[str, int]:
    return {'pesto': 12, 'pine_nuts': 4}


# 3. Generate with flat thinking, safety, search grounding, and a tool
res = await ai.generate(
    model=GoogleAI.gemini('gemini-flash-latest'),
    prompt='Check pesto inventory and cite current safety storage guidelines.',
    tools=[check_inventory],
    config=GeminiConfig(
        thinking_level='HIGH',
        include_thoughts=True,
        safety_settings=[
            {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_ONLY_HIGH'},
        ],
        google_search=True,
    ),
)

print(res.text)
# => Inventory verified: 12 jars of pesto in stock. Storage guidelines require refrigeration at 40°F (4°C) or below after opening.
```

#### Audio generation with GeminiTtsConfig

```python
from genkit import Genkit
from genkit_google_genai import GeminiTtsConfig, GoogleAI

ai = Genkit(plugins=[GoogleAI()])

# Generate multi-speaker audio dialog
res = await ai.generate(
    model=GoogleAI.gemini_tts('gemini-2.5-flash-preview-tts'),
    prompt='Host: Welcome to Morning Roaster. Chef: Today we review Ethiopian beans.',
    config=GeminiTtsConfig(
        speakers={'Host': 'Kore', 'Chef': 'Puck'},
        language_code='en-US',
    ),
)

print(res.media.url)
# => data:audio/wav;base64,UklGRiQAAABXQVZFZ...
```

#### Image generation with GeminiImageConfig

```python
from genkit import Genkit
from genkit_google_genai import GeminiImageConfig, GoogleAI

ai = Genkit(plugins=[GoogleAI()])

# Generate high-resolution image with specified aspect ratio and size
res = await ai.generate(
    model=GoogleAI.gemini_image('gemini-2.5-flash-preview-image-generation'),
    prompt='A modern minimalist espresso bar with warm cedar accents.',
    config=GeminiImageConfig(
        aspect_ratio='16:9',
        image_size='2K',
    ),
)

print(res.media.url)
# => data:image/png;base64,iVBORw0KGgoAAAANSU...
```

#### Default config on model references

Pass a config instance directly to `GoogleAI.gemini(...)` or `VertexAI.gemini(...)` to configure defaults that every call inherits:

```python
model = GoogleAI.gemini(
    'gemini-flash-latest',
    config=GeminiConfig(temperature=0.7, thinking_level='MEDIUM'),
)
```

### Config rules and validation

Choice fields take plain strings. The Literal aliases (`ThinkingLevel`,
`HarmCategory`, `ImageAspectRatio`, ...) export from the package root for
annotating your own code. `safety_settings` entries are `SafetySetting`
TypedDicts: pass a dict literal or `SafetySetting(category=..., threshold=...)`.

Setting any `google_search_*` option turns Google Search on;
`google_search=False` next to an option raises. A nested key such as
`thinkingConfig` raises and names the flat fields that replace it.

Function calling mode is controlled by `ai.generate(tool_choice=...)`, not config:
`'auto'`, `'required'`, or `'none'`. In `'required'` mode, Gemini binds the request's
function tools as `allowedFunctionNames`.

Settings that can't be combined raise before anything is sent:

- `thinking_level` with `thinking_budget`, and `voice` with `speakers`. When
  one comes from a model reference default, pass `None` for it on the call.
- `google_search_start_time` without `google_search_end_time`, and
  `file_search_top_k` or `file_search_metadata_filter` without
  `file_search_store_names`.
- Vertex AI only settings on the Gemini API: `google_search_exclude_domains`,
  `google_search_blocking_confidence`, safety `method`, `output_mime_type`,
  `output_compression_quality`, `person_generation`, `prominent_people`.

A cloned (replicated) voice has no flat field; send it through `extra`:

```python
config = GeminiTtsConfig(
    extra={'generationConfig': {'speechConfig': {'voiceConfig': {'replicatedVoiceConfig': {
        'mimeType': 'audio/wav',
        'voiceSampleAudio': sample_b64,
        'consentAudio': consent_b64,
    }}}}},
)
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

# 1. Initialize Genkit with GoogleAI
ai = Genkit(plugins=[GoogleAI()])

# 2. Launch background video generation
operation = await ai.generate_operation(
    model=GoogleAI.veo('veo-3.1-fast-generate-preview'),
    prompt='A paper airplane gliding through a bright classroom',
)
while not operation.done:
    operation = await ai.check_operation(operation)

# 3. Access output media artifact
print(operation.output.message.content[0].media.url)
# => https://generativelanguage.googleapis.com/v1beta/files/preview.mp4
```

**With `VertexAI`:**

```python
from genkit import Genkit
from genkit_google_genai import VertexAI

# 1. Initialize Genkit with VertexAI
ai = Genkit(plugins=[VertexAI()])

# 2. Launch background video generation
operation = await ai.generate_operation(
    model=VertexAI.veo('veo-3.1-generate-001'),
    prompt='A paper airplane gliding through a bright classroom',
)
while not operation.done:
    operation = await ai.check_operation(operation)

# 3. Access output media artifact
print(operation.output.message.content[0].media.url)
# => gs://my-bucket/videos/output.mp4
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
    model=VertexAI.virtual_try_on('virtual-try-on-001'),
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
