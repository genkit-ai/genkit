# Genkit Amazon Bedrock Plugin

Amazon Bedrock plugin for Genkit Python. Provides text generation with
Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral,
Cohere, and others) through the Bedrock Converse and ConverseStream APIs, and
embeddings and image generation through InvokeModel.

> Status: in progress. Text generation (streaming and non-streaming),
> embedders, and image generation are available. Reranking is still to come.

## Installation

```bash
pip install genkit-amazon-bedrock
```

## Usage

```python
from genkit import Genkit
from genkit_amazon_bedrock import Bedrock, ModelDefinition

ai = Genkit(
    plugins=[
        Bedrock(
            region='us-east-1',
            models=[ModelDefinition(name='anthropic.claude-sonnet-4-5-20250929-v1:0')],
        )
    ],
    model='bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0',
)
```

Credentials resolve through the standard AWS SDK chain (environment,
`~/.aws/credentials`, instance metadata). Pass a pre-configured
`boto3.session.Session` via `session=` for custom wiring. The region comes
from `region=` or the SDK chain (`AWS_REGION`, `AWS_DEFAULT_REGION`,
`~/.aws/config`); there is deliberately no default region.

### Client tuning

`max_retries`, `read_timeout`, `connect_timeout` and `max_pool_connections`
are unset by default, so your own AWS configuration wins and the package
defaults (3 retries in standard mode, a 3600s read timeout, a 60s connect
timeout and a pool of 50) fill in only where it is silent. Passing an argument
explicitly overrides both.

Which sources apply differs by knob, because botocore only reads some of them:

- Retries come from `AWS_MAX_ATTEMPTS`, `AWS_RETRY_MODE`, the `max_attempts`
  and `retry_mode` keys in `~/.aws/config`, or a session's default client
  config. The attempt count and the mode are resolved separately, so setting
  just one of them leaves the other at the package default.
- The two timeouts and the pool size have no environment or config-file
  equivalent in botocore. Their only external source is a `botocore.config.Config`
  installed on a session you pass via `session=`.

`total_timeout` (3600s, `None` to disable) is separate: it caps the whole
call, including retries. The read timeout only bounds silence between two
reads, so a connection that dribbles bytes never trips it. When the deadline
fires the caller gets a `DEADLINE_EXCEEDED` error, though the boto3 call
itself cannot be aborted and its worker thread runs until the socket timeouts
end it.

### Config fields that are ignored

`BedrockConfig` inherits the core `ModelConfig` fields, so the Dev UI offers
`topK` and `version`. Converse has no equivalent parameters and both values
are dropped. Models that support a top-k knob take it
through `additionalModelRequestFields` instead:

```python
BedrockConfig(additional_model_request_fields={'top_k': 40})
```

## Embedders

Embedding models are listed separately from chat models, by bare model ID:

```python
from genkit import Genkit
from genkit_amazon_bedrock import Bedrock

ai = Genkit(plugins=[Bedrock(embedders=['amazon.titan-embed-text-v2:0'])])

embeddings = await ai.embed(
    embedder='bedrock/amazon.titan-embed-text-v2:0',
    content='Bedrock hosts models from several providers.',
)
```

| Model                                      | Dimensions | Modalities   |
| ------------------------------------------ | ---------- | ------------ |
| `amazon.titan-embed-text-v1`               | 1536       | text         |
| `amazon.titan-embed-text-v2:0`             | 1024       | text         |
| `amazon.titan-embed-image-v1`              | 1024       | text, image  |
| `cohere.embed-english-v3`                  | 1024       | text         |
| `cohere.embed-multilingual-v3`             | 1024       | text         |
| `amazon.nova-2-multimodal-embeddings-v1:0` | 3072       | text         |

Notes:

- As with models, listing is optional: any routable embedding model ID
  resolves on demand, including inference-profile and ARN forms. Listing one
  only adds it to the Dev UI.
- `amazon.titan-embed-image-v1` is the only image embedder here. It accepts
  JPEG and PNG, one image per document, and averages the two vectors when a
  document carries text as well.
- **Cohere embedding is text-only on Bedrock.** The AWS parameters page
  documents an `images` field for Embed v3, but `aws bedrock
  get-foundation-model` reports `inputModalities: [TEXT]` for both v3 model IDs
  and a live image request is refused. A document with no text is rejected
  rather than sent; a media part alongside text is ignored.
- `cohere.embed-v4` is not supported yet -- it uses a different request schema and fails with `UNIMPLEMENTED`. It is also where Cohere image embedding will arrive.
- Per-request options are not honoured yet. Cohere requests are pinned to
  `input_type: search_document`, so these embedders are for indexing
  documents, not for embedding queries.
- `amazon.nova-2-multimodal-embeddings-v1:0` is offered in `us-east-1` only,
  and is text-only here; its image, audio, and video inputs are not ported.

## Image generation

Image models go through InvokeModel rather than Converse, but they are ordinary
Genkit model actions: `ai.generate` returns the result as media parts carrying
data URLs. Declare one with `type='image'`:

```python
from genkit import Genkit
from genkit_amazon_bedrock import Bedrock, ModelDefinition

ai = Genkit(
    plugins=[
        Bedrock(
            # us-west-2: the active text-to-image models are offered there only.
            region='us-west-2',
            models=[ModelDefinition(name='stability.sd3-5-large-v1:0', type='image')],
        )
    ]
)

response = await ai.generate(
    model='bedrock/stability.sd3-5-large-v1:0',
    prompt='A tabby cat asleep on a sunlit windowsill, watercolour.',
)
image = response.media[0].url  # data:image/png;base64,...
```

Declaring is optional here too. An undeclared ID in one of the two families
below is classified as an image model on the spot, so a bare
`ai.generate(model='bedrock/<id>')` resolves and takes the InvokeModel path;
declaring adds the model to the Dev UI and pins the routing. The prompt is the
text of the most recent user message, concatenated across its text parts; other
parts are ignored, as these models are text-to-image only.

Streaming callbacks are never invoked for image models. There is nothing to
stream, so `generate_stream` yields no chunks and the images arrive on the
final response.

### Request shapes

The two families take incompatible bodies, so the request is built from the
model ID.

Amazon (IDs containing `titan-image` or `nova-canvas`) nests its options under
`imageGenerationConfig`, and that is the only config key read: any other
top-level key is dropped. Your entries are merged key by key over these
defaults:

```python
config={
    'imageGenerationConfig': {
        'numberOfImages': 1,
        'height': 1024,
        'width': 1024,
        'cfgScale': 8.0,
        'seed': 0,
        'quality': 'standard',  # Nova Canvas only, not sent for Titan Image
    }
}
```

Output is always `image/png`.

Stability (IDs containing `sd3-`, `stable-image-core`, or `stable-image-ultra`)
takes flat top-level fields, and the whole config dict is merged over the
defaults (`{'prompt': ..., 'output_format': 'png'}`), so `aspect_ratio`,
`seed`, `negative_prompt`, `output_format` and the rest all apply:

```python
config={'aspect_ratio': '16:9', 'output_format': 'jpeg', 'seed': 42}
```

The media part's MIME type follows `output_format`.

`BedrockImageConfig` is the exported config type for these calls. It declares
no fields and rejects nothing, on purpose: the two families take disjoint keys,
and `BedrockConfig` describes Converse parameters, so it would reject every
family-specific key here.

Genkit's generic generation options (`temperature`, `topP`, `maxOutputTokens`,
`apiKey`, and the rest of the common config) are not forwarded to image models,
since Bedrock's image APIs do not accept them.

### Availability

Verified against `aws bedrock get-foundation-model` on 2026-08-11:

| Model                               | Status                         | Regions                                    |
| ----------------------------------- | ------------------------------ | ------------------------------------------ |
| `stability.sd3-5-large-v1:0`        | Active                         | `us-west-2`                                |
| `stability.stable-image-core-v1:1`  | Active                         | `us-west-2`                                |
| `stability.stable-image-ultra-v1:1` | Active                         | `us-west-2`                                |
| `amazon.nova-canvas-v1:0`           | Legacy, end of life 2026-09-30 | `us-east-1`, `eu-west-1`, `ap-northeast-1` |

**The active text-to-image models are offered in `us-west-2` only.** Every image
model on offer in `us-east-1` is an editing service (inpaint, upscale, and the
rest), not text-to-image, so a text-to-image call needs `us-west-2`.

`amazon.nova-canvas-v1:0` is the one Amazon option and it is Legacy, which is a
stronger restriction than the end-of-life date suggests: new accounts cannot
enable it at all, and an account that has enabled it loses access after 30 days
of disuse. Bedrock reports both as `ResourceNotFoundException`, which the plugin
surfaces verbatim, so a working integration can start failing without any change
on your side. Prefer a Stability model unless you specifically need Canvas.

Notes:

- Amazon Titan Image Generator (v1 and v2), Stable Diffusion XL, and the `v1:0`
  Stability SKUs (`sd3-large-v1:0`, `stable-image-core-v1:0`,
  `stable-image-ultra-v1:0`) are end of life on Bedrock and no longer callable.
- The `titan-image` family is still recognised, since it shares its request
  shape with Nova Canvas. The legacy Stable Diffusion XL schema
  (`text_prompts`/`artifacts`) is deliberately not ported.
- `amazon.nova-canvas-v1:0` has no `us.` cross-region inference profile, so
  those three regions are the whole of it.
- Stability's `stable-image-*` editing services (inpaint, erase object, search
  and replace, background removal, style guide, control sketch) are not
  text-to-image and are out of scope.
- Nova Canvas may return fewer images than `numberOfImages` asked for.
  Individually content-filtered images are dropped silently, and the plugin
  returns whatever arrived.

## License

Apache 2.0
