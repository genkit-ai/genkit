# Genkit Amazon Bedrock Plugin

Amazon Bedrock plugin for Genkit Python. Provides text generation with
Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral,
Cohere, and others) through the Bedrock Converse and ConverseStream APIs, and
embeddings, image generation, and reranking through InvokeModel.

> Status: text generation (streaming and non-streaming), embedders, image
> generation, and reranking are available. A runnable sample app covers all of
> it, and the plugin is documented on the Genkit docsite at
> [genkit.dev/docs/python/integrations/aws-bedrock](https://genkit.dev/docs/python/integrations/aws-bedrock/).

## Installation

```bash
pip install genkit-amazon-bedrock
```

## AWS setup

### Model access

Model access is granted per AWS account and per region, in the Bedrock console
under Model access, and most models cannot be called until it is: a grant in
`us-east-1` says nothing about `us-west-2`, so a working setup can break purely
by changing region.

The Anthropic models additionally need the account's one-time use-case
agreement (Bedrock console, Model access, Anthropic use case details). Until it
is accepted, Claude calls fail with `ResourceNotFoundException`, which reads
like a mistyped model ID rather than a missing agreement.

### IAM

The minimal policy covering everything this plugin does:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*",
        "arn:aws:bedrock:*:*:inference-profile/*"
      ]
    }
  ]
}
```

Converse is authorized by `bedrock:InvokeModel` and ConverseStream by
`bedrock:InvokeModelWithResponseStream`; there is no separate Converse action to
grant. Embedding, image generation and reranking all go through InvokeModel, so
they need only the first.

The inference-profile resource is the part that is easy to leave out.
Cross-region profile IDs such as `us.anthropic.claude-sonnet-4-5-20250929-v1:0`
are account-scoped inference-profile ARNs rather than foundation-model ARNs, so
a policy limited to `foundation-model/*` refuses them with
`AccessDeniedException` even when model access is granted.

### Credentials

Credentials resolve through the standard AWS SDK chain, so any of these work:

- environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
  `AWS_SESSION_TOKEN` for temporary credentials)
- the shared config and credentials files under `~/.aws`, selected with
  `AWS_PROFILE`
- IAM roles attached to EC2, ECS, or Lambda, supplied by the platform at runtime
- SSO profiles, after `aws sso login`

Anything the chain does not cover goes through `session=`, which takes a
pre-configured `boto3.session.Session`. Region resolution is separate from
credentials and is described under [Usage](#usage).

## Usage

```python
from genkit import Genkit
from genkit_amazon_bedrock import Bedrock, ModelDefinition

ai = Genkit(
    plugins=[
        Bedrock(
            region='us-east-1',
            models=[ModelDefinition(name='us.anthropic.claude-sonnet-4-5-20250929-v1:0')],
        )
    ],
    model='bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',
)
```

Credentials resolve through the standard AWS SDK chain (environment,
`~/.aws/credentials`, instance metadata). Pass a pre-configured
`boto3.session.Session` via `session=` for custom wiring. The region comes
from `region=` or the SDK chain (`AWS_REGION`, `AWS_DEFAULT_REGION`,
`~/.aws/config`); there is deliberately no default region.

### Plugin options

Every `Bedrock()` parameter, with its default:

| Option                 | Default  | Meaning                                                                                                                     |
| ---------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------- |
| `region`               | unset    | AWS region. Falls back to the SDK chain; initialization fails when nothing resolves.                                        |
| `max_retries`          | unset    | Retries after the initial attempt. Falls back to `3`, in botocore's `standard` mode, where your AWS configuration is silent. |
| `read_timeout`         | unset    | Socket read timeout in seconds. Falls back to `3600.0`.                                                                     |
| `connect_timeout`      | unset    | Socket connect timeout in seconds, covering the TCP handshake only. Falls back to `60.0`.                                   |
| `max_pool_connections` | unset    | HTTP connection pool size. Falls back to `50`, raised off botocore's default of 10.                                         |
| `total_timeout`        | `3600.0` | Whole-call deadline in seconds for a non-streaming generation, retries included. `None` removes it.                         |
| `session`              | unset    | Pre-configured `boto3.session.Session` for custom credentials or SDK wiring.                                                |
| `models`               | `[]`     | `ModelDefinition` entries to register. Unlisted IDs still resolve on demand.                                                |
| `embedders`            | `[]`     | Embedding model IDs to register. Unlisted IDs still resolve on demand.                                                      |

`max_retries`, `read_timeout`, `connect_timeout` and `max_pool_connections` are
unset by default, so your own AWS configuration wins and the fallbacks above
fill in only where it is silent. Passing an argument explicitly overrides both.

Which sources apply differs by knob, because botocore only reads some of them:

- Retries come from `AWS_MAX_ATTEMPTS`, `AWS_RETRY_MODE`, the `max_attempts`
  and `retry_mode` keys in `~/.aws/config`, or a session's default client
  config. The attempt count and the mode are resolved separately, so setting
  just one of them leaves the other at the package fallback, and passing
  `max_retries` tunes the count without dragging you off `adaptive`.
- The two timeouts and the pool size have no environment or config-file
  equivalent in botocore. Their only external source is a `botocore.config.Config`
  installed on a session you pass via `session=`.

**`read_timeout` is a socket read timeout, not a whole-call deadline.** It
bounds the wait for the next byte off the socket, not the length of the call,
which is why its fallback is a full hour: a generation can legitimately run for
many minutes (Nova allows 60-minute inference). Because it resets on every byte
received, a connection that dribbles bytes never trips it.

`total_timeout` is the wall-clock limit that does. It caps a non-streaming
generation end to end, retries included, and is on by default at 3600s; pass
`None` to remove it and leave only the socket timeouts. When the deadline fires
the caller gets a `DEADLINE_EXCEEDED` error, though the boto3 call itself cannot
be aborted and its worker thread runs until the socket timeouts end it.
Streaming generations, embeddings, image generation and reranking are bounded
by the socket timeouts alone.

`max_pool_connections` falls back to 50 rather than botocore's 10 so the pool is
never the bottleneck; concurrency is bounded first by the event loop's default
thread pool, which the boto3 calls are dispatched to.

`models` and `embedders` are what the Dev UI lists, so a bare `Bedrock()` lists
nothing. The plugin does not discover the catalogue for you: that would mean a
second client against the `bedrock` control plane and the IAM actions to go with
it, on top of the runtime-only policy [above](#iam). Calls are unaffected, since
any model ID resolves on demand either way.

### Config fields that are ignored

`BedrockConfig` inherits the core `ModelConfig` fields, so the Dev UI offers
`topK` and `version`. Converse has no equivalent parameters and both values
are dropped. Models that support a top-k knob take it
through `additionalModelRequestFields` instead:

```python
BedrockConfig(additional_model_request_fields={'top_k': 40})
```

### Inference profiles

Model IDs carrying one of the prefixes `global.`, `us-gov.`, `us.`, `eu.`,
`jp.`, `apac.`, or `au.` are cross-region inference profiles, which route a call
to whichever region in the geography has capacity:

```python
ModelDefinition(name='us.anthropic.claude-sonnet-4-5-20250929-v1:0')
```

The full ID is always sent to Bedrock verbatim. The prefix is stripped only for
the local capability lookup, so a profile inherits its base model's declared
capabilities instead of falling back to the unknown-model defaults. Several of
the newer models are only invocable through a profile, never by bare
foundation-model ID, so this is the normal form rather than an advanced option.
Any profile ID or full ARN is accepted, across the `arn:aws:`, `arn:aws-us-gov:`
and `arn:aws-cn:` partitions.

## Prompt caching

Bedrock can cache a prompt prefix and reuse it across calls, which is worth
doing whenever a large static system prompt is sent repeatedly.
`cache_point_part()` marks where the cacheable prefix ends:

```python
from genkit import Part, TextPart
from genkit_amazon_bedrock import cache_point_part

CLAUDE = 'bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0'

# The cache point goes after the content it should cache.
system = [Part(root=TextPart(text=LONG_STATIC_PROMPT)), cache_point_part()]

first = await ai.generate(model=CLAUDE, system=system, prompt='What are the delivery tiers?')
second = await ai.generate(model=CLAUDE, system=system, prompt='Which tier needs a signature?')

print(second.usage.cached_content_tokens)
```

Notes:

- The cache point goes **after** the content it should cache, never before.
- Cache points work in the system prompt and in ordinary user and model
  messages. System messages keep only text and cache points; every other part
  kind is dropped there.
- The cached prefix must be byte-identical across calls to hit, so build it from
  a constant rather than reassembling it per request.
- A prefix below the model's minimum cacheable size (roughly 1,000 tokens for
  Claude Sonnet, higher for the smaller models) is silently not cached. There is
  no error and no warning, so a short prompt just quietly never hits.
- The cache lives for a few minutes, which also means a re-run inside that
  window can read a cache the previous run wrote.
- On usage, `cacheReadInputTokens` surfaces as `usage.cached_content_tokens`.
  `cacheWriteInputTokens` is deliberately dropped, so a cache write is
  invisible: the second call reporting cached tokens the first did not is the
  only evidence you get.
- Bedrock counts cached tokens **outside** `inputTokens`, not within it, so
  `usage.input_tokens` is only the uncached remainder and stays small however
  well the cache works. The cached prefix appears in `usage.total_tokens`. Do
  not treat a small `input_tokens` as a cache failure, and do not assert on it
  to prove a cache hit.
- Passing a plain string as `system` sends it through Genkit's dotprompt
  templating, which rewrites the text and cannot express a cache point. A list
  of parts is what you want here, both to hold the cache point and to keep the
  cached prefix byte-identical.

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

## Reranking

Reranking scores documents against a query, so a retrieval step can return its
hits in relevance order. It is a method on the plugin instance rather than a
Genkit action, so keep a reference to the `Bedrock` you pass to `Genkit`:

```python
from genkit import Document, Genkit
from genkit_amazon_bedrock import Bedrock, BedrockRerankOptions

bedrock = Bedrock(region='us-east-1')
ai = Genkit(plugins=[bedrock])

response = await bedrock.rerank(
    'cohere.rerank-v3-5:0',
    query='How do I configure authentication for Bedrock?',
    documents=[
        Document.from_text('Configure AWS credentials with environment variables or AWS SSO.'),
        Document.from_text('Nova Canvas returns generated images as base64-encoded PNG data.'),
        Document.from_text('Model access is granted per account and region in the Bedrock console.'),
    ],
    options=BedrockRerankOptions(top_n=2),
)

for document in response.documents:
    print(document.metadata.score, document.content[0].root.text)
```

Genkit Python has no reranker primitive: `ActionKind.RERANKER` exists as a bare
enum member, and the request and response types are not generated, so there is
nothing to register an action against. The types this plugin exports
(`BedrockRerankOptions`, `RankedDocumentData`, `RankedDocumentMetadata`,
`RerankerRequest`, `RerankerResponse`) mirror the schema types by the same
names.

Notes:

- `top_n` (`topN` when the options are passed as a dict) is clamped down to the
  number of documents sent, and `<= 0` or unset means all of them.
- Results arrive in the service's descending-relevance order and are neither
  re-sorted nor truncated client-side.
- A ranked document carries the input document's content verbatim and fresh
  `{score}` metadata. The input document's own metadata is not carried through.
- Rerank models have no Converse path, so they never resolve as chat models.
  Listing one in `models=` is ignored; pass the ID to `rerank()` instead.
- Only `bedrock:InvokeModel` is required. `bedrock:Rerank` is not: that
  permission belongs to the separate Bedrock Agent Runtime `Rerank` API, which
  this plugin does not call.

### Model support

Verified against `aws bedrock get-foundation-model` on 2026-08-11, and matching
AWS's [supported regions and models for reranking](https://docs.aws.amazon.com/bedrock/latest/userguide/rerank-supported.html):

| Model                  | Status | Regions                                                                    |
| ---------------------- | ------ | -------------------------------------------------------------------------- |
| `cohere.rerank-v3-5:0` | Active | `us-east-1`, `us-west-2`, `eu-central-1`, `ap-northeast-1`, `ca-central-1` |
| `amazon.rerank-v1:0`   | Active | `us-west-2`, `eu-central-1`, `ap-northeast-1`, `ca-central-1`              |

**`amazon.rerank-v1:0` is not offered in `us-east-1`.** Only
`cohere.rerank-v3-5:0` covers every region on the list, so a setup that defaults
to `us-east-1` has one rerank model available, not two.

The two families take different bodies, so the request is built from the model
ID. Both send `query`, `documents` and `top_n`; only Cohere takes
`api_version`, whose schema requires the key, while the Amazon schema rejects
any body carrying it. An ID matching neither family gets the Cohere body, that
being the only shape AWS documents for InvokeModel reranking.

Any model ID is passed to InvokeModel verbatim, so inference profiles and ARNs
work too.

## Troubleshooting

The plugin maps Bedrock error codes onto Genkit error statuses, so these arrive as typed `GenkitError` (`PERMISSION_DENIED`, `NOT_FOUND`, `INVALID_ARGUMENT`, `RESOURCE_EXHAUSTED`, and the rest) with the service's own message preserved. The AWS code is the useful part, and each one means something specific:

**No region resolved.** The plugin fails at initialization rather than on the
first call, and deliberately does not default to a region: a silent `us-east-1`
fallback would send traffic and data somewhere you never chose. Set `region=`,
`AWS_REGION`, `AWS_DEFAULT_REGION`, or a region on the active profile.

**`AccessDeniedException`.** Either model access has not been granted for that
model in that region, or the IAM policy is missing the inference-profile
resource. Both are covered under [AWS setup](#aws-setup); check the policy first
if the model ID carries a cross-region prefix, because that failure looks
identical to a missing grant.

**`ResourceNotFoundException`.** The model exists but this account cannot use
it there: an Anthropic model whose use-case agreement has not been accepted, or
a Legacy model. The Nova Canvas case is its own trap and is described under
[Availability](#availability). A model that plainly is not offered in the
region reports as a `ValidationException` instead, below.

**`ValidationException` reading "The provided model identifier is invalid".**
The ID is mistyped, or the model is not offered in the region the call went to.
Both Converse and InvokeModel report a missing model this way, verified live,
so it is what a region mistake actually looks like: calling
`stability.sd3-5-large-v1:0` (us-west-2 only) from a session whose default
region is `us-east-1` produces exactly this. Check the region before doubting
the ID.

**Any other `ValidationException`.** The request or config is malformed for
that particular model: a thinking budget outside the bounds the model allows,
or an image config field the family does not accept (the whole config dict
reaches Stability, and everything inside `imageGenerationConfig` reaches the
Amazon family). Bedrock validates per model, so a config that works on one
model can be rejected by the next.

**`ThrottlingException`.** The on-demand capacity limit for that model and
region. The plugin retries automatically with exponential backoff, up to
`max_retries` retries after the initial attempt, and surfaces the error once
those are exhausted. Raise `max_retries`, spread the load, or move to
provisioned throughput.

### Debug logging

`GENKIT_LOG=debug` turns on the plugin's own log lines: the resolved region and
client config, which model each call routed to, stop reasons, token counts, and
what `resolve` declined. They carry metadata only, never prompts, embeddings or
image bytes, so they are safe to leave on.

## Examples

See [`py/samples/amazon-bedrock-sample`](../../samples/amazon-bedrock-sample)
for a runnable sample covering every surface here: text generation and
streaming, structured output, tool calling, reasoning and extended thinking,
embedders, prompt caching, image and document input, image generation, and
reranking. Each one is a flow, so they can be run individually in the Dev UI.

## License

Apache 2.0
