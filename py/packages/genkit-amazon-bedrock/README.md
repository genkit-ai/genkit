# Genkit Amazon Bedrock Plugin

Amazon Bedrock plugin for Genkit Python. Provides text generation with
Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral,
Cohere, and others) through the Bedrock Converse and ConverseStream APIs, and
embeddings through InvokeModel.

> Status: in progress. Text generation (streaming and non-streaming) and
> embedders are available. Image generation and reranking are still to come.

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

## License

Apache 2.0
