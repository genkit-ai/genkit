# Genkit Amazon Bedrock Plugin

Amazon Bedrock plugin for Genkit Python. Provides text generation with
Bedrock-hosted models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral,
Cohere, and others) through the Bedrock Converse and ConverseStream APIs.

> Status: in progress. Text generation is available, streaming and
> non-streaming. Embedders, image generation, and reranking are still being
> ported from the mature Go plugin
> ([genkit-ai/aws-bedrock-go-plugin](https://github.com/genkit-ai/aws-bedrock-go-plugin)).

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

### Config fields that are ignored

`BedrockConfig` inherits the core `ModelConfig` fields, so the Dev UI offers
`topK` and `version`. Converse has no equivalent parameters and both values
are dropped, matching the Go plugin. Models that support a top-k knob take it
through `additionalModelRequestFields` instead:

```python
BedrockConfig(additional_model_request_fields={'top_k': 40})
```

## License

Apache 2.0
