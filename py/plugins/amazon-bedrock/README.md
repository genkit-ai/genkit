# Genkit Amazon Bedrock Plugin

Amazon Bedrock plugin for Genkit Python. Provides access to Bedrock-hosted
models (Anthropic Claude, Amazon Nova, Meta Llama, Mistral, Cohere, and
others), Titan/Cohere/Nova embedders, image generation, and Cohere reranking.

> Status: scaffold. Feature slices are being ported from the mature Go plugin
> ([genkit-ai/aws-bedrock-go-plugin](https://github.com/genkit-ai/aws-bedrock-go-plugin)).

## Installation

```bash
pip install genkit-plugin-amazon-bedrock
```

## Usage

```python
from genkit import Genkit
from genkit.plugins.amazon_bedrock import Bedrock, ModelDefinition

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
`aioboto3.Session` via `session=` for custom wiring.

## License

Apache 2.0
