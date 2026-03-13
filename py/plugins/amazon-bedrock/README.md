# Genkit Amazon Bedrock Plugin

> **Community Plugin** – This plugin is maintained by the community and is supported on a best-effort basis. It is not an official AWS product.
>
> **Preview** — This plugin is in preview and may have API changes in future releases.

`genkit-plugin-amazon-bedrock` is a plugin for using Amazon Bedrock models with [Genkit](https://github.com/firebase/genkit). It provides access to foundation models AND AWS X-Ray telemetry in a single package.

Amazon Bedrock is a fully managed service that provides access to foundation models from leading AI providers including Amazon, Anthropic, Meta, Mistral, Cohere, DeepSeek, and more through a unified API.

## Documentation Links

- **Amazon Bedrock Console**: https://console.aws.amazon.com/bedrock/
- **Supported Models**: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- **Model Parameters**: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
- **Converse API**: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
- **Boto3 Bedrock Runtime**: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime.html

## Installation

```bash
pip install genkit-plugin-amazon-bedrock
```

## Setup

You'll need AWS credentials configured. The plugin supports multiple authentication methods.

### Option 1: Bedrock API Key (Simplest)

Amazon Bedrock now supports API keys similar to OpenAI/Anthropic:

```bash
export AWS_REGION="us-east-1"
export AWS_BEARER_TOKEN_BEDROCK="your-api-key"
```

Generate an API key in [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/) > API keys.

**Important**: API keys require [inference profiles](#cross-region-inference-profiles) instead of direct model IDs. Use the `inference_profile()` helper:

```python
from genkit.plugins.amazon_bedrock import inference_profile

model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
```

See: [Getting Started with Bedrock API Keys](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api-keys.html)

### Option 2: IAM Credentials (Recommended for Production)

```bash
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```

### Option 3: AWS Profile

```bash
export AWS_PROFILE="your-profile-name"
export AWS_REGION="us-east-1"
```

### Option 4: IAM Role (AWS Infrastructure)

When running on EC2, Lambda, ECS, or EKS, credentials are automatically provided by the IAM role.

```bash
export AWS_REGION="us-east-1"
# No credentials needed - IAM role provides them
```

### IAM Permissions (for Options 2-4)

Your AWS credentials need the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/*"
    }
  ]
}
```

## AWS X-Ray Telemetry

This plugin includes built-in AWS X-Ray telemetry support. Enable it to send distributed traces to AWS X-Ray console:

```python
from genkit.plugins.amazon_bedrock import add_aws_telemetry

# Enable X-Ray telemetry (call once at startup)
add_aws_telemetry(region="us-east-1")
```

Traces will be visible in the [AWS X-Ray Console](https://console.aws.amazon.com/xray/home).

### IAM Permissions for X-Ray

Add these permissions for telemetry:

```json
{
  "Effect": "Allow",
  "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords"],
  "Resource": "*"
}
```

Or attach the managed policy: `AWSXrayWriteOnlyPolicy`

## Basic Usage

```python
from genkit import Genkit
from genkit.plugins.amazon_bedrock import AmazonBedrock, bedrock_model

ai = Genkit(
    plugins=[
        AmazonBedrock(region="us-east-1")
    ],
    model=bedrock_model("anthropic.claude-sonnet-4-5-20250929-v1:0"),
)

response = await ai.generate(prompt="Tell me a joke.")
print(response.text)
```

### With Explicit Credentials

```python
ai = Genkit(
    plugins=[
        AmazonBedrock(
            region="us-east-1",
            access_key_id="your-access-key",
            secret_access_key="your-secret-key",
        )
    ],
    model=bedrock_model("anthropic.claude-sonnet-4-5-20250929-v1:0"),
)
```

### With AWS Profile

```python
ai = Genkit(
    plugins=[
        AmazonBedrock(
            region="us-east-1",
            profile_name="my-aws-profile",
        )
    ],
    model=bedrock_model("anthropic.claude-sonnet-4-5-20250929-v1:0"),
)
```

## Supported Model Providers

| Provider | Model Examples | Model ID Prefix |
|----------|---------------|-----------------|
| Amazon | Nova Pro, Nova Lite, Nova Micro | `amazon.nova-*` |
| Anthropic | Claude Sonnet 4.5, Claude Opus 4.5 | `anthropic.claude-*` |
| AI21 Labs | Jamba 1.5 Large, Jamba 1.5 Mini | `ai21.jamba-*` |
| Cohere | Command R, Command R+ | `cohere.command-*` |
| DeepSeek | DeepSeek-R1, DeepSeek-V3 | `deepseek.*` |
| Google | Gemma 3 4B, Gemma 3 12B | `google.gemma-*` |
| Meta | Llama 3.3 70B, Llama 4 Maverick | `meta.llama*` |
| MiniMax | MiniMax M2 | `minimax.*` |
| Mistral | Mistral Large 3, Pixtral Large | `mistral.*` |
| Moonshot | Kimi K2 Thinking | `moonshot.*` |
| NVIDIA | Nemotron Nano 9B, 12B | `nvidia.*` |
| OpenAI | GPT-OSS 120B, GPT-OSS 20B | `openai.*` |
| Qwen | Qwen3 32B, Qwen3 235B | `qwen.*` |
| Writer | Palmyra X4, Palmyra X5 | `writer.*` |

## Model Examples

### Anthropic Claude

```python
from genkit.plugins.amazon_bedrock import (
    AmazonBedrock,
    bedrock_model,
    claude_sonnet_4_5,
    claude_opus_4_5,
)

ai = Genkit(
    plugins=[AmazonBedrock(region="us-east-1")],
    model=claude_sonnet_4_5,
)

response = await ai.generate(prompt="Explain quantum computing")
```

### Amazon Nova

```python
from genkit.plugins.amazon_bedrock import nova_pro, nova_lite

response = await ai.generate(
    model=nova_pro,
    prompt="Describe the image",
    # Nova supports images and video
)
```

### Meta Llama

```python
from genkit.plugins.amazon_bedrock import llama_3_3_70b, llama_4_maverick

response = await ai.generate(
    model=llama_4_maverick,
    prompt="Write a poem about AI",
)
```

### DeepSeek R1 (Reasoning)

```python
from genkit.plugins.amazon_bedrock import deepseek_r1

response = await ai.generate(
    model=deepseek_r1,
    prompt="Solve this step by step: What is 15% of 240?",
)
# Response includes reasoning_content for reasoning models
```

## Configuration

The plugin supports model-specific configuration parameters:

```python
from genkit.plugins.amazon_bedrock import BedrockConfig

response = await ai.generate(
    prompt="Tell me a story",
    config=BedrockConfig(
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        stop_sequences=["THE END"],
    ),
)
```

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `temperature` | float (0.0-1.0) | Sampling temperature. Higher = more random. |
| `max_tokens` | int | Maximum tokens to generate. |
| `top_p` | float (0.0-1.0) | Nucleus sampling probability. |
| `top_k` | int | Top-k sampling (model-specific). |
| `stop_sequences` | list[str] | Stop sequences to end generation. |

### Model-Specific Configs

Each model family has its own configuration class with provider-specific parameters:

```python
from genkit.plugins.amazon_bedrock import AnthropicConfig, CohereConfig, MetaLlamaConfig

# Anthropic Claude with top_k
response = await ai.generate(
    model=claude_sonnet_4_5,
    prompt="...",
    config=AnthropicConfig(
        temperature=0.7,
        top_k=40,
    ),
)

# Cohere with documents for RAG
response = await ai.generate(
    model=bedrock_model("cohere.command-r-plus-v1:0"),
    prompt="...",
    config=CohereConfig(
        temperature=0.5,
        k=50,
        p=0.9,
    ),
)

# Meta Llama
response = await ai.generate(
    model=llama_3_3_70b,
    prompt="...",
    config=MetaLlamaConfig(
        temperature=0.6,
        max_gen_len=1024,
    ),
)
```

## Multimodal Support

Models like Claude, Nova, and Llama 4 support images:

```python
from genkit.types import Media, MediaPart, Part, TextPart

response = await ai.generate(
    model=claude_sonnet_4_5,
    prompt=[
        Part(root=TextPart(text="What's in this image?")),
        Part(root=MediaPart(media=Media(url="https://example.com/image.jpg"))),
    ],
)
```

## Embeddings

```python
from genkit.blocks.document import Document

# Amazon Titan Embeddings
response = await ai.embed(
    embedder="amazon-bedrock/amazon.titan-embed-text-v2:0",
    input=[Document.from_text("Hello, world!")],
)

# Cohere Embeddings
response = await ai.embed(
    embedder="amazon-bedrock/cohere.embed-english-v3",
    input=[Document.from_text("Hello, world!")],
)
```

## Cross-Region Inference Profiles

**When using API keys (`AWS_BEARER_TOKEN_BEDROCK`)**, you must use inference profiles instead of direct model IDs. The plugin provides helpers for this:

### Inference Profile Helper

```python
from genkit.plugins.amazon_bedrock import inference_profile

# Auto-detects region from AWS_REGION environment variable
model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0')
# → 'amazon-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0' (if AWS_REGION=us-east-1)

# Or specify region explicitly
model = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'eu-west-1')
# → 'amazon-bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0'
```

### Regional Prefixes

| Region | Prefix | Example Regions |
|--------|--------|-----------------|
| United States | `us.` | us-east-1, us-west-2 |
| Europe | `eu.` | eu-west-1, eu-central-1 |
| Asia Pacific | `apac.` | ap-northeast-1, ap-southeast-1 |

### When to Use Inference Profiles

| Auth Method | Model ID Format |
|-------------|-----------------|
| API Key (`AWS_BEARER_TOKEN_BEDROCK`) | Inference profile required: `us.anthropic.claude-...` |
| IAM Credentials | Direct model ID works: `anthropic.claude-...` |

See: [Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)

## Streaming

Streaming is supported via the Genkit streaming API:

```python
async for chunk in ai.generate_stream(
    model=claude_sonnet_4_5,
    prompt="Write a long story",
):
    print(chunk.text, end="", flush=True)
```

## Tool Use (Function Calling)

```python
from genkit.ai import tool

@tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny."

response = await ai.generate(
    model=claude_sonnet_4_5,
    prompt="What's the weather in Tokyo?",
    tools=[get_weather],
)
```

## References

### AWS Documentation

- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
- [Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- [Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)
- [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html)
- [Tool Use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html)

### Model Provider Documentation

- [Anthropic Claude](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- [Amazon Nova](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html)
- [Meta Llama](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html)
- [Mistral AI](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html)
- [Cohere](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html)
- [DeepSeek](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html)

## Disclaimer

This is a community plugin and is not officially supported or endorsed by Amazon Web Services.

"Amazon", "AWS", "Amazon Bedrock", and related marks are trademarks of Amazon.com, Inc.
or its affiliates. This plugin is developed independently and is not affiliated with,
endorsed by, or sponsored by Amazon.

The use of AWS APIs is subject to AWS's terms of service. Users are responsible for
ensuring their usage complies with AWS's API terms and any applicable rate limits or
usage policies.

## License

Apache 2.0
