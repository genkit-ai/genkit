# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Amazon Bedrock hello sample - Foundation models and observability with Genkit.

This sample demonstrates how to use AWS Bedrock models with Genkit,
including tools, streaming, multimodal, embedding, and AWS X-Ray telemetry.

See README.md for setup and testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ AWS Bedrock         │ Amazon's AI model marketplace. One place to        │
    │                     │ access Claude, Llama, Nova, and more.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Converse API        │ A unified way to talk to ANY Bedrock model.        │
    │                     │ Same code works for Claude, Llama, Nova, etc.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference Profile   │ A cross-region alias for a model. Required         │
    │                     │ when using API keys instead of IAM roles.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ IAM Role            │ AWS's way of granting permissions. Like a          │
    │                     │ badge that lets your code access models.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Region              │ Which AWS data center to use. Pick one near        │
    │                     │ you (us-east-1, eu-west-1, ap-northeast-1).        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ AWS X-Ray           │ Amazon's distributed tracing service. See how      │
    │                     │ requests flow through your AI application.         │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet          |
|-----------------------------------------|------------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[AmazonBedrock()])` |
| AWS X-Ray Telemetry                     | `add_aws_telemetry(region=...)`          |
| Default Model Configuration             | `ai = Genkit(model=...)`                 |
| Defining Flows                          | `@ai.flow()` decorator                   |
| Defining Tools                          | `@ai.tool()` decorator                   |
| Pydantic for Tool Input Schema          | `WeatherInput`, `CurrencyInput`          |
| Simple Generation (Prompt String)       | `generate_greeting`                      |
| System Prompts                          | `generate_with_system_prompt`            |
| Multi-turn Conversations (`messages`)   | `generate_multi_turn_chat`               |
| Streaming Generation                    | `generate_streaming_story`               |
| Generation with Tools                   | `generate_weather`, `convert_currency`   |
| Generation Configuration (temperature)  | `generate_with_config`                   |
| Multimodal (Image Input)                | `describe_image`                         |
| Code Generation                         | `generate_code`                          |
| Embeddings                              | `embed_text`                             |

Supported Models
================
- Claude (Anthropic): claude-sonnet-4-5, claude-opus-4-5, claude-haiku-4-5
- Nova (Amazon): nova-pro, nova-lite, nova-micro
- Llama (Meta): llama-3.3-70b, llama-4-maverick
- Mistral: mistral-large-3, pixtral-large
- DeepSeek: deepseek-r1, deepseek-v3
- And many more...
"""

import asyncio
import os

from genkit.ai import Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.amazon_bedrock import (
    AmazonBedrock,
    add_aws_telemetry,
    bedrock_model,
    claude_opus_4_6,
    claude_sonnet_4_5,
    deepseek_r1,
    inference_profile,
    nova_pro,
)
from genkit.types import Media, MediaPart, Part, TextPart
from samples.shared import (
    CharacterInput,
    CodeInput,
    CurrencyExchangeInput,
    EmbedInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    WeatherInput,
    convert_currency as _convert_currency_tool,
    convert_currency_logic,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_weather_logic,
    generate_with_config_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
)

setup_sample()

# Prompt for AWS region if not set
if 'AWS_REGION' not in os.environ:
    os.environ['AWS_REGION'] = input('Please enter your AWS_REGION (e.g., us-east-1): ')

logger = get_logger(__name__)

# Enable AWS X-Ray telemetry (traces exported to X-Ray console)
# This provides distributed tracing for all Genkit flows and model calls
# View traces at: https://console.aws.amazon.com/xray/home
add_aws_telemetry(region=os.environ.get('AWS_REGION'))

# Default model configuration
# Model IDs without regional prefix - used as base for both auth methods
_CLAUDE_SONNET_MODEL_ID = 'anthropic.claude-sonnet-4-5-20250929-v1:0'
_CLAUDE_OPUS_MODEL_ID = 'anthropic.claude-opus-4-6-20260205-v1:0'
_DEEPSEEK_R1_MODEL_ID = 'deepseek.r1-v1:0'
_NOVA_PRO_MODEL_ID = 'amazon.nova-pro-v1:0'
_TITAN_EMBED_MODEL_ID = 'amazon.titan-embed-text-v2:0'

# Detect authentication method and choose appropriate model reference
# - API keys (AWS_BEARER_TOKEN_BEDROCK) require inference profiles
# - IAM credentials work with direct model IDs
_using_api_key = 'AWS_BEARER_TOKEN_BEDROCK' in os.environ

# Choose models based on auth method
# API keys require inference profiles with regional prefix (us., eu., apac.)
# IAM credentials work with direct model IDs
if _using_api_key:
    _default_model = inference_profile(_CLAUDE_SONNET_MODEL_ID)
    _opus_model = inference_profile(_CLAUDE_OPUS_MODEL_ID)
    _deepseek_model = inference_profile(_DEEPSEEK_R1_MODEL_ID)
    _nova_model = inference_profile(_NOVA_PRO_MODEL_ID)
    _embed_model = inference_profile(_TITAN_EMBED_MODEL_ID)
    logger.info('Using API key auth - model IDs will use inference profiles')
else:
    _default_model = claude_sonnet_4_5
    _opus_model = claude_opus_4_6
    _deepseek_model = deepseek_r1
    _nova_model = nova_pro
    _embed_model = bedrock_model(_TITAN_EMBED_MODEL_ID)
    logger.info('Using IAM credentials - model IDs are direct')

logger.info('AWS X-Ray telemetry enabled - traces visible in X-Ray console')

# Initialize Genkit with AWS Bedrock plugin
# Region is required - either from env var (prompted above) or explicit
ai = Genkit(
    plugins=[AmazonBedrock(region=os.environ.get('AWS_REGION'))],
    model=_default_model,
)

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        Greeting message.
    """
    return await generate_greeting_logic(ai, input.name)


@ai.flow()
async def generate_with_system_prompt(input: SystemPromptInput) -> str:
    """Demonstrate system prompts to control model persona and behavior.

    Args:
        input: Input with a question to ask.

    Returns:
        The model's response in the persona defined by the system prompt.
    """
    return await generate_with_system_prompt_logic(ai, input.question)


@ai.flow()
async def generate_multi_turn_chat(input: MultiTurnInput) -> str:
    """Demonstrate multi-turn conversations using the messages parameter.

    Args:
        input: Input with a travel destination.

    Returns:
        The model's final response, demonstrating context retention.
    """
    return await generate_multi_turn_chat_logic(ai, input.destination)


@ai.flow()
async def generate_streaming_story(
    input: StreamInput,
    ctx: ActionRunContext = None,  # type: ignore[assignment]
) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming story.
        ctx: Action run context for streaming.

    Returns:
        Complete generated text.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


@ai.flow()
async def generate_with_config(input: GreetingInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with name to greet.

    Returns:
        Greeting message.
    """
    return await generate_with_config_logic(ai, input.name)


@ai.flow()
async def generate_weather(input: WeatherInput) -> str:
    """Get weather information using tool calling.

    Args:
        input: Input with location to get weather for.

    Returns:
        Weather information.
    """
    return await generate_weather_logic(ai, input)


@ai.flow()
async def convert_currency(input: CurrencyExchangeInput) -> str:
    """Convert currency using tool calling.

    Args:
        input: Currency exchange parameters.

    Returns:
        Conversion result.
    """
    return await convert_currency_logic(ai, input)


@ai.flow()
async def generate_character(input: CharacterInput) -> RpgCharacter:
    """Generate an RPG character with structured output.

    Args:
        input: Input with character name.

    Returns:
        The generated RPG character.
    """
    return await generate_character_logic(ai, input.name)


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Claude or Nova (multimodal models).

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url)


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using AWS Bedrock models.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    # NOTE: Claude Opus 4.6 (_opus_model) is ideal for complex code generation,
    # but it may not be available in all regions yet (released Feb 5, 2026).
    # Swap to _opus_model once it's enabled in your region's inference profiles.
    return await generate_code_logic(ai, input.task, model=_default_model)


@ai.flow()
async def generate_streaming_with_tools(
    input: StreamingToolInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Demonstrate streaming generation with tool calling.

    Args:
        input: Input with location for weather lookup.
        ctx: Action context for streaming chunks to the client.

    Returns:
        The complete generated text.
    """
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


@ai.flow()
async def describe_image_nova(input: ImageDescribeInput) -> str:
    """Describe an image using Amazon Nova Pro.

    When using API keys (AWS_BEARER_TOKEN_BEDROCK), this automatically uses
    the inference profile version of the model (e.g., us.amazon.nova-pro-v1:0).

    Args:
        input: Input with image URL.

    Returns:
        Image description.
    """
    response = await ai.generate(
        model=_nova_model,
        prompt=[
            Part(root=TextPart(text='Describe this image')),
            Part(root=MediaPart(media=Media(url=input.image_url, content_type='image/jpeg'))),
        ],
    )
    return response.text


@ai.flow()
async def embed_text(input: EmbedInput) -> list[float]:
    """Generate text embeddings using Amazon Titan.

    When using API keys (AWS_BEARER_TOKEN_BEDROCK), this automatically uses
    the inference profile version of the model.

    Args:
        input: Input with text to embed.

    Returns:
        Embedding vector (first 10 dimensions shown).
    """
    embeddings = await ai.embed(
        embedder=_embed_model,
        content=input.text,
    )
    # Return first 10 dimensions as a sample
    embedding = embeddings[0].embedding if embeddings else []
    return embedding[:10] if len(embedding) > 10 else embedding


@ai.flow()
async def reasoning_demo(input: ReasoningInput) -> str:
    """Demonstrate reasoning with DeepSeek R1.

    Note: DeepSeek R1 includes reasoning content in the response.
    For optimal quality, limit max_tokens to 8,192 or fewer.

    When using API keys (AWS_BEARER_TOKEN_BEDROCK), this automatically uses
    the inference profile version of the model (e.g., us.deepseek.r1-v1:0).

    Args:
        input: Input with question requiring reasoning.

    Returns:
        Answer with reasoning steps.
    """
    response = await ai.generate(
        model=_deepseek_model,
        prompt=input.prompt,
        config={
            'max_tokens': 4096,
            'temperature': 0.5,
        },
    )
    return response.text


async def main() -> None:
    """Main entry point for the AWS Bedrock sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
