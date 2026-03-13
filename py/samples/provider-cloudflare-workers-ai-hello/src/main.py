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

"""Cloudflare Workers AI Hello World Sample.

This sample demonstrates how to use Cloudflare Workers AI models with Genkit,
including text generation, streaming, tool calling, and embeddings.

Cloudflare Workers AI runs AI models at the edge, providing low-latency
inference with global availability. This plugin supports:

- Text generation with Llama, Mistral, Qwen, and other models
- Streaming responses via Server-Sent Events (SSE)
- Tool/function calling for supported models
- Text embeddings using BGE and other embedding models

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Workers AI          │ AI models running on Cloudflare's global network.  │
    │                     │ Like having smart robots in data centers worldwide.│
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Edge Computing      │ Processing data close to where users are located.  │
    │                     │ Like having mini-computers in every neighborhood.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Getting the AI response word-by-word as it thinks. │
    │                     │ Like watching someone type their answer in chat.   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Embeddings          │ Converting text into numbers that capture meaning. │
    │                     │ Like translating words into coordinates on a map.  │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    User Request
         │
         ▼
    ┌───────────────────┐
    │  Genkit Flow      │
    │  (generate_greeting) │
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐     ┌───────────────────┐
    │ CF Workers AI     │────▶│ Cloudflare Edge   │
    │ (cf-workers-ai)   │     │ (Global Network)  │
    └───────────────────┘     └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ Llama 3.1 8B      │
                              │ (or other model)  │
                              └─────────┬─────────┘
                                        │
                                        ▼
                              ┌───────────────────┐
                              │ AI Response       │
                              │ (text/stream)     │
                              └───────────────────┘

Testing Instructions:

    1. Set environment variables:
       export CLOUDFLARE_ACCOUNT_ID=your_account_id
       export CLOUDFLARE_API_TOKEN=your_api_token

    2. Run the sample:
       ./run.sh

    3. Open DevUI at http://localhost:4000

    4. Test flows:
        - generate_greeting: Enter a name, get a greeting
        - generate_with_system_prompt: System prompt persona demo
        - generate_multi_turn_chat: Multi-turn conversation demo
        - streaming_demo: Watch tokens stream in real-time
        - generate_weather: See weather tool in action
        - embedding_demo: Generate text embeddings
        - generate_character: Structured output as JSON
"""

import asyncio

from pydantic import BaseModel, Field

from genkit import Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.cloudflare_workers_ai import (
    CloudflareWorkersAI,
    bge_base_en,
    cloudflare_model,
)
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig
from samples.shared import (
    CharacterInput,
    CodeInput,
    ImageDescribeInput,
    MultiTurnInput,
    RpgCharacter,
    StreamingToolInput,
    SystemPromptInput,
    WeatherInput,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_with_tools_logic,
    generate_weather_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
)

setup_sample()


# Enable OTLP telemetry export (optional, requires CF_OTLP_ENDPOINT env var)
# To enable, add 'from genkit.plugins.cloudflare_workers_ai import add_cloudflare_telemetry' and call:
# add_cloudflare_telemetry()

logger = get_logger(__name__)

ai = Genkit(
    plugins=[CloudflareWorkersAI()],
    model=cloudflare_model('@cf/meta/llama-3.1-8b-instruct'),
)

ai.tool()(get_weather)


class HelloInput(BaseModel):
    """Input for the generate_greeting flow.

    Attributes:
        name: Name of the person to greet.
    """

    name: str = Field(
        default='World',
        description='Name of the person to greet',
    )


@ai.flow()
async def generate_greeting(input: HelloInput) -> str:
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


class StreamingInput(BaseModel):
    """Input for the streaming demo flow.

    Attributes:
        prompt: The prompt to send to the model.
    """

    prompt: str = Field(
        default='Tell me a short joke about programming.',
        description='Prompt for streaming generation',
    )


@ai.flow()
async def streaming_demo(input: StreamingInput) -> str:
    """Demonstrate streaming text generation.

    This flow shows how to use streaming with Cloudflare Workers AI,
    where tokens are received one at a time as the model generates them.

    Args:
        input: StreamingInput with the prompt.

    Returns:
        The complete generated text.
    """
    result_text = ''
    stream, _ = ai.generate_stream(prompt=input.prompt)
    async for chunk in stream:
        result_text += chunk.text
    return result_text


class ToolDemoInput(BaseModel):
    """Input for the tool demo flow.

    Attributes:
        location: Location to check weather for.
    """

    location: str = Field(
        default='San Francisco',
        description='Location to check weather for',
    )


@ai.flow()
async def generate_weather(input: WeatherInput) -> str:
    """Get weather information using tool calling.

    Args:
        input: Input with location to get weather for.

    Returns:
        Weather information.
    """
    return await generate_weather_logic(ai, WeatherInput(location=input.location))


class EmbeddingInput(BaseModel):
    """Input for the embedding demo flow.

    Attributes:
        text: Text to generate embedding for.
    """

    text: str = Field(
        default='Hello, world! This is a test of text embeddings.',
        description='Text to embed',
    )


@ai.flow()
async def embedding_demo(input: EmbeddingInput) -> dict[str, object]:
    """Demonstrate text embedding generation.

    This flow shows how to generate vector embeddings for text using
    Cloudflare's BGE embedding models.

    Args:
        input: EmbeddingInput with the text to embed.

    Returns:
        Dictionary with embedding dimensions and first few values.
    """
    result = await ai.embed(
        embedder=bge_base_en,
        content=input.text,
    )

    if result:
        embedding = result[0].embedding
        return {
            'text': input.text,
            'dimensions': len(embedding),
            'first_5_values': embedding[:5],
            'model': '@cf/baai/bge-base-en-v1.5',
        }

    return {'error': 'No embeddings generated'}


class ModelComparisonInput(BaseModel):
    """Input for comparing different models.

    Attributes:
        prompt: The prompt to test with different models.
    """

    prompt: str = Field(
        default='Explain quantum computing in one sentence.',
        description='Prompt for model comparison',
    )


@ai.flow()
async def model_comparison(input: ModelComparisonInput) -> dict[str, str]:
    """Compare responses from different Cloudflare AI models.

    This flow demonstrates using multiple models and comparing their outputs.

    Args:
        input: ModelComparisonInput with the prompt.

    Returns:
        Dictionary mapping model names to their responses.
    """
    models = [
        '@cf/meta/llama-3.1-8b-instruct',
        '@hf/mistral/mistral-7b-instruct-v0.2',
    ]

    results: dict[str, str] = {}

    for model_id in models:
        try:
            response = await ai.generate(
                prompt=input.prompt,
                model=cloudflare_model(model_id),
            )
            results[model_id] = response.text
        except Exception as e:
            results[model_id] = f'Error: {e!s}'

    return results


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using a multimodal Cloudflare AI model."""
    return await describe_image_logic(
        ai, input.image_url, model=cloudflare_model('@cf/meta/llama-4-scout-17b-16e-instruct')
    )


class ConfigDemoInput(BaseModel):
    """Input for config demo flow.

    Attributes:
        prompt: The prompt to generate with.
    """

    prompt: str = Field(
        default='Write a haiku about programming.',
        description='Prompt for generation with custom config',
    )


@ai.flow()
async def generate_with_config(input: ConfigDemoInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with prompt for generation.

    Returns:
        Generated text.
    """
    response = await ai.generate(
        prompt=input.prompt,
        config=CloudflareConfig(
            temperature=0.9,  # Higher for more creativity
            top_k=40,  # Limit token selection
            repetition_penalty=1.2,  # Discourage repetition
            max_output_tokens=256,
        ),
    )
    return response.text


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using Cloudflare Workers AI models.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


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


async def main() -> None:
    """Main entry point for the sample application."""
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
