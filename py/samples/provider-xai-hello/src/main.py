# Copyright 2025 Google LLC
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

"""xAI Genkit sample - Grok models with Genkit.

This sample demonstrates how to use xAI's Grok models with Genkit,
including basic generation, streaming, and tool calling.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ xAI                 │ Elon Musk's AI company. Makes the Grok models.     │
    │                     │                                                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Grok                │ xAI's AI assistant. Known for being witty and      │
    │                     │ having access to real-time X/Twitter data.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Grok-3              │ The main Grok model. Good balance of speed         │
    │                     │ and capability for most tasks.                     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Let Grok use functions you define. Like giving     │
    │                     │ it a calculator or weather lookup to use.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Get the response word-by-word as it's generated.   │
    │                     │ Feels faster, like watching someone type.          │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[XAI()])`      |
| Model Configuration                     | `xai_name('grok-3')`                |
| Simple Text Generation                  | `generate_greeting`                 |
| System Prompts                          | `generate_with_system_prompt`       |
| Multi-turn Conversations (`messages`)   | `generate_multi_turn_chat`           |
| Streaming Generation                    | `generate_streaming_story`           |
| Tool Usage (Decorated)                  | `get_weather`, `calculate`          |
| Generation Configuration                | `generate_with_config`              |
| Multimodal (Image Input / Vision)       | `describe_image`                    |
| Reasoning (Chain-of-Thought)            | `solve_reasoning_problem`           |
| Code Generation                         | `generate_code`                         |
| Tool Calling                            | `generate_weather`                   |
"""

import asyncio
import os

from genkit.ai import Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.firebase import add_firebase_telemetry
from genkit.plugins.xai import XAI, xai_name
from samples.shared import (
    CalculatorInput,
    CharacterInput,
    CodeInput,
    ConfigInput,
    CurrencyExchangeInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    WeatherInput,
    calculate,
    calculation_logic,
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
    solve_reasoning_problem_logic,
)

setup_sample()

if 'XAI_API_KEY' not in os.environ:
    os.environ['XAI_API_KEY'] = input('Please enter your XAI_API_KEY: ')

logger = get_logger(__name__)

add_firebase_telemetry(force_dev_export=True, log_input_and_output=True)

ai = Genkit(
    plugins=[XAI()],
    model=xai_name('grok-3'),
)


# Decorated tools
ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)
ai.tool()(calculate)


@ai.flow()
async def convert_currency(input_data: CurrencyExchangeInput) -> str:
    """Convert currency using tool calling.

    Args:
        input_data: Currency exchange parameters.

    Returns:
        Conversion result.
    """
    return await convert_currency_logic(ai, input_data)


@ai.flow()
async def calculate(input_data: CalculatorInput) -> str:
    """Perform a calculation using tool calling.

    Args:
        input_data: Calculator parameters.

    Returns:
        Calculation result.
    """
    return await calculation_logic(ai, input_data)


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
    ctx: ActionRunContext | None = None,
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
async def generate_with_config(input: ConfigInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with name to greet.

    Returns:
        Greeting message.
    """
    return await generate_with_config_logic(ai, input.name)


@ai.flow()
async def generate_weather(input_data: WeatherInput) -> str:
    """Get weather information using tool calling.

    Args:
        input_data: Input with location to get weather for.

    Returns:
        Weather information.
    """
    return await generate_weather_logic(ai, input_data)


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using Grok.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Grok 2 Vision.

    Uses grok-2-vision-1212 which supports media=True for image understanding.
    The xAI gRPC SDK handles image URLs in MediaPart messages.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url, model=xai_name('grok-2-vision-1212'))


@ai.flow()
async def solve_reasoning_problem(input: ReasoningInput) -> str:
    """Solve reasoning problems using Grok 4.

    Grok 4 is a reasoning model that provides chain-of-thought responses.
    It is registered with REASONING_MODEL_SUPPORTS in the xAI plugin.

    Args:
        input: Input with reasoning question to solve.

    Returns:
        The reasoning and answer.
    """
    return await solve_reasoning_problem_logic(ai, input.prompt, model=xai_name('grok-4'))


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
    """Main entry point - keep alive for Dev UI."""
    logger.info('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
