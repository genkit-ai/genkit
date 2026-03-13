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

"""Anthropic hello sample - Claude models with Genkit.

This sample demonstrates how to use Anthropic's Claude models with Genkit,
including tools, streaming, thinking mode, and multimodal capabilities.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Claude              │ Anthropic's AI assistant. Like a helpful friend   │
    │                     │ who's great at explaining things and writing.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Haiku               │ Claude's fast model. Quick responses for simple   │
    │                     │ tasks. Great for chatbots and quick Q&A.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Sonnet              │ Claude's balanced model. Good at most tasks       │
    │                     │ without being too slow or expensive.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Let Claude use functions you define. "Get the     │
    │                     │ weather" → Claude calls your weather function.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Thinking Mode       │ Claude shows its reasoning step by step.          │
    │                     │ Like showing your work on a math test.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Get Claude's response word-by-word.               │
    │                     │ Feels faster, like watching someone type.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Multimodal          │ Claude can see images! Send a photo and ask       │
    │                     │ "What's in this picture?"                         │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[Anthropic()])` |
| Default Model Configuration             | `ai = Genkit(model=...)`            |
| Defining Flows                          | `@ai.flow()` decorator              |
| Defining Tools                          | `@ai.tool()` decorator              |
| Pydantic for Tool Input Schema          | `WeatherInput`, `CurrencyInput`     |
| Simple Generation (Prompt String)       | `generate_greeting`                            |
| System Prompts                          | `generate_with_system_prompt`                     |
| Multi-turn Conversations (`messages`)   | `generate_multi_turn_chat`                   |
| Streaming Generation                    | `generate_streaming_story`                     |
| Generation with Tools                   | `generate_weather`, `convert_currency`  |
| Generation Configuration (temperature)  | `generate_with_config`                |
| Thinking (CoT)                          | `thinking_demo`                     |
| Code Generation                         | `generate_code`                         |
| Multimodal (Image Input)                | `describe_image`                    |
| Prompt Caching                          | `cached_generation`                 |
| PDF Document Input                      | `analyze_pdf`                       |
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.anthropic import Anthropic, anthropic_name
from genkit.types import Media, MediaPart, Message, Metadata, Part, Role, TextPart
from samples.shared import (
    CharacterInput,
    CodeInput,
    CurrencyExchangeInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
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

if 'ANTHROPIC_API_KEY' not in os.environ:
    os.environ['ANTHROPIC_API_KEY'] = input('Please enter your ANTHROPIC_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[Anthropic()],
    model=anthropic_name('claude-3-5-haiku'),
)

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


class ThinkingInput(BaseModel):
    """Input for thinking demo."""

    question: str = Field(default='Why do cats purr?', description='Question to answer')


class CacheInput(BaseModel):
    """Input for prompt caching demo."""

    question: str = Field(default='What are the key themes?', description='Question about the cached text')


class PdfInput(BaseModel):
    """Input for PDF analysis demo."""

    pdf_url: str = Field(
        # Public domain sample PDF.
        default='https://pdfobject.com/pdf/sample.pdf',
        description='URL of the PDF to analyze',
    )
    question: str = Field(default='Describe the contents of this document.', description='Question about the PDF')


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
async def generate_code(input: CodeInput) -> str:
    """Generate code using Claude.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task, model=anthropic_name('claude-opus-4-6'))


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
async def generate_character(input: CharacterInput) -> RpgCharacter:
    """Generate an RPG character with structured output.

    Args:
        input: Input with character name.

    Returns:
        The generated RPG character.
    """
    return await generate_character_logic(ai, input.name)


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
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Anthropic.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url)


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
async def thinking_demo(input: ThinkingInput) -> str:
    """Demonstrate Anthropic thinking capability.

    Note: 'thinking' requires a compatible model (e.g., Claude 3.7 Sonnet).
    """
    response = await ai.generate(
        model=anthropic_name('claude-3-7-sonnet-20250219'),
        prompt=input.question,
        config={
            'thinking': {'type': 'enabled', 'budget_tokens': 1024},
            'max_output_tokens': 4096,  # Required when thinking is enabled
        },
    )
    return response.text


@ai.flow()
async def cached_generation(input: CacheInput) -> str:
    """Demonstrate Anthropic prompt caching.

    Prompt caching lets Anthropic cache large context blocks across requests,
    reducing latency and cost for repeated prompts with shared prefixes.

    Set ``cache_control`` metadata on any content part to mark it as a
    cache breakpoint. Anthropic will cache everything up to and including
    that part.

    See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
    """
    # The long context is marked with cache_control metadata.
    # On subsequent calls with the same prefix, Anthropic reuses the cache.
    long_context = (
        'The Genkit framework provides a unified interface for working with '
        'generative AI models across multiple providers. It supports flows, '
        'tools, structured output, streaming, and advanced features like '
        'prompt caching and document input. ' * 10
    )
    response = await ai.generate(
        model=anthropic_name('claude-3-5-haiku'),
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(
                        root=TextPart(
                            text=long_context,
                            metadata=Metadata({'cache_control': {'type': 'ephemeral'}}),
                        )
                    ),
                    Part(root=TextPart(text=input.question)),
                ],
            ),
        ],
    )
    return response.text


@ai.flow()
async def analyze_pdf(input: PdfInput) -> str:
    """Analyze a PDF document using Anthropic's document input.

    Anthropic supports sending PDF files directly to Claude for analysis.
    PDFs can be provided as URLs or base64-encoded data URIs.

    See: https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
    """
    response = await ai.generate(
        model=anthropic_name('claude-3-5-haiku'),
        prompt=[
            Part(root=TextPart(text=input.question)),
            Part(root=MediaPart(media=Media(url=input.pdf_url, content_type='application/pdf'))),
        ],
    )
    return response.text


async def main() -> None:
    """Main entry point for the Anthropic sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
