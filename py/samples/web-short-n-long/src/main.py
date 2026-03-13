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

r"""Long-running server mode sample - ASGI deployment with Genkit.

This sample demonstrates how to deploy Genkit flows as a production-ready
ASGI application using uvicorn, with proper lifecycle management.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ASGI                │ A standard for Python web servers. Like USB        │
    │                     │ but for connecting web frameworks.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ uvicorn             │ A fast ASGI server. Runs your Genkit app and       │
    │                     │ handles HTTP requests efficiently.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Long-running        │ Server that stays up continuously. Not just        │
    │                     │ one request, but serving forever.                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Lifecycle Hooks     │ Functions called when server starts/stops.         │
    │                     │ Setup database, cleanup connections, etc.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Production-ready    │ Properly handles errors, shutdown signals,         │
    │                     │ and concurrent requests.                           │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                                      | Example Function / Code Snippet        |
|----------------------------------------------------------|----------------------------------------|
| Deployment as ASGI App                                   | `create_flows_asgi_app`                |
| Custom Server Lifecycle Hooks                            | `on_app_startup`, `on_app_shutdown`    |
| Running as HTTP Server                                   | `uvicorn.Server`                       |
| Plugin Initialization                                    | `ai = Genkit(plugins=[GoogleAI()])`    |
| Default Model Configuration                              | `ai = Genkit(model=...)`               |
| Defining Flows                                           | `@ai.flow()` decorator (multiple uses) |
| Defining Tools                                           | `@ai.tool()` decorator (multiple uses) |
| Tool Input Schema (Pydantic)                             | `GablorkenInput`                       |
| Simple Generation (Prompt String)                        | `say_hi`                               |
| System Prompt                                            | `system_prompt`                        |
| Multi-turn Conversation                                  | `multi_turn_chat`                      |
| Generation with Messages (`Message`, `Role`, `TextPart`) | `simple_generate_with_tools_flow`      |
| Generation with Tools                                    | `simple_generate_with_tools_flow`      |
| Tool Response Handling                                   | `simple_generate_with_interrupts`      |
| Tool Interruption (`ctx.interrupt`)                      | `gablorken_tool2`                      |
| Embedding (`ai.embed`, `Document`)                       | `embed_docs`                           |
| Generation Configuration (`temperature`, etc.)           | `say_hi_with_configured_temperature`   |
| Streaming Generation (`ai.generate_stream`)              | `say_hi_stream`                        |
| Streaming Chunk Handling (`ctx.send_chunk`)              | `say_hi_stream`, `generate_character`  |
| Structured Output (Schema)                               | `generate_character`                   |
| Streaming Structured Output                              | `streaming_structured_output`          |
| Pydantic for Structured Output Schema                    | `RpgCharacter`                         |
| Structured Output (Instruction-Based)                    | `generate_character_instructions`      |
| Multi-modal Output Configuration                         | `generate_images`                      |

See README.md for testing instructions.
"""

import argparse
import asyncio
import os

import uvicorn
from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output, ToolRunContext, tool_response
from genkit.blocks.model import GenerateResponseWrapper
from genkit.core.action import ActionRunContext
from genkit.core.flows import create_flows_asgi_app
from genkit.core.logging import get_logger
from genkit.core.typing import Part
from genkit.plugins.google_genai import (
    EmbeddingTaskType,
    GeminiConfigSchema,
    GeminiEmbeddingModels,
    GoogleAI,
)
from genkit.plugins.google_genai.models import gemini
from genkit.types import (
    Embedding,
    GenerationCommonConfig,
    Message,
    Role,
    TextPart,
)
from samples.shared.logging import setup_sample

setup_sample()

logger = get_logger(__name__)

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-pro-preview',
)


class GablorkenInput(BaseModel):
    """The Pydantic model for tools."""

    value: int = Field(description='value to calculate gablorken for')


class ToolsFlowInput(BaseModel):
    """Input for tools flow."""

    value: int = Field(default=42, description='Value for gablorken calculation')


class SayHiInput(BaseModel):
    """Input for say_hi flow."""

    name: str = Field(default='Mittens', description='Name to greet')


class SystemPromptInput(BaseModel):
    """Input for system_prompt flow."""

    question: str = Field(default='What is your quest?', description='Question to ask')


class MultiTurnInput(BaseModel):
    """Input for multi_turn_chat flow."""

    destination: str = Field(default='Japan', description='Travel destination')


class TemperatureInput(BaseModel):
    """Input for temperature config flow."""

    data: str = Field(default='Mittens', description='Name to greet')


class StreamInput(BaseModel):
    """Input for streaming flow."""

    name: str = Field(default='Shadow', description='Name for streaming greeting')


class StreamGreetingInput(BaseModel):
    """Input for stream greeting flow."""

    name: str = Field(default='Whiskers', description='Name for greeting')


class EmbedDocsInput(BaseModel):
    """Input for embed_docs flow."""

    docs: list[str] = Field(
        default=['Hello world', 'Genkit is great', 'Embeddings are fun'],
        description='List of texts to embed',
    )


class CharacterInput(BaseModel):
    """Input for character generation."""

    name: str = Field(default='Whiskers', description='Character name')


class GenerateImagesInput(BaseModel):
    """Input for image generation flow."""

    name: str = Field(default='a fluffy cat', description='Subject to generate images about')


@ai.tool(name='gablorkenTool')
def gablorken_tool(input_: GablorkenInput) -> int:
    """Calculate a gablorken.

    Args:
        input_: The input to calculate gablorken for.

    Returns:
        The calculated gablorken.
    """
    return input_.value * 3 - 5


@ai.flow()
async def simple_generate_with_tools_flow(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response = await ai.generate(
        model=f'googleai/{gemini.GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text=f'what is a gablorken of {input.value}'))],
            ),
        ],
        tools=['gablorkenTool'],
    )
    return response.text


@ai.tool(name='interruptingTool')
def interrupting_tool(input_: GablorkenInput, ctx: ToolRunContext) -> None:
    """The user-defined tool function.

    Args:
        input_: the input to the tool
        ctx: the tool run context

    Returns:
        The calculated gablorken.
    """
    ctx.interrupt()


@ai.flow()
async def simple_generate_with_interrupts(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response1 = await ai.generate(
        model=f'googleai/{gemini.GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text=f'what is a gablorken of {input.value}'))],
            ),
        ],
        tools=['interruptingTool'],
    )
    await logger.ainfo(f'len(response.tool_requests)={len(response1.tool_requests)}')
    if len(response1.interrupts) == 0:
        return response1.text

    tr = tool_response(response1.interrupts[0], 178)
    response = await ai.generate(
        model=f'googleai/{gemini.GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
        messages=response1.messages,
        tool_responses=[tr],
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def say_hi(input: SayHiInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with name to greet.

    Returns:
        The generated response with a function.
    """
    resp = await ai.generate(
        prompt=f'hi {input.name}',
    )
    return resp.text


@ai.flow()
async def system_prompt(input: SystemPromptInput) -> str:
    """Demonstrate system prompts to control model persona and behavior.

    System prompts give the model instructions about how to respond, such as
    adopting a specific persona, tone, or response format.

    See: https://genkit.dev/docs/models#system-prompts

    Args:
        input: Input with a question to ask.

    Returns:
        The model's response in the persona defined by the system prompt.
    """
    response = await ai.generate(
        prompt=input.question,
        system='You are a pirate captain from the 18th century. Always respond in character, '
        'using pirate slang and nautical terminology.',
    )
    return response.text


@ai.flow()
async def multi_turn_chat(input: MultiTurnInput) -> str:
    """Demonstrate multi-turn conversations using the messages parameter.

    The messages parameter allows you to pass a conversation history to
    maintain context across multiple interactions with the model. Each
    message has a role ('user' or 'model') and content.

    See: https://genkit.dev/docs/models#multi-turn-conversations-with-messages

    Args:
        input: Input with a travel destination.

    Returns:
        The model's final response, demonstrating context retention.
    """
    # Turn 1: Start the conversation
    response1 = await ai.generate(
        system='You are a helpful travel assistant.',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text=f'I want to visit {input.destination} for two weeks in spring.'))],
            ),
        ],
    )

    # Turn 2: Follow-up question that requires context from turn 1
    response2 = await ai.generate(
        system='You are a helpful travel assistant.',
        messages=[
            *response1.messages,
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='What should I pack for that trip?'))],
            ),
        ],
    )
    return response2.text


@ai.flow()
async def embed_docs(input: EmbedDocsInput) -> list[Embedding]:
    """Generate an embedding for the words in a list.

    Args:
        input: list of texts (string)

    Returns:
        The generated embedding.
    """
    options = {'task_type': EmbeddingTaskType.CLUSTERING}
    return await ai.embed_many(
        embedder=f'googleai/{GeminiEmbeddingModels.GEMINI_EMBEDDING_001}',
        content=input.docs,
        options=options,
    )


@ai.flow()
async def say_hi_with_configured_temperature(input: TemperatureInput) -> GenerateResponseWrapper:
    """Generate a greeting for the given name.

    Args:
        input: Input with name to greet.

    Returns:
        The generated response with a function.
    """
    return await ai.generate(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=f'hi {input.data}'))])],
        config=GenerationCommonConfig(temperature=0.1),
    )


@ai.flow()
async def say_hi_stream(
    input: StreamInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with name for streaming.
        ctx: the context of the tool

    Returns:
        The generated response with a function.
    """
    stream, _ = ai.generate_stream(prompt=f'hi {input.name}')
    result: str = ''
    async for data in stream:
        if ctx is not None:
            ctx.send_chunk(data.text)
        result += data.text

    return result


@ai.flow()
async def stream_greeting(
    input: StreamGreetingInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Stream a greeting for the given name.

    Args:
        input: Input with name for greeting.
        ctx: the context of the tool

    Returns:
        The generated response with a function.
    """
    chunks = [
        'hello',
        input.name,
        'how are you?',
    ]
    for data in chunks:
        await asyncio.sleep(1)
        if ctx is not None:
            ctx.send_chunk(data)

    return 'test streaming response'


class Skills(BaseModel):
    """Skills for an RPG character."""

    strength: int = Field(description='strength (0-100)')
    charisma: int = Field(description='charisma (0-100)')
    endurance: int = Field(description='endurance (0-100)')


class RpgCharacter(BaseModel):
    """An RPG character."""

    name: str = Field(description='name of the character')
    back_story: str = Field(description='back story', alias='backStory')
    abilities: list[str] = Field(description='list of abilities (3-4)')
    skills: Skills


@ai.flow()
async def generate_character(
    input: CharacterInput,
    ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Generate an RPG character.

    Args:
        input: Input with character name.
        ctx: the context of the tool

    Returns:
        The generated RPG character.
    """
    if ctx is not None and ctx.is_streaming:
        stream, result = ai.generate_stream(
            prompt=f'generate an RPG character named {input.name}',
            output=Output(schema=RpgCharacter),
        )
        async for data in stream:
            ctx.send_chunk(data.output)

        return (await result).output
    else:
        result = await ai.generate(
            prompt=f'generate an RPG character named {input.name}',
            output=Output(schema=RpgCharacter),
        )
        return result.output


@ai.flow()
async def generate_character_instructions(
    input: CharacterInput,
    _ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Generate an RPG character using instruction-based structured output.

    Unlike ``generate_character`` which uses constrained decoding (the model
    is forced to output valid JSON matching the schema), this flow uses
    ``output_constrained=False`` to guide the model via prompt instructions
    instead. This is useful when::

        - The model doesn't support constrained decoding.
        - You want the model to have more flexibility in its output.
        - You're debugging schema adherence issues.

    See: https://genkit.dev/docs/models#structured-output

    Args:
        input: Input with character name.
        _ctx: the context of the tool (unused)

    Returns:
        The generated RPG character.
    """
    result = await ai.generate(
        prompt=f'generate an RPG character named {input.name}',
        output=Output(schema=RpgCharacter),
        output_constrained=False,
        output_instructions=True,
    )
    return result.output


@ai.flow()
async def streaming_structured_output(
    input: CharacterInput,
    ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Demonstrate streaming with structured output schemas.

    Combines `generate_stream` with `Output(schema=...)` so the model
    streams JSON tokens that are progressively parsed into the Pydantic
    model. Each chunk exposes a partial `.output` you can forward to
    clients for incremental rendering.

    See: https://genkit.dev/docs/models#streaming

    Args:
        input: Input with character name.
        ctx: Action context for streaming partial outputs.

    Returns:
        The fully-parsed RPG character once streaming completes.
    """
    stream, result = ai.generate_stream(
        prompt=(
            f'Generate an RPG character named {input.name}. '
            'Include a creative backstory, 3-4 unique abilities, '
            'and skill ratings for strength, charisma, and endurance (0-100 each).'
        ),
        output=Output(schema=RpgCharacter),
    )
    async for chunk in stream:
        if ctx is not None:
            ctx.send_chunk(chunk.output)

    return (await result).output


@ai.flow()
async def generate_images(
    input: GenerateImagesInput,
    ctx: ActionRunContext | None = None,
) -> GenerateResponseWrapper:
    """Generate images for the given name.

    Args:
        input: Input with subject for image generation.
        ctx: the context of the tool

    Returns:
        The generated response with a function.
    """
    return await ai.generate(
        model='googleai/gemini-3-pro-image-preview',
        prompt=f'Tell me about {input.name} with photos.',
        config=GeminiConfigSchema.model_validate({
            'response_modalities': ['text', 'image'],
        }).model_dump(),
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        The parsed command line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true', help='Run the application as a server')
    return parser.parse_args()


async def server_main(ai: Genkit) -> None:
    """Entry point function for the server application."""

    async def on_app_startup() -> None:
        """Handle application startup."""
        await logger.ainfo('[LIFESPAN] Starting flows server...')
        # Any initialization could go here

    async def on_app_shutdown() -> None:
        """Handle application shutdown."""
        await logger.ainfo('[LIFESPAN] Shutting down flows server...')

    app = create_flows_asgi_app(
        registry=ai.registry,
        context_providers=[],
        on_app_startup=on_app_startup,
        on_app_shutdown=on_app_shutdown,
    )
    # pyrefly: ignore[bad-argument-type] - app type is compatible with uvicorn
    config = uvicorn.Config(app, host='localhost', port=3400)
    server = uvicorn.Server(config)
    await server.serve()


async def main(ai: Genkit) -> None:
    """Main function."""
    await logger.ainfo(await say_hi(SayHiInput(name='tell me a joke')))


if __name__ == '__main__':
    config: argparse.Namespace = parse_args()
    runner = server_main if config.server else main
    ai.run_main(runner(ai))
