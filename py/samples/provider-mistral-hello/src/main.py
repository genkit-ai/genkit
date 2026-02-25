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

"""Mistral AI hello sample - Mistral models with Genkit.

This sample demonstrates how to use Mistral AI's models with Genkit,
including Mistral Large 3, Mistral Small 3.2, Codestral, Magistral

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral AI          │ French AI company known for efficient, powerful    │
    │                     │ models. Great balance of speed and quality.        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ mistral-large       │ Most capable model. Best for complex reasoning,    │
    │                     │ coding, and nuanced tasks.                         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ mistral-small       │ Fast and efficient. Great for everyday tasks       │
    │                     │ like chat, summarization, and simple coding.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ codestral           │ Specialized coding model. Optimized for code       │
    │                     │ generation, completion, and explanation.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Let the model call your functions. Like giving     │
    │                     │ the AI a toolbox to help answer questions.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Structured Output   │ Get responses in a specific format (JSON).         │
    │                     │ Like filling out a form instead of free text.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Get the response word-by-word as it's generated.   │
    │                     │ Feels faster, like watching someone type.          │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet         |
|-----------------------------------------|-----------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[Mistral(...)])`   |
| Default Model Configuration             | `ai = Genkit(model=mistral_name(...))` |
| Defining Flows                          | `@ai.flow()` decorator                  |
| Defining Tools                          | `@ai.tool()` decorator                  |
| Simple Generation (Prompt String)       | `generate_greeting`                      |
| System Prompt                           | `generate_with_system_prompt`            |
| Multi-turn Conversation                 | `generate_multi_turn_chat`               |
| Streaming Response                      | `generate_streaming_story`               |
| Code Generation (Codestral)             | `generate_code`                          |
| Generation with Config                  | `generate_with_config`                   |
| Multi-turn Chat                         | `chat_flow`                              |
| Tool Calling                            | `generate_weather`                       |
| Structured Output (JSON)                | `generate_character`                     |
| Streaming Structured Output             | `streaming_structured_output`            |
| Multimodal (Image Input)                | `describe_image`                         |
| Reasoning (Magistral)                   | `solve_reasoning_problem`                |
| Embeddings (Text)                       | `embed_flow`                            |
| Embeddings (Code)                       | `code_embed_flow`                       |
| Audio Transcription (Voxtral)           | `audio_flow`                            |
"""

import asyncio
import base64
import os
from pathlib import Path

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output
from genkit.blocks.document import Document
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.core.typing import Media, MediaPart, Message, Part, Role, TextPart
from genkit.plugins.mistral import Mistral, mistral_name
from samples.shared import (
    CharacterInput,
    CodeInput,
    ConfigInput,
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
    chat_flow_logic,
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

if 'MISTRAL_API_KEY' not in os.environ:
    os.environ['MISTRAL_API_KEY'] = input('Please enter your MISTRAL_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[Mistral()],
    model=mistral_name('mistral-small-latest'),
)

ai.tool()(get_weather)


class CodeEmbedInput(BaseModel):
    """Input for code embedding flow."""

    code: str = Field(
        default='def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
        description='Code to embed',
    )


class AudioInput(BaseModel):
    """Input for audio transcription flow."""

    audio_path: str = Field(
        default='',
        description='Path to audio file (defaults to bundled genkit.wav)',
    )


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
async def streaming_structured_output(
    input: CharacterInput,
    ctx: ActionRunContext | None = None,
) -> RpgCharacter:
    """Streaming with structured output schema.

    Combines `generate_stream` with `Output(schema=...)` so the model
    streams JSON tokens that are progressively parsed into the Pydantic
    model. Each chunk exposes a partial `.output` you can forward to
    clients for incremental rendering.

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
async def generate_code(input: CodeInput) -> str:
    """Generate code using Codestral model.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


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
async def chat_flow() -> str:
    """Multi-turn chat demonstrating context retention across 3 turns.

    Returns:
        Final chat response.
    """
    return await chat_flow_logic(
        ai,
        system_prompt='You are a helpful travel assistant specializing in French destinations.',
        prompt1=(
            "Hi! I'm planning a trip to Paris next month. I'm really excited because I love French cuisine, "
            'especially croissants and macarons.'
        ),
        followup_question='What foods did I say I enjoy?',
        final_question='Based on our conversation, suggest one bakery I should visit.',
    )


@ai.flow()
async def large_model_flow() -> str:
    """Use Mistral Large for complex reasoning tasks.

    Returns:
        Response from Mistral Large model.
    """
    response = await ai.generate(
        model=mistral_name('mistral-large-latest'),
        prompt=(
            'Analyze the pros and cons of microservices vs monolithic architecture. '
            'Consider scalability, maintainability, and team organization.'
        ),
        system='You are a senior software architect with 20 years of experience.',
    )
    return response.text


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
async def embed_flow(input: EmbedInput) -> list[float]:
    """Generate embeddings for text using Mistral's mistral-embed model.

    Args:
        input: Input with text to embed.

    Returns:
        The embedding vector (list of floats).
    """
    doc = Document.from_text(input.text)
    embeddings = await ai.embed(
        embedder=mistral_name('mistral-embed'),
        content=doc,
    )
    return embeddings[0].embedding


@ai.flow()
async def code_embed_flow(input: CodeEmbedInput) -> list[float]:
    """Generate code embeddings using Mistral's codestral-embed model.

    Args:
        input: Input with code snippet to embed.

    Returns:
        The embedding vector (list of floats).
    """
    doc = Document.from_text(input.code)
    embeddings = await ai.embed(
        embedder=mistral_name('codestral-embed-2505'),
        content=doc,
    )
    return embeddings[0].embedding


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Mistral Large 3 (vision).

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url, model=mistral_name('mistral-large-latest'))


@ai.flow()
async def solve_reasoning_problem(input: ReasoningInput) -> str:
    """Solve reasoning problems using Magistral.

    Args:
        input: Input with reasoning question to solve.

    Returns:
        The reasoning and answer.
    """
    return await solve_reasoning_problem_logic(ai, input.prompt, model=mistral_name('magistral-small-latest'))


@ai.flow()
async def audio_flow(input: AudioInput) -> str:
    """Transcribe audio using Voxtral Mini.

    Uses the bundled genkit.wav file by default.

    Args:
        input: Input with optional path to an audio file.

    Returns:
        Transcription of the audio content.
    """
    audio_path = input.audio_path or str(Path(__file__).parent.parent / 'assets' / 'genkit.wav')
    audio_bytes = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('ascii')
    data_uri = f'data:audio/wav;base64,{audio_b64}'

    response = await ai.generate(
        model=mistral_name('voxtral-mini-latest'),
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=MediaPart(media=Media(url=data_uri, content_type='audio/wav'))),
                    Part(root=TextPart(text='Transcribe this audio. Return only the transcription.')),
                ],
            ),
        ],
    )
    return response.text


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
    """Main entry point for the Mistral sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
