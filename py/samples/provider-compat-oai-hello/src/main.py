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

"""OpenAI hello sample - GPT models with Genkit.

This sample demonstrates how to use OpenAI's GPT models with Genkit
using the OpenAI-compatible plugin, including multimodal capabilities
like image generation (DALL-E), text-to-speech (TTS), and
speech-to-text (STT / Whisper).

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OpenAI              │ The company that made ChatGPT. This sample         │
    │                     │ talks to their API directly.                       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ GPT-4o              │ OpenAI's multimodal model. Can see images,         │
    │                     │ hear audio, and chat - all in one model.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ OpenAI-compatible   │ Many AI providers copy OpenAI's API format.        │
    │                     │ This plugin works with ALL of them!                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ DALL-E / TTS / STT  │ OpenAI also generates images, speaks text          │
    │                     │ aloud, and transcribes audio. All supported!       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Let GPT use functions you define. Like giving      │
    │                     │ it a calculator or search engine to use.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Get the response word-by-word as it's generated.   │
    │                     │ Feels faster, like watching someone type.          │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                                      | Example Function / Code Snippet        |
|----------------------------------------------------------|----------------------------------------|
| Plugin Initialization                                    | `ai = Genkit(plugins=[OpenAI()])`      |
| Default Model Configuration                              | `ai = Genkit(model=...)`               |
| Defining Flows                                           | `@ai.flow()` decorator (multiple uses) |
| Defining Tools                                           | `@ai.tool()` decorator (multiple uses) |
| Tool Input Schema (Pydantic)                             | `GablorkenInput`                       |
| Simple Generation (Prompt String)                        | `generate_greeting`                    |
| System Prompts                                           | `generate_with_system_prompt`          |
| Multi-turn Conversations (`messages`)                    | `generate_multi_turn_chat`             |
| Structured Output (Simple)                               | `structured_menu_suggestion`           |
| Streaming Generation                                     | `generate_streaming_story`             |
| Generation with Tools                                    | `calculate_gablorken`                  |
| Tool Response Handling with context                      | `generate_character`                   |
| Multimodal (Image Input / Vision)                        | `describe_image`                       |
| Reasoning (Chain-of-Thought)                             | `solve_reasoning_problem`              |
| Image Generation (DALL-E)                                | `generate_image`                       |
| Text-to-Speech (TTS)                                     | `text_to_speech`                       |
| Code Generation                                          | `generate_code`                            |
| TTS → STT Round-Trip                                     | `round_trip_tts_stt`                   |

See README.md for testing instructions.
"""

import asyncio
import os

import httpx
from pydantic import BaseModel, Field

from genkit.ai import ActionRunContext, Genkit, Output
from genkit.core.logging import get_logger
from genkit.plugins.compat_oai import OpenAI, openai_model
from genkit.types import Media, MediaPart, Message, Part, Role, TextPart
from samples.shared import (
    CharacterInput,
    CodeInput,
    CurrencyExchangeInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    convert_currency as _convert_currency_tool,
    convert_currency_logic,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
    solve_reasoning_problem_logic,
)

setup_sample()

if 'OPENAI_API_KEY' not in os.environ:
    os.environ['OPENAI_API_KEY'] = input('Please enter your OPENAI_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(plugins=[OpenAI()], model=openai_model('gpt-4o'))

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


class GablorkenInput(BaseModel):
    """The Pydantic model for tools."""

    value: int = Field(description='value to calculate gablorken for')


class MenuSuggestion(BaseModel):
    """A suggested menu item from a themed restaurant.

    Demonstrates structured output with multiple field types: strings,
    numbers, lists, and booleans — matching the Genkit documentation
    example for structured output.
    """

    name: str = Field(description='The name of the menu item')
    description: str = Field(description='A short, appetizing description')
    price: float = Field(description='Estimated price in USD')
    allergens: list[str] = Field(description='Known allergens (e.g., nuts, dairy, gluten)')
    is_vegetarian: bool = Field(description='Whether the item is vegetarian')


class MyInput(BaseModel):
    """My input."""

    a: int = Field(default=5, description='a field')
    b: int = Field(default=3, description='b field')


class WeatherRequest(BaseModel):
    """Weather request."""

    latitude: float
    longitude: float


class Temperature(BaseModel):
    """Temperature by location."""

    location: str
    temperature: float
    gablorken: float | str | None = None


class WeatherResponse(BaseModel):
    """Weather response."""

    answer: list[Temperature]


class GablorkenFlowInput(BaseModel):
    """Input for gablorken calculation flow."""

    value: int = Field(default=42, description='Value to calculate gablorken for')


class MenuSuggestionInput(BaseModel):
    """Input for structured menu suggestion flow."""

    theme: str = Field(default='pirate', description='Restaurant theme (e.g., pirate, space, medieval)')


class WeatherFlowInput(BaseModel):
    """Input for weather flow."""

    location: str = Field(default='New York', description='Location to get weather for')


class ImagePromptInput(BaseModel):
    """Input for image generation flow."""

    prompt: str = Field(
        default='A watercolor painting of a cat sitting on a windowsill at sunset',
        description='Text prompt describing the image to generate',
    )


class TTSInput(BaseModel):
    """Input for text-to-speech flow."""

    text: str = Field(
        default='Hello! This is Genkit speaking through OpenAI text-to-speech.',
        description='Text to convert to speech',
    )
    voice: str = Field(
        default='alloy',
        description='Voice to use (alloy, echo, fable, onyx, nova, shimmer)',
    )


class RoundTripInput(BaseModel):
    """Input for the TTS → STT round-trip demo.

    Provide text to convert to speech, then transcribe back. This
    demonstrates both TTS and STT in a single testable flow.
    """

    text: str = Field(
        default='The quick brown fox jumps over the lazy dog.',
        description='Text to speak and then transcribe back',
    )
    voice: str = Field(
        default='alloy',
        description='Voice to use for TTS (alloy, echo, fable, onyx, nova, shimmer)',
    )


@ai.tool(description='calculates a gablorken', name='gablorkenTool')
def gablorken_tool(input_: GablorkenInput) -> int:
    """Calculate a gablorken.

    Args:
        input_: The input to calculate gablorken for.

    Returns:
        The calculated gablorken.
    """
    return input_.value * 3 - 5


@ai.tool(description='Get current temperature for provided coordinates in celsius')
def get_weather_tool(coordinates: WeatherRequest) -> float:
    """Get the current temperature for provided coordinates in celsius.

    Args:
        coordinates: The coordinates to get the weather for.

    Returns:
        The current temperature for the provided coordinates.
    """
    url = (
        f'https://api.open-meteo.com/v1/forecast?'
        f'latitude={coordinates.latitude}&longitude={coordinates.longitude}'
        f'&current=temperature_2m'
    )
    with httpx.Client() as client:
        response = client.get(url)
        data = response.json()
        return float(data['current']['temperature_2m'])


@ai.flow()
async def calculate_gablorken(input: GablorkenFlowInput) -> str:
    """Generate a request to calculate gablorken according to gablorken_tool.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        A GenerateRequest object with the evaluation output
    """
    response = await ai.generate(
        prompt=f'what is the gablorken of {input.value}',
        model=openai_model('gpt-4'),
        tools=['gablorkenTool'],
    )

    return response.text


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
async def get_weather_flow(input: WeatherFlowInput) -> WeatherResponse:
    """Get the weather for a location.

    Args:
        input: Input with location to get weather for.

    Returns:
        The weather for the location.
    """
    response = await ai.generate(
        model=openai_model('gpt-4o-mini'),
        system='You are an assistant that provides current weather information in JSON format.',
        config={'model': 'gpt-4o-mini-2024-07-18', 'temperature': 1},
        prompt=f"What's the weather like in {input.location} today?",
        tools=['get_weather_tool'],
        output=Output(schema=WeatherResponse),
    )
    return WeatherResponse.model_validate(response.output)


@ai.flow()
async def get_weather_flow_stream(input: WeatherFlowInput) -> WeatherResponse:
    """Get the weather for a location using a stream.

    Args:
        input: Input with location to get weather for.

    Returns:
        The weather for the location.
    """
    stream, response = ai.generate_stream(
        model=openai_model('gpt-4o'),
        system=(
            'You are an assistant that provides current weather information in JSON format and calculates '
            'gablorken based on weather value'
        ),
        config={'model': 'gpt-4o-2024-08-06', 'temperature': 1},
        prompt=f"What's the weather like in {input.location} today?",
        tools=['get_weather_tool', 'gablorkenTool'],
        output=Output(schema=WeatherResponse),
    )
    async for _chunk in stream:
        pass
    final = await response
    return WeatherResponse.model_validate(final.output)


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        The response from the OpenAI API.
    """
    return await generate_greeting_logic(ai, input.name)


@ai.flow()
async def generate_with_system_prompt(input: SystemPromptInput) -> str:
    """Demonstrate system prompts to control model persona and behavior.

    System prompts give the model instructions about how to respond, such as
    adopting a specific persona, tone, or response format.

    See: https://genkit.dev/docs/models#system-prompts

    Args:
        input: Input with a question to ask.

    Returns:
        The model's response in the persona defined by the system prompt.
    """
    return await generate_with_system_prompt_logic(ai, input.question)


@ai.flow()
async def generate_multi_turn_chat(input: MultiTurnInput) -> str:
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
    return await generate_multi_turn_chat_logic(ai, input.destination)


@ai.flow()
async def structured_menu_suggestion(input: MenuSuggestionInput) -> MenuSuggestion:
    """Suggest a themed menu item using structured output.

    Demonstrates Genkit's structured output feature: the model returns
    data conforming to a Pydantic schema with multiple field types
    (str, float, list, bool) rather than free-form text.

    See: https://genkit.dev/docs/models#structured-output

    Args:
        input: Input with restaurant theme.

    Returns:
        A MenuSuggestion with name, description, price, allergens, etc.
    """
    response = await ai.generate(
        prompt=f'Suggest a menu item for a {input.theme}-themed restaurant.',
        output=Output(schema=MenuSuggestion),
    )
    return response.output


@ai.flow()
async def generate_streaming_story(input: StreamInput, ctx: ActionRunContext | None = None) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming greeting.
        ctx: Action context for streaming.

    Returns:
        The response from the OpenAI API.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


@ai.flow()
async def sum_two_numbers2(my_input: MyInput) -> int:
    """Sum two numbers.

    Args:
        my_input: The input to sum.

    Returns:
        The sum of the input.
    """
    return my_input.a + my_input.b


def _extract_media_url(response: object) -> str:
    """Extract the media data URI from a generate response.

    Model responses for TTS and image generation return MediaParts
    instead of TextParts. This helper finds the first MediaPart and
    returns its URL (a base64 data URI).
    """
    msg = getattr(response, 'message', None)
    if msg is not None:
        for part in msg.content:
            if isinstance(part.root, MediaPart) and part.root.media:
                return part.root.media.url
    # Fallback to text (empty string for media-only responses).
    return getattr(response, 'text', '')


@ai.flow()
async def generate_image(input: ImagePromptInput) -> str:
    """Generate an image using DALL-E.

    Sends a text prompt to the DALL-E model and returns the generated
    image as a base64 data URI. The image can be viewed directly in
    the Genkit Dev UI.

    Args:
        input: Input with text prompt for image generation.

    Returns:
        The base64 data URI of the generated image.
    """
    response = await ai.generate(
        model=openai_model('dall-e-3'),
        prompt=input.prompt,
    )
    return _extract_media_url(response)


@ai.flow()
async def text_to_speech(input: TTSInput) -> str:
    """Convert text to speech using OpenAI TTS.

    Sends text to the TTS model and returns the audio as a base64
    data URI. The Dev UI can play back the resulting audio.

    Args:
        input: Input with text and voice selection.

    Returns:
        The base64 data URI of the generated audio.
    """
    response = await ai.generate(
        model=openai_model('tts-1'),
        prompt=input.text,
        config={'voice': input.voice},
    )
    return _extract_media_url(response)


@ai.flow()
async def round_trip_tts_stt(input: RoundTripInput) -> str:
    """Round-trip demo: Text → Speech → Text.

    Demonstrates the full TTS + STT pipeline:
    1. Converts the input text to speech using OpenAI TTS.
    2. Extracts the base64 audio from the TTS response.
    3. Feeds that audio into OpenAI Whisper for transcription.
    4. Returns the transcribed text (should match the original input).

    This flow is self-contained and can be tested directly from the
    Dev UI without needing to provide raw audio data.

    Args:
        input: Input with text to speak and voice selection.

    Returns:
        The transcribed text from the round-trip.
    """
    # Step 1: Generate audio from text via TTS.
    tts_response = await ai.generate(
        model=openai_model('tts-1'),
        prompt=input.text,
        config={'voice': input.voice},
    )
    audio_data_uri = _extract_media_url(tts_response)
    if not audio_data_uri:
        return 'Error: TTS did not return audio data.'

    # Step 2: Transcribe the audio back to text via STT.
    stt_response = await ai.generate(
        model=openai_model('whisper-1'),
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=MediaPart(media=Media(url=audio_data_uri, content_type='audio/mpeg'))),
                    Part(root=TextPart(text='Transcribe this audio')),
                ],
            )
        ],
    )
    return stt_response.text


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using GPT-4o vision capabilities.

    This demonstrates multimodal image understanding via the
    OpenAI-compatible plugin. GPT-4o can analyze images and
    provide detailed descriptions.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url)


@ai.flow()
async def solve_reasoning_problem(input: ReasoningInput) -> str:
    """Solve reasoning problems using OpenAI o4-mini.

    o4-mini is a reasoning model that shows chain-of-thought
    steps. This demonstrates how reasoning models work with
    the OpenAI-compatible plugin.

    Args:
        input: Input with reasoning question to solve.

    Returns:
        The reasoning and answer.
    """
    return await solve_reasoning_problem_logic(ai, input.prompt, model=openai_model('o4-mini'))


@ai.flow()
async def generate_code(input: CodeInput) -> str:
    """Generate code using OpenAI models.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


@ai.flow()
async def generate_streaming_with_tools(
    input: StreamingToolInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Demonstrate streaming generation with tool calling.

    The model streams its response while also calling tools mid-generation.
    Tool calls are resolved automatically and the model continues generating.

    Args:
        input: Input with location for weather lookup.
        ctx: Action context for streaming chunks to the client.

    Returns:
        The complete generated text.
    """
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


async def main() -> None:
    """Main entry point for the OpenAI sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    _ = await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
