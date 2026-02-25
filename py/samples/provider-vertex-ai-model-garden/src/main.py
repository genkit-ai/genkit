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

"""Model Garden sample.

This sample demonstrates how to use Vertex AI Model Garden, which provides
access to various third-party models (like Anthropic Claude) through
Google Cloud's infrastructure.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model Garden        │ Google Cloud's model marketplace. Access Claude,   │
    │                     │ Llama, Mistral, etc. through one platform.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Third-party Models  │ Models from other companies (Anthropic, Meta).     │
    │                     │ Run on Google's infrastructure.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ GCP Credentials     │ Your Google Cloud login. One auth method for       │
    │                     │ all models (no separate API keys needed).          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ModelGardenPlugin   │ The plugin that connects to Model Garden.          │
    │                     │ Add it to Genkit, access many models.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ model_garden_name   │ Helper to create model references.                 │
    │                     │ "anthropic/claude-3-5-sonnet" becomes full path.   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Model Garden Plugin                     | `ModelGardenPlugin()`               |
| Specific Model Usage                    | `model_garden_name('anthropic/...')`|
| Simple Generation                       | `say_hi`                            |
| System Prompt                           | `system_prompt`                     |
| Multi-turn Conversation                 | `multi_turn_chat`                   |
| Streaming Generation                    | `say_hi_stream`                     |
| Tool Calling                            | `weather_flow`                      |
| Structured Output                       | `generate_character`                |
| Streaming Structured Output             | `streaming_structured_output`       |
| Generation Config                       | `max_output_tokens`, `temperature`  |

See README.md for testing instructions.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import VertexAI
from genkit.plugins.vertex_ai.model_garden import ModelGardenPlugin, model_garden_name
from genkit.types import Message, Part, Role, TextPart
from samples.shared.logging import setup_sample

setup_sample()

logger = get_logger(__name__)


class RpgCharacter(BaseModel):
    """An RPG character with stats."""

    name: str = Field(..., description='Character name')
    backstory: str = Field(..., description='Character backstory')
    abilities: list[str] = Field(..., description='List of character abilities')


def get_project_id() -> str:
    """Get Google Cloud project ID from environment."""
    project_id = os.getenv('GCLOUD_PROJECT') or os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        project_id = input('Enter your Google Cloud Project ID: ').strip()
        if not project_id:
            raise ValueError('GCLOUD_PROJECT, GOOGLE_CLOUD_PROJECT or user input must be set.')

    # Sanitize project_id to remove potential smart quotes or regular quotes
    project_id = project_id.strip().strip("'").strip('"').strip(""").strip(""")

    # Update env vars so other plugins (like VertexAI) pick up the sanitized ID
    os.environ['GCLOUD_PROJECT'] = project_id
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

    return project_id


def get_location() -> str:
    """Get Google Cloud location from environment."""
    location = os.getenv('GOOGLE_CLOUD_LOCATION') or os.getenv('GOOGLE_CLOUD_REGION')
    if not location:
        location = input('Enter your Google Cloud Location (default: us-central1): ').strip()
        if not location:
            location = 'us-central1'
    os.environ['GOOGLE_CLOUD_LOCATION'] = location
    return location


project_id = get_project_id()
location = get_location()

ai = Genkit(
    plugins=[
        ModelGardenPlugin(
            project_id=project_id,
            location=location,
            model_locations={
                'anthropic/claude-sonnet-4@20250514': 'us-east5',
                'anthropic/claude-3-5-sonnet-v2@20241022': 'us-east5',
                'meta/llama-3.2-90b-vision-instruct-maas': 'us-central1',
                'mistralai/ministral-3-14b-instruct-2512': 'us-central1',
            },
        ),
        VertexAI(location=location),
    ],
)


class WeatherInput(BaseModel):
    """Input for getting weather."""

    location: str = Field(..., description='Location for which to get the weather, ex: San-Francisco, CA')


class SayHiInput(BaseModel):
    """Input for say_hi flow."""

    name: str = Field(default='Mittens', description='Name to greet')


class SystemPromptInput(BaseModel):
    """Input for system_prompt flow."""

    question: str = Field(default='What is your quest?', description='Question to ask')


class MultiTurnInput(BaseModel):
    """Input for multi_turn_chat flow."""

    destination: str = Field(default='Japan', description='Travel destination')


class StreamInput(BaseModel):
    """Input for streaming flow."""

    name: str = Field(default='Shadow', description='Name for streaming greeting')


class CharacterInput(BaseModel):
    """Input for character generation."""

    name: str = Field(default='Whiskers', description='Character name')


class JokesInput(BaseModel):
    """Input for jokes flow."""

    subject: str = Field(default='banana', description='Subject for the joke')


class WeatherFlowInput(BaseModel):
    """Input for weather flow."""

    location: str = Field(default='Paris, France', description='Location to get weather for')


class TemperatureInput(BaseModel):
    """Input for converting temperature."""

    celsius: float = Field(..., description='Temperature in Celsius')


class ToolFlowInput(BaseModel):
    """Input for tool flow."""

    location: str = Field('Paris, France', description='Location to check weather for')


@ai.tool()
def get_weather(input: WeatherInput) -> dict:
    """Used to get current weather for a location."""
    return {
        'location': input.location,
        'temperature_celsius': 21.5,
        'conditions': 'cloudy',
    }


@ai.tool()
def celsius_to_fahrenheit(input: TemperatureInput) -> float:
    """Converts Celsius to Fahrenheit."""
    return (input.celsius * 9) / 5 + 32


@ai.tool(name='getWeather')
def get_weather_tool(input_: WeatherInput) -> str:
    """Used to get current weather for a location."""
    return f'Weather in {input_.location}: Sunny, 21.5°C'


@ai.flow(name='gemini-2.5-flash - tool_flow')
async def gemini_model(input: ToolFlowInput) -> str:
    """Gemini tool flow.

    Args:
        input: The location input.
    """
    response = await ai.generate(
        model='vertexai/gemini-2.5-flash',
        config={'temperature': 1},
        tools=['get_weather', 'celsius_to_fahrenheit'],
        prompt=f"What's the weather in {input.location}? Convert the temperature to Fahrenheit.",
    )
    return response.text


@ai.flow(name='llama-3.2 - basic_flow')
async def llama_model() -> str:
    """Generate a greeting."""
    try:
        response = await ai.generate(
            model=model_garden_name('meta/llama-3.2-90b-vision-instruct-maas'),
            config={
                'temperature': 1,
            },
            prompt='You are a helpful assistant named Walt. Say hello',
        )
        logger.info(f'Response received: {response.text[:100] if response.text else "None"}')
        return response.text
    except Exception as e:
        logger.error(f'Error in llama 3.2 basic_flow: {e}', exc_info=True)
        raise


@ai.flow(name='claude-sonnet-4 - tool_calling_flow')
async def anthropic_model(input: ToolFlowInput) -> str:
    """Anthropic tool flow.

    Args:
        input: The location input.
    """
    response = await ai.generate(
        model=model_garden_name('anthropic/claude-sonnet-4@20250514'),
        config={'temperature': 1},
        tools=['get_weather', 'celsius_to_fahrenheit'],
        prompt=f"What's the weather in {input.location}? Convert the temperature to Fahrenheit.",
    )
    return response.text


@ai.flow()
async def generate_character(input: CharacterInput) -> RpgCharacter:
    """Generate an RPG character.

    Args:
        input: Input with character name.

    Returns:
        The generated RPG character.
    """
    result = await ai.generate(
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
        prompt=f'generate an RPG character named {input.name}',
        output=Output(schema=RpgCharacter),
    )
    return result.output


@ai.flow()
async def jokes_flow(input: JokesInput) -> str:
    """Generate a joke about the given subject.

    Args:
        input: Input with joke subject.

    Returns:
        The generated joke.
    """
    response = await ai.generate(
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
        config={'temperature': 1, 'max_output_tokens': 1024},
        prompt=f'Tell a short joke about {input.subject}',
    )

    return response.text


@ai.flow()
async def say_hi(input: SayHiInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with name to greet.

    Returns:
        The generated greeting response.
    """
    response = await ai.generate(
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
        config={'temperature': 1},
        prompt=f'hi {input.name}',
    )

    return response.text


@ai.flow()
async def say_hi_stream(
    input: StreamInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Say hi to a name and stream the response.

    Args:
        input: Input with name for streaming.
        ctx: Action context for streaming.

    Returns:
        The response from the model.
    """
    stream, _ = ai.generate_stream(
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
        config={'temperature': 1},
        prompt=f'hi {input.name}',
    )
    result = ''
    async for data in stream:
        if ctx is not None:
            ctx.send_chunk(data.text)
        result += data.text
    return result


@ai.flow()
async def weather_flow(input: WeatherFlowInput) -> str:
    """Tool calling with Model Garden."""
    response = await ai.generate(
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
        tools=['getWeather'],
        prompt=f"What's the weather in {input.location}?",
    )

    return response.text


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
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
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
    model = model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022')

    # Turn 1: Start the conversation
    response1 = await ai.generate(
        model=model,
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
        model=model,
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
        model=model_garden_name('anthropic/claude-3-5-sonnet-v2@20241022'),
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


async def main() -> None:
    """Main entry point for the Model Garden sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
