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

"""Hello Google GenAI sample - Comprehensive Google AI features demo.

This sample demonstrates the full range of Google GenAI plugin capabilities,
from basic generation to advanced features like tool calling, streaming,
structured output, and multimodal inputs.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Genkit              │ A framework for building AI apps. Like Lego       │
    │                     │ blocks for connecting AI models to your code.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Flow                │ A function that does AI work. Decorated with      │
    │                     │ @ai.flow() so Genkit can track and manage it.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool                │ A function the AI can call. Like giving the AI    │
    │                     │ a calculator or search engine to use.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Generation          │ Asking the AI to write something. "Generate       │
    │                     │ a poem about cats" → AI writes the poem.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Getting the response word-by-word as it's made.   │
    │                     │ Feels faster, like watching someone type.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Structured Output   │ Getting AI responses as proper data (JSON).       │
    │                     │ Not just text, but name="Bob", age=25, etc.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Embedding           │ Converting text to numbers that capture meaning.  │
    │                     │ "Happy" and "joyful" become similar numbers.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Multimodal          │ AI that understands multiple types of input.      │
    │                     │ Not just text, but also images and videos.        │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                                      | Example Function / Code Snippet        |
|----------------------------------------------------------|----------------------------------------|
| Plugin Initialization                                    | `ai = Genkit(plugins=[GoogleAI()])`    |
| Default Model Configuration                              | `ai = Genkit(model=...)`               |
| Defining Flows                                           | `@ai.flow()` decorator (multiple uses) |
| Defining Tools                                           | `@ai.tool()` decorator (multiple uses) |
| Pydantic for Tool Input Schema                           | `GablorkenInput`                       |
| Simple Generation (Prompt String)                        | `generate_greeting`                    |
| System Prompts                                           | `generate_with_system_prompt`          |
| Multi-turn Conversations (`messages`)                    | `generate_multi_turn_chat`             |
| Generation with Messages (`Message`, `Role`, `TextPart`) | `simple_generate_with_tools_flow`      |
| Generation with Tools                                    | `simple_generate_with_tools_flow`      |
| Tool Response Handling                                   | `simple_generate_with_interrupts`      |
| Tool Interruption (`ctx.interrupt`)                      | `gablorken_tool2`                      |
| Embedding (`ai.embed`, `Document`)                       | `embed_docs`                           |
| Generation Configuration (`temperature`, etc.)           | `generate_with_config`                 |
| Streaming Generation (`ai.generate_stream`)              | `generate_streaming_story`             |
| Streaming Chunk Handling (`ctx.send_chunk`)              | `generate_streaming_story`, `generate_character` |
| Streaming Structured Output                              | `streaming_structured_output`          |
| Structured Output (Schema)                               | `generate_character`                   |
| Pydantic for Structured Output Schema                    | `RpgCharacter`                         |
| Structured Output (Instruction-Based)                    | `generate_character_instructions`      |
| Multi-modal Output Configuration                         | `generate_images`                      |
| GCP Telemetry (Traces and Metrics)                       | `add_gcp_telemetry()`                  |
| Thinking Mode (CoT)                                      | `thinking_level_pro`, `thinking_level_flash` |
| Code Generation                                          | `generate_code`                        |
| Search Grounding                                         | `search_grounding`                     |
| URL Context                                              | `url_context`                          |
| Multimodal Generation (Video input)                      | `youtube_videos`                       |
| Context Propagation                                      | `context_demo`                         |

Edge Cases
==========
The following edge cases were discovered during testing and are worth noting:

1. **Tool inputs must be Pydantic models, not bare primitives.**
   LLMs always send tool arguments as JSON objects with named keys (e.g.
   ``{'celsius': 21.5}``). A tool with a bare ``float`` input generates
   a ``{'type': 'number'}`` schema that doesn't match the object the model
   sends, causing a validation ``TypeError``. Always wrap tool inputs in a
   ``BaseModel``. See ``CelsiusInput`` below.

2. **YouTube URLs are natively resolved by the Gemini API.**
   Do not try to download YouTube video URLs — the HTTP response is an HTML
   page, not raw video. The Gemini API resolves YouTube URLs server-side
   when passed as ``file_data`` with the original URL. The plugin handles
   this automatically via ``_GEMINI_NATIVE_HOSTS``.

3. **Use ``gemini-embedding-001``, not ``text-embedding-004``.**
   The ``text-embedding-004`` model returns ``404 NOT_FOUND`` on the
   ``v1beta`` API endpoint with an API key. Use ``gemini-embedding-001``
   as the replacement.

4. **``GoogleSearch`` vs ``GoogleSearchRetrieval``.**
   The ``google.genai`` SDK's ``Tool.google_search`` field expects a
   ``GoogleSearch`` object. The legacy ``GoogleSearchRetrieval`` type is
   for the separate ``Tool.google_search_retrieval`` field. Mixing them
   up produces a silent type mismatch warning.
"""

import argparse
import asyncio
import os
import sys
import tempfile

from google import genai as google_genai_sdk

if sys.version_info < (3, 11):
    from strenum import StrEnum  # pyright: ignore[reportUnreachable]
else:
    from enum import StrEnum

import pathlib

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output, ToolRunContext, tool_response
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.evaluators import GenkitMetricType, MetricConfig, define_genkit_evaluators
from genkit.plugins.google_cloud import add_gcp_telemetry
from genkit.plugins.google_genai import (
    EmbeddingTaskType,
    GoogleAI,
)
from genkit.types import (
    GenerationCommonConfig,
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
)
from samples.shared import (
    CharacterInput,
    CodeInput,
    ConfigInput,
    GreetingInput,
    ImageDescribeInput,
    MultiTurnInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    WeatherInput,
    convert_currency as _convert_currency_tool,
    describe_image_logic,
    generate_character_logic,
    generate_code_logic,
    generate_greeting_logic,
    generate_multi_turn_chat_logic,
    generate_streaming_story_logic,
    generate_streaming_with_tools_logic,
    generate_with_config_logic,
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
)

setup_sample()

logger = get_logger(__name__)


if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')


ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-pro-preview',
)

define_genkit_evaluators(
    ai,
    [
        MetricConfig(metric_type=GenkitMetricType.REGEX),
        MetricConfig(metric_type=GenkitMetricType.DEEP_EQUAL),
        MetricConfig(metric_type=GenkitMetricType.JSONATA),
    ],
)

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


class GablorkenInput(BaseModel):
    """The Pydantic model for tools."""

    value: int = Field(description='value to calculate gablorken for')


class ThinkingLevel(StrEnum):
    """Thinking level enum."""

    LOW = 'LOW'
    HIGH = 'HIGH'


class ThinkingLevelFlash(StrEnum):
    """Thinking level flash enum."""

    MINIMAL = 'MINIMAL'
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class TemperatureInput(BaseModel):
    """Input for temperature config flow."""

    data: str = Field(default='Mittens', description='Name to greet')


class ToolsFlowInput(BaseModel):
    """Input for tools flow."""

    value: int = Field(default=42, description='Value for gablorken calculation')


class ToolCallingInput(BaseModel):
    """Input for tool calling flow."""

    location: str = Field(default='Paris, France', description='Location to get weather for')


class ContextDemoInput(BaseModel):
    """Input for context demo flow."""

    user_id: int = Field(default=42, description='User ID (try 42 or 123)')


class ScreenshotInput(BaseModel):
    """Input for screenshot tool."""

    url: str = Field(description='The URL to take a screenshot of')


@ai.tool(name='gablorkenTool')
def gablorken_tool(input_: GablorkenInput) -> dict[str, int]:
    """Calculate a gablorken.

    Returns:
        The calculated gablorken.
    """
    return {'result': input_.value * 3 - 5}


@ai.tool(name='gablorkenTool2')
def gablorken_tool2(_input: GablorkenInput, ctx: ToolRunContext) -> None:
    """The user-defined tool function."""
    pass


@ai.tool(name='screenShot')
def take_screenshot(input_: ScreenshotInput) -> dict:
    """Take a screenshot of a given URL."""
    return {'url': input_.url, 'screenshot_path': '/tmp/screenshot.png'}  # noqa: S108 - sample code


@ai.tool(name='getWeather')
def get_weather_detailed(input_: WeatherInput) -> dict:
    """Used to get current weather for a location."""
    return {
        'location': input_.location,
        'temperature_celcius': 21.5,
        'conditions': 'cloudy',
    }


class CelsiusInput(BaseModel):
    """Input for the Celsius to Fahrenheit conversion tool."""

    celsius: float = Field(description='Temperature in Celsius to convert')


@ai.tool(name='celsiusToFahrenheit')
def celsius_to_fahrenheit(input_: CelsiusInput) -> float:
    """Converts Celsius to Fahrenheit."""
    return (input_.celsius * 9) / 5 + 32


@ai.tool()
def get_user_data() -> str:
    """Fetch user data based on context."""
    context = Genkit.current_context()
    raw_user = context.get('user') if context else {}
    user_id = 0
    if isinstance(raw_user, dict):
        user_id = int(raw_user.get('id', 0))
    if user_id == 42:
        return 'User is Arthur Dent, an intergalactic traveler.'
    elif user_id == 123:
        return 'User is Jane Doe, a premium member.'
    else:
        return 'User is Guest.'


@ai.flow()
async def simple_generate_with_interrupts(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response1 = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text=f'what is a gablorken of {input.value}'))],
            ),
        ],
        tools=['gablorkenTool2'],
    )
    await logger.ainfo(f'len(response.tool_requests)={len(response1.tool_requests)}')
    if len(response1.interrupts) == 0:
        return response1.text

    tr = tool_response(response1.interrupts[0], {'output': 178})
    response = await ai.generate(
        messages=response1.messages,
        tool_responses=[tr],
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        The generated greeting response.
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
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using Gemini.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url)


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
async def generate_streaming_story(
    input: StreamInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming.
        ctx: Action context for streaming.

    Returns:
        The complete story text.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


@ai.flow()
async def generate_with_config(input: ConfigInput) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        input: Input with name for greeting.

    Returns:
        The generated greeting.
    """
    return await generate_with_config_logic(ai, input.name)


@ai.flow()
async def search_grounding() -> str:
    """Search grounding demo - retrieves current info from the web.

    This flow demonstrates Google Search grounding, which allows the model
    to access real-time information from the web to answer questions.

    Returns:
        Information about cats with web-grounded facts.
    """
    response = await ai.generate(
        prompt='What are the most popular cat breeds in 2024 and their characteristics?',
        config={'google_search_retrieval': True},
    )
    return response.text


@ai.flow()
async def simple_generate_with_tools_flow(input: ToolsFlowInput) -> str:
    """Generate a greeting for the given name.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The generated response with a function.
    """
    response = await ai.generate(
        prompt=f'what is a gablorken of {input.value}',
        tools=['gablorkenTool'],
    )
    return response.text


@ai.flow()
async def thinking_level_flash(_level: ThinkingLevelFlash) -> str:
    """Gemini 3.0 thinkingLevel config (Flash)."""
    response = await ai.generate(
        prompt=(
            'Alice, Bob, and Carol each live in a different house on the '
            'same street: red, green, and blue. The person who lives in the red house '
            'owns a cat. Bob does not live in the green house. Carol owns a dog. The '
            'green house is to the left of the red house. Alice does not own a cat. '
            'The person in the blue house owns a fish. '
            'Who lives in each house, and what pet do they own? Provide your '
            'step-by-step reasoning.'
        ),
        config={
            'thinking_config': {
                'include_thoughts': True,
            }
        },
    )
    return response.text


@ai.flow()
async def thinking_level_pro(_level: ThinkingLevel) -> str:
    """Gemini 3.0 thinkingLevel config (Pro)."""
    response = await ai.generate(
        prompt=(
            'Alice, Bob, and Carol each live in a different house on the '
            'same street: red, green, and blue. The person who lives in the red house '
            'owns a cat. Bob does not live in the green house. Carol owns a dog. The '
            'green house is to the left of the red house. Alice does not own a cat. '
            'The person in the blue house owns a fish. '
            'Who lives in each house, and what pet do they own? Provide your '
            'step-by-step reasoning.'
        ),
        config={
            'thinking_config': {
                'include_thoughts': True,
            }
        },
    )
    return response.text


@ai.flow()
async def url_context() -> str:
    """URL context demo - analyzes content from web pages.

    This flow demonstrates URL context feature, allowing the model to
    read and analyze content from specified web URLs.

    Returns:
        Analysis of cat care information from the web.
    """
    response = await ai.generate(
        prompt=('Summarize the key cat care tips from https://www.aspca.org/pet-care/cat-care/general-cat-care'),
        config={'url_context': {}, 'api_version': 'v1alpha'},
    )
    return response.text


async def create_file_search_store(client: google_genai_sdk.Client) -> str:
    """Creates a file search store."""
    file_search_store = await client.aio.file_search_stores.create()
    if not file_search_store.name:
        raise ValueError('File Search Store created without a name.')
    return file_search_store.name


async def upload_blob_to_file_search_store(client: google_genai_sdk.Client, file_search_store_name: str) -> None:
    """Uploads a blob to the file search store."""
    text_content = (
        'The Whispering Woods In the heart of Eldergrove, there stood a forest whispered about by the villagers. '
        'They spoke of trees that could talk and streams that sang. Young Elara, curious and adventurous, '
        'decided to explore the woods one crisp autumn morning. As she wandered deeper, the leaves rustled with '
        'excitement, revealing hidden paths. Elara noticed the trees bending slightly as if beckoning her to come '
        'closer. When she paused to listen, she heard soft murmurs—stories of lost treasures and forgotten dreams. '
        'Drawn by the enchanting sounds, she followed a narrow trail until she stumbled upon a shimmering pond. '
        'At its edge, a wise old willow tree spoke, "Child of the village, what do you seek?" "I seek adventure," '
        'Elara replied, her heart racing. "Adventure lies not in faraway lands but within your spirit," the willow '
        'said, swaying gently. "Every choice you make is a step into the unknown." With newfound courage, Elara left '
        'the woods, her mind buzzing with possibilities. The villagers would say the woods were magical, but to Elara, '
        'it was the spark of her imagination that had transformed her ordinary world into a realm of endless '
        'adventures. She smiled, knowing her journey was just beginning'
    )

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp:
        tmp.write(text_content)
        tmp_path = tmp.name

    try:
        # Use the high-level helper to upload directly to the store with metadata
        op = await client.aio.file_search_stores.upload_to_file_search_store(
            file_search_store_name=file_search_store_name,
            file=tmp_path,
            config={'custom_metadata': [{'key': 'author', 'string_value': 'foo'}]},
        )

        # Poll for completion
        while not op.done:
            await asyncio.sleep(2)
            # Fetch the updated operation status
            op = await client.aio.operations.get(operation=op)

    finally:
        pathlib.Path(tmp_path).unlink()
    return


async def delete_file_search_store(client: google_genai_sdk.Client, name: str) -> None:
    """Deletes the file search store."""
    await client.aio.file_search_stores.delete(name=name, config={'force': True})


@ai.flow()
async def file_search() -> str:
    """File Search."""
    # Create a client using the same API Key as the plugin
    api_key = os.environ.get('GEMINI_API_KEY')
    client = google_genai_sdk.Client(api_key=api_key)

    # 1. Create Store
    store_name = await create_file_search_store(client)

    try:
        # 2. Upload Blob (Story)
        await upload_blob_to_file_search_store(client, store_name)

        # 3. Generate
        response = await ai.generate(
            prompt="What is the character's name in the story?",
            config={
                'file_search': {
                    'file_search_store_names': [store_name],
                    'metadata_filter': 'author=foo',
                },
                'api_version': 'v1alpha',
            },
        )
        return response.text
    finally:
        # 4. Cleanup
        await delete_file_search_store(client, store_name)


@ai.flow()
async def embed_docs(
    docs: list[str] | None = None,
) -> list[dict]:
    """Generate an embedding for the words in a list.

    Args:
        docs: list of texts (string)

    Returns:
        The generated embeddings as serializable dicts.
    """
    if docs is None:
        docs = ['Hello world', 'Genkit is great', 'Embeddings are fun']
    options = {'task_type': EmbeddingTaskType.CLUSTERING}
    embeddings = await ai.embed_many(
        embedder='googleai/gemini-embedding-001',
        content=docs,
        options=options,
    )
    # Serialize embeddings to dicts for JSON compatibility
    return [emb.model_dump(by_alias=True) for emb in embeddings]


@ai.flow()
async def youtube_videos() -> str:
    """YouTube videos."""
    response = await ai.generate(
        prompt=[
            Part(root=TextPart(text='transcribe this video')),
            Part(
                root=MediaPart(media=Media(url='https://www.youtube.com/watch?v=3p1P5grjXIQ', content_type='video/mp4'))
            ),
        ],
        config={},
    )
    return response.text


@ai.flow()
async def tool_calling(input: ToolCallingInput) -> str:
    """Tool calling with Gemini."""
    response = await ai.generate(
        tools=['getWeather', 'celsiusToFahrenheit'],
        prompt=f"What's the weather in {input.location}? Convert the temperature to Fahrenheit.",
        config=GenerationCommonConfig(temperature=1),
    )
    return response.text


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
async def generate_code(input: CodeInput) -> str:
    """Generate code using Gemini.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


@ai.flow()
async def context_demo(input: ContextDemoInput) -> str:
    """Demonstrate passing context to tools.

    This flow shows how to propagate application context (like user ID or auth info)
    from the flow input into the generation and tool execution.

    Args:
        input: Input with user ID.

    Returns:
        The model's response using the context-dependent tool output.
    """
    response = await ai.generate(
        prompt='Tell me about the current user based on their ID.',
        tools=['get_user_data'],
        context={'user': {'id': input.user_id}},
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
    """Main function - keep alive for Dev UI."""
    await logger.ainfo('Starting main execution loop')
    while True:
        await asyncio.sleep(3600)
    await logger.ainfo('Exiting main execution loop')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Google GenAI Hello Sample')
    _ = parser.add_argument(
        '--enable-gcp-telemetry',
        action='store_true',
        help='Enable Google Cloud Platform telemetry',
    )
    args = parser.parse_args()
    if args.enable_gcp_telemetry:
        add_gcp_telemetry()
    ai.run_main(main())
