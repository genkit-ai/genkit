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

"""Common flow logic for provider samples.

Each function takes a ``Genkit`` instance (and typed inputs) so that
provider samples can delegate to them from thin ``@ai.flow()`` wrappers.
Provider-specific flow logic stays in each sample's main.py.
"""

from genkit.ai import Genkit, Output
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.types import Media, MediaPart, Message, Part, Role, TextPart

from .types import CalculatorInput, CurrencyExchangeInput, RpgCharacter, WeatherInput

logger = get_logger(__name__)


async def calculation_logic(ai: Genkit, input: CalculatorInput, model: str | None = None) -> str:
    """Perform a calculation via an LLM tool call.

    Args:
        ai: The initialized Genkit instance.
        input: Calculator parameters.
        model: Optional model override (needed for providers whose default
            model doesn't support tool calling, e.g. Ollama/Gemma).

    Returns:
        Calculation result.
    """
    response = await ai.generate(
        model=model,
        prompt=f'Calculate {input.a} {input.operation} {input.b}',
        tools=['calculate'],
    )
    return response.text


async def describe_image_logic(ai: Genkit, image_url: str, model: str | None = None) -> str:
    """Describe an image using multimodal generation.

    Args:
        ai: The Genkit instance.
        image_url: URL of the image to describe.
        model: Optional model override for provider-specific vision models.

    Returns:
        A textual description of the image.
    """
    response = await ai.generate(
        model=model,
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='Describe this image in detail')),
                    Part(root=MediaPart(media=Media(url=image_url, content_type='image/jpeg'))),
                ],
            )
        ],
    )
    return response.text


async def convert_currency_logic(ai: Genkit, input: CurrencyExchangeInput, model: str | None = None) -> str:
    """Convert currency using tool calling.

    Args:
        ai: The initialized Genkit instance.
        input: Validated currency exchange parameters.
        model: Optional model override (needed for providers whose default
            model doesn't support tool calling, e.g. Ollama/Gemma).

    Returns:
        Conversion result.
    """
    response = await ai.generate(
        model=model,
        prompt=f'Convert {input.amount} {input.from_currency} to {input.to_currency}',
        tools=['convert_currency'],
    )
    return response.text


async def generate_character_logic(ai: Genkit, name: str) -> RpgCharacter:
    """Generate an RPG character with structured output.

    Args:
        ai: The Genkit instance.
        name: The name of the character.

    Returns:
        The generated RPG character.
    """
    # Example schema hint for models that don't fully support constrained JSON output
    schema_hint = """
        Example output:
        {
            "name": "<character_name>",
            "backStory": "<character_backStory>",
            "abilities": ["<character_ability1>", "<character_ability2>"],
            "skills": {
                "strength": <character_strength>,
                "charisma": <character_charisma>,
                "endurance": <character_endurance>
            }
        }
    """
    result = await ai.generate(
        prompt=f'Generate a RPG character named {name}.\n{schema_hint}',
        output=Output(schema=RpgCharacter),
    )
    return result.output


async def generate_code_logic(ai: Genkit, task: str, model: str | None = None) -> str:
    """Generate code for a given task.

    Args:
        ai: The Genkit instance.
        task: Coding task description.
        model: Optional model override for provider-specific code models.

    Returns:
        Generated code.
    """
    response = await ai.generate(
        model=model,
        prompt=task,
        system='You are an expert programmer. Provide clean, well-documented code with explanations.',
    )
    return response.text


async def generate_greeting_logic(ai: Genkit, name: str) -> str:
    """Generate a simple greeting.

    Args:
        ai: The Genkit instance.
        name: Name to greet.

    Returns:
        Greeting message from the LLM.
    """
    response = await ai.generate(prompt=f'Say hello to {name}!')
    return response.text


async def generate_multi_turn_chat_logic(ai: Genkit, destination: str) -> str:
    """Demonstrate multi-turn conversations using the messages parameter.

    Builds a 2-turn travel conversation where the second turn requires
    context from the first.

    Args:
        ai: The Genkit instance.
        destination: Travel destination.

    Returns:
        The model's final response, demonstrating context retention.
    """
    response1 = await ai.generate(
        system='You are a helpful travel assistant.',
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text=f'I want to visit {destination} for two weeks in spring.'))],
            ),
        ],
    )
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


async def generate_streaming_story_logic(ai: Genkit, name: str, ctx: ActionRunContext | None) -> str:
    """Generate a streaming story response.

    Args:
        ai: The Genkit instance.
        name: Name to greet.
        ctx: Action context for streaming.

    Returns:
        Complete story text.
    """
    stream, response = ai.generate_stream(
        prompt=f'Tell me a short story about {name}',
    )
    async for chunk in stream:
        if chunk.text:
            if ctx is not None:
                ctx.send_chunk(chunk.text)
    return (await response).text


async def generate_streaming_with_tools_logic(
    ai: Genkit, location: str, ctx: ActionRunContext | None, model: str | None = None
) -> str:
    """Demonstrate streaming generation with tool calling.

    The model streams its response while also calling tools mid-generation.

    Args:
        ai: The Genkit instance.
        location: Location for weather lookup.
        ctx: Action context for streaming chunks to the client.
        model: Optional model override (needed for providers whose default
            model doesn't support tool calling, e.g. Ollama/Gemma).

    Returns:
        The complete generated text.
    """
    stream, response = ai.generate_stream(
        model=model,
        prompt=f'What is the weather in {location}? Describe it poetically.',
        tools=['get_weather'],
    )
    async for chunk in stream:
        if chunk.text:
            if ctx is not None:
                ctx.send_chunk(chunk.text)
    return (await response).text


async def generate_weather_logic(ai: Genkit, input: WeatherInput, model: str | None = None) -> str:
    """Get weather information using tool calling.

    Args:
        ai: The Genkit instance.
        input: Weather input data.
        model: Optional model override (needed for providers whose default
            model doesn't support tool calling, e.g. Ollama/Gemma).

    Returns:
        Formatted weather string.
    """
    response = await ai.generate(
        model=model,
        prompt=f'What is the weather in {input.location}?',
        tools=['get_weather'],
    )
    return response.text


async def generate_with_config_logic(ai: Genkit, name: str) -> str:
    """Generate a greeting with custom model configuration.

    Args:
        ai: The Genkit instance.
        name: User name.

    Returns:
        Greeting message from the LLM.
    """
    response = await ai.generate(
        prompt=f'Write a creative greeting for {name}',
        config={'temperature': 1.0, 'max_output_tokens': 200},
    )
    return response.text


async def generate_with_system_prompt_logic(ai: Genkit, question: str) -> str:
    """Demonstrate system prompts to control model persona and behavior.

    Uses a pirate captain persona as a fun, recognizable example.

    Args:
        ai: The Genkit instance.
        question: Question to ask.

    Returns:
        The model's response in the persona defined by the system prompt.
    """
    response = await ai.generate(
        prompt=question,
        system='You are a pirate captain from the 18th century. Always respond in character, '
        'using pirate slang and nautical terminology.',
    )
    return response.text


async def solve_reasoning_problem_logic(ai: Genkit, prompt: str, model: str | None = None) -> str:
    """Solve reasoning problems using a reasoning model.

    Args:
        ai: The Genkit instance.
        prompt: Reasoning question to solve.
        model: Optional model override for provider-specific reasoning models.

    Returns:
        The reasoning and answer.
    """
    response = await ai.generate(
        model=model,
        prompt=prompt,
    )
    return response.text


async def translate_text_logic(
    ai: Genkit,
    text: str,
    target_language: str,
    model: str | None = None,
) -> str:
    """Translate text to a target language.

    Args:
        ai: The Genkit instance.
        text: Text to translate.
        target_language: Target language name (e.g. "French").
        model: Optional model override for provider-specific translation models.

    Returns:
        The translated text.
    """
    response = await ai.generate(
        model=model,
        prompt=f'Translate the following text to {target_language}:\n{text}',
        system=f'You are a professional translator. Output only the {target_language} translation, nothing else.',
    )
    return response.text


async def chat_flow_logic(
    ai: Genkit,
    system_prompt: str,
    prompt1: str,
    followup_question: str,
    final_question: str,
) -> str:
    """Run a 3-turn conversation demonstrating context retention.

    Args:
        ai: The Genkit instance.
        system_prompt: System prompt for all turns.
        prompt1: First user message that sets context.
        followup_question: Second turn question requiring first-turn context.
        final_question: Third turn question building on prior context.

    Returns:
        The model's response to the final question.
    """
    history: list[Message] = []

    response1 = await ai.generate(prompt=prompt1, system=system_prompt)
    history.append(Message(role=Role.USER, content=[Part(root=TextPart(text=prompt1))]))
    if response1.message:
        history.append(response1.message)
    await logger.ainfo('chat_flow turn 1', result=response1.text)

    response2 = await ai.generate(
        messages=[
            *history,
            Message(role=Role.USER, content=[Part(root=TextPart(text=followup_question))]),
        ],
        system=system_prompt,
    )
    history.append(Message(role=Role.USER, content=[Part(root=TextPart(text=followup_question))]))
    if response2.message:
        history.append(response2.message)
    await logger.ainfo('chat_flow turn 2', result=response2.text)

    response3 = await ai.generate(
        messages=[
            *history,
            Message(role=Role.USER, content=[Part(root=TextPart(text=final_question))]),
        ],
        system=system_prompt,
    )
    return response3.text
