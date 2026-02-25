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

"""Hugging Face hello sample - Access 1M+ models with Genkit.

This sample demonstrates how to use Hugging Face's Inference API with Genkit,
giving you access to millions of open-source models through a unified interface.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Hugging Face        │ The "GitHub for AI models". Hosts millions of      │
    │                     │ models you can use through their API.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference API       │ HF's API to run models. Like a free trial for      │
    │                     │ AI models with rate limits.                        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model ID            │ The model's address on HF, like "meta-llama/       │
    │                     │ Llama-3.3-70B-Instruct". Owner/model-name format.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference Providers │ 17+ partner services (Cerebras, Groq, Together)    │
    │                     │ accessible through one HF API. Pick the fastest!   │
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
| Plugin Initialization                   | `ai = Genkit(plugins=[HuggingFace()])` |
| Default Model Configuration             | `ai = Genkit(model=huggingface_name())`|
| Defining Flows                          | `@ai.flow()` decorator                  |
| Defining Tools                          | `@ai.tool()` decorator                  |
| Simple Generation (Prompt String)       | `generate_greeting`                     |
| System Prompt                           | `generate_with_system_prompt`           |
| Multi-turn Conversation                 | `generate_multi_turn_chat`              |
| Streaming Response                      | `generate_streaming_story`              |
| Different Models                        | `llama_flow`, `qwen_flow`               |
| Generation with Config                  | `generate_with_config`                  |
| Code Generation                         | `generate_code`                         |
| Multi-turn Chat                         | `chat_flow`                             |
| Tool Calling                            | `generate_weather`                      |
| Structured Output (JSON)                | `generate_character`                    |
| Streaming Structured Output             | `streaming_structured_output`           |
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.huggingface import HuggingFace, huggingface_name
from samples.shared import (
    CharacterInput,
    CodeInput,
    ConfigInput,
    GreetingInput,
    MultiTurnInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    WeatherInput,
    chat_flow_logic,
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

if 'HF_TOKEN' not in os.environ:
    os.environ['HF_TOKEN'] = input('Please enter your HF_TOKEN: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[HuggingFace(provider='auto')],
    model=huggingface_name('meta-llama/Llama-3.1-8B-Instruct'),
)


class ModelInput(BaseModel):
    """Input for model-specific flows."""

    prompt: str = Field(default='What is the meaning of life?', description='Prompt to send to the model')


ai.tool()(get_weather)


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
    """Demonstrate multi-turn conversations using the messages parameter."""
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
async def llama_flow(input: ModelInput) -> str:
    """Use Meta's Llama model for generation.

    Args:
        input: Input with prompt.

    Returns:
        Generated response from Llama.
    """
    response = await ai.generate(
        model=huggingface_name('meta-llama/Llama-3.1-8B-Instruct'),
        prompt=input.prompt,
    )
    return response.text


@ai.flow()
async def qwen_flow(input: ModelInput) -> str:
    """Use Alibaba's Qwen model for generation.

    Args:
        input: Input with prompt.

    Returns:
        Generated response from Qwen.
    """
    response = await ai.generate(
        model=huggingface_name('Qwen/Qwen2.5-7B-Instruct'),
        prompt=input.prompt,
    )
    return response.text


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
        system_prompt='You are a helpful AI tutor specializing in machine learning.',
        prompt1=(
            "Hi! I'm learning about neural networks. I find the concept of backpropagation particularly interesting."
        ),
        followup_question='What concept did I say I find interesting?',
        final_question='Can you explain that concept in simple terms?',
    )


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
        model=huggingface_name('meta-llama/Llama-3.1-8B-Instruct'),
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
    """Generate code using Hugging Face models.

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

    Args:
        input: Input with location for weather lookup.
        ctx: Action context for streaming chunks to the client.

    Returns:
        The complete generated text.
    """
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


async def main() -> None:
    """Main entry point for the Hugging Face sample - keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
