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

"""Cohere AI hello sample — Cohere models with Genkit.

This sample demonstrates how to use Cohere's models with Genkit,
including Command A, Command A Reasoning, Command A Translate,
Command R+, Command R, and embedding models.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Cohere              │ AI company known for enterprise-grade language     │
    │                     │ models and excellent multilingual support.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Command A           │ Most capable model. Best for complex reasoning,    │
    │                     │ coding, and tool calling.                          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Command A Reasoning │ Reasoning-optimised variant. Excels at multi-step  │
    │                     │ agentic workflows and complex problem solving.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Command A Translate │ Translation-optimised variant. 23 languages with   │
    │                     │ excellent cross-lingual transfer quality.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Command R+          │ Strong general-purpose model. Great for RAG,       │
    │                     │ summarization, and multi-step tasks.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Command R           │ Fast and efficient. Great for everyday tasks       │
    │                     │ like chat, summarization, and simple coding.       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Calling        │ Let the model call your functions. Like giving     │
    │                     │ the AI a toolbox to help answer questions.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Structured Output   │ Get responses in a specific format (JSON).         │
    │                     │ Like filling out a form instead of free text.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Streaming           │ Get the response word-by-word as it's generated.   │
    │                     │ Feels faster, like watching someone type.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Embeddings          │ Convert text to numbers for finding similar        │
    │                     │ content. Powers semantic search and RAG.           │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet         |
|-----------------------------------------|-----------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[Cohere(...)])`    |
| Default Model Configuration             | `ai = Genkit(model=cohere_name(...))`  |
| Defining Flows                          | `@ai.flow()` decorator                  |
| Defining Tools                          | `@ai.tool()` decorator                  |
| Simple Generation (Prompt String)       | `generate_greeting`                      |
| System Prompt                           | `generate_with_system_prompt`            |
| Multi-turn Conversation                 | `generate_multi_turn_chat`               |
| Streaming Response                      | `generate_streaming_story`               |
| Generation with Config                  | `generate_with_config`                   |
| Tool Calling                            | `generate_weather`                       |
| Currency Conversion (Tool Calling)      | `convert_currency`                       |
| Structured Output (JSON)                | `generate_character`                     |
| Code Generation                         | `generate_code`                          |
| Reasoning (Command A Reasoning)         | `solve_reasoning_problem`                |
| Translation (Command A Translate)       | `translate_flow`                         |
| Streaming with Tools                    | `generate_streaming_with_tools`          |
| Embeddings (Text)                       | `embed_flow`                            |
"""

import asyncio
import os

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.cohere import Cohere, cohere_name
from samples.shared import (
    CharacterInput,
    CodeInput,
    ConfigInput,
    CurrencyExchangeInput,
    EmbedInput,
    GreetingInput,
    MultiTurnInput,
    ReasoningInput,
    RpgCharacter,
    StreamingToolInput,
    StreamInput,
    SystemPromptInput,
    TranslateInput,
    WeatherInput,
    chat_flow_logic,
    convert_currency as _convert_currency_tool,
    convert_currency_logic,
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
    translate_text_logic,
)

setup_sample()

if 'COHERE_API_KEY' not in os.environ and 'CO_API_KEY' not in os.environ:
    os.environ['COHERE_API_KEY'] = input('Please enter your COHERE_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[Cohere()],
    model=cohere_name('command-a-03-2025'),
)

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


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
async def generate_code(input: CodeInput) -> str:
    """Generate code using Cohere.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


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
async def generate_streaming_with_tools(
    input: StreamingToolInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Demonstrate streaming generation with tool calling."""
    return await generate_streaming_with_tools_logic(ai, input.location, ctx)


@ai.flow()
async def embed_flow(input: EmbedInput) -> list[float]:
    """Generate embeddings for text using Cohere's embed-v4.0 model.

    Args:
        input: Input with text to embed.

    Returns:
        The embedding vector (list of floats).
    """
    doc = Document.from_text(input.text)
    embeddings = await ai.embed(
        embedder=cohere_name('embed-v4.0'),
        content=doc,
    )
    return embeddings[0].embedding


@ai.flow()
async def solve_reasoning_problem(input: ReasoningInput) -> str:
    """Solve reasoning problems using Cohere's reasoning-optimised model.

    Uses ``command-a-reasoning-08-2025`` which excels at multi-step
    reasoning, agentic workflows, and complex problem solving.

    Args:
        input: Input with reasoning question to solve.

    Returns:
        The reasoning and answer.
    """
    return await solve_reasoning_problem_logic(ai, input.prompt, model=cohere_name('command-a-reasoning-08-2025'))


@ai.flow()
async def translate_flow(input: TranslateInput) -> str:
    """Translate text using Cohere's translation-optimised model.

    Uses ``command-a-translate-08-2025`` which supports 23 languages
    with excellent cross-lingual transfer quality.

    Args:
        input: Input with text and target language.

    Returns:
        The translated text.
    """
    return await translate_text_logic(
        ai, input.text, input.target_language, model=cohere_name('command-a-translate-08-2025')
    )


async def main() -> None:
    """Main entry point for the Cohere sample — keep alive for Dev UI."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
