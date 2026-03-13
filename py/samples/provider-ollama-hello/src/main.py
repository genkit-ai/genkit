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

"""Ollama sample - Local LLM inference, tools, vision, and embeddings.

This sample demonstrates how to use Ollama for local AI with Genkit, covering
generation, streaming, tool calling, multimodal vision, structured output,
and embedding-based RAG — all without external API dependencies.

See README.md for setup and testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Ollama              │ Software that runs AI on YOUR computer. No cloud  │
    │                     │ needed - your data stays private!                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Local LLM           │ An AI that runs offline on your machine.          │
    │                     │ Like having a mini ChatGPT at home.               │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Gemma               │ Google's open-source model. Free to run locally.  │
    │                     │ Good for general tasks and coding help.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Mistral             │ Another open-source model. Good at reasoning      │
    │                     │ and supports tool calling.                        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Fathom-R1           │ Fractal AI's reasoning model (14B). Excels at    │
    │                     │ math & chain-of-thought. Based on DeepSeek-R1.   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ LLaVA               │ A vision model that understands images locally.   │
    │                     │ Describe photos without uploading them anywhere.  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Moondream           │ A tiny but capable vision model for object        │
    │                     │ detection. Great for bounding box tasks.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Embeddings          │ Convert text to numbers so AI can compare them.   │
    │                     │ "Pikachu" → [0.2, -0.5, 0.8, ...]                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ RAG                 │ Retrieval Augmented Generation: find relevant     │
    │                     │ context, then ask the LLM to answer from it.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Vector Similarity   │ Find similar items by comparing number arrays.   │
    │                     │ "electric mouse" finds Pikachu!                   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                                      | Flow / Function              |
|----------------------------------------------------------|------------------------------|
| Simple Generation (Prompt String)                        | ``generate_greeting``                   |
| System Prompts                                           | ``generate_with_system_prompt``            |
| Multi-turn Conversations (``messages``)                  | ``generate_multi_turn_chat``          |
| Streaming Generation                                     | ``generate_streaming_story``            |
| Structured Output (Simple)                               | ``structured_menu_suggestion``|
| Structured Output (Complex / Nested)                     | ``generate_character``       |
| Tool Calling                                             | ``calculate_gablorken``      |
| Tool Calling (Currency)                                  | ``convert_currency``        |
| Tool Calling (Weather)                                   | ``generate_weather``             |
| Multimodal Vision (Image Input)                          | ``describe_image``           |
| Object Detection (Bounding Boxes)                        | ``detect_objects``           |
| Chain-of-Thought Math Reasoning                          | ``solve_math_problem``       |
| Code Generation                                          | ``generate_code``                |
| Local Embeddings                                         | ``embed_pokemons``           |
| Vector Similarity Search                                 | ``pokedex``                  |
"""

from math import sqrt

from pydantic import BaseModel, Field

from genkit.ai import Genkit, Output
from genkit.blocks.model import GenerateResponseWrapper
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.core.typing import Media, MediaPart, Part, TextPart
from genkit.plugins.ollama import Ollama, ollama_name
from genkit.plugins.ollama.embedders import EmbeddingDefinition
from genkit.plugins.ollama.models import ModelDefinition
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
    generate_with_system_prompt_logic,
    get_weather,
    setup_sample,
)

setup_sample()

logger = get_logger(__name__)

# Pull models with: ollama pull <model>
GEMMA_MODEL = 'gemma3:latest'

# gemma2:latest does NOT support tool calling — use mistral-nemo instead.
MISTRAL_MODEL = 'mistral-nemo:latest'

# Vision models: llava and moondream support image understanding locally.
LLAVA_MODEL = 'llava:latest'
MOONDREAM_MODEL = 'moondream:v2'

# Reasoning model: Fractal AI's Fathom-R1-14B for math & chain-of-thought.
# Pulled from HuggingFace via: ollama pull hf.co/Mungert/Fathom-R1-14B-GGUF
FATHOM_MODEL = 'hf.co/Mungert/Fathom-R1-14B-GGUF'

# Embedding model for RAG.
EMBEDDER_MODEL = 'nomic-embed-text'

ai = Genkit(
    plugins=[
        Ollama(
            models=[
                ModelDefinition(name=GEMMA_MODEL),
                ModelDefinition(name=MISTRAL_MODEL),
                ModelDefinition(name=LLAVA_MODEL),
                ModelDefinition(name=MOONDREAM_MODEL),
                ModelDefinition(name=FATHOM_MODEL),
            ],
            embedders=[
                EmbeddingDefinition(name=EMBEDDER_MODEL, dimensions=512),
            ],
        )
    ],
    model=ollama_name(GEMMA_MODEL),
)

ai.tool()(get_weather)
ai.tool()(_convert_currency_tool)


class GablorkenInput(BaseModel):
    """Input model for the gablorken tool function.

    Attributes:
        value: The value to calculate gablorken for.
    """

    value: int = Field(description='value to calculate gablorken for')


class GablorkenOutputSchema(BaseModel):
    """Gablorken output schema.

    Args:
        result: The result of the gablorken.
    """

    result: int


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


class MenuSuggestionInput(BaseModel):
    """Input for structured menu suggestion flow."""

    theme: str = Field(default='pirate', description='Restaurant theme (e.g., pirate, space, medieval)')


class GablorkenFlowInput(BaseModel):
    """Input for gablorken calculation flow."""

    value: int = Field(default=33, description='Value to calculate gablorken for')


class PokemonInfo(BaseModel):
    """Information about a Pokemon for the embedding demo."""

    name: str
    description: str
    embedding: list[float] | None = None


class PokemonFlowInput(BaseModel):
    """Input for Pokemon RAG flow."""

    question: str = Field(default='Who is the best water pokemon?', description='Question about Pokemon')


@ai.tool()
def gablorken_tool(input: GablorkenInput) -> int:
    """Calculate a gablorken."""
    return input.value * 3 - 5


@ai.flow()
async def generate_greeting(input: GreetingInput) -> str:
    """Generate a simple greeting.

    Args:
        input: Input with name to greet.

    Returns:
        The greeting message.
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
async def generate_streaming_story(
    input: StreamInput,
    ctx: ActionRunContext | None = None,
) -> str:
    """Generate a streaming story response.

    Args:
        input: Input with name for streaming.
        ctx: the context of the tool

    Returns:
        The complete response text.
    """
    return await generate_streaming_story_logic(ai, input.name, ctx)


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
async def generate_code(input: CodeInput) -> str:
    """Generate code using local Ollama models.

    Args:
        input: Input with coding task description.

    Returns:
        Generated code.
    """
    return await generate_code_logic(ai, input.task)


@ai.flow()
async def calculate_gablorken(input: GablorkenFlowInput) -> str:
    """Use the gablorken_tool to calculate a gablorken value.

    Args:
        input: Input with value for gablorken calculation.

    Returns:
        The gablorken result.
    """
    response = await ai.generate(
        prompt=f'Use the gablorken_tool to calculate the gablorken of {input.value}',
        model=ollama_name(MISTRAL_MODEL),
        tools=['gablorken_tool'],
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
    return await convert_currency_logic(ai, input, model=ollama_name(MISTRAL_MODEL))


@ai.flow()
async def generate_weather(input: WeatherInput) -> str:
    """Get weather information using tool calling.

    Args:
        input: Input with location for weather.

    Returns:
        Weather information for the location.
    """
    return await generate_weather_logic(ai, input, model=ollama_name(MISTRAL_MODEL))


@ai.flow()
async def describe_image(input: ImageDescribeInput) -> str:
    """Describe an image using a local vision model (llava).

    Uses the llava model for local, private image understanding.
    Requires: ``ollama pull llava`` before running.

    The Ollama plugin handles MediaPart by downloading the image URL
    client-side and converting it to an Ollama Image object.

    Args:
        input: Input with image URL to describe.

    Returns:
        A textual description of the image.
    """
    return await describe_image_logic(ai, input.image_url, model=ollama_name(LLAVA_MODEL))


class ObjectDetectionInput(BaseModel):
    """Input for object detection with bounding boxes."""

    image_url: str = Field(
        default='https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png',
        description='URL of the image to detect objects in',
    )
    prompt: str = Field(
        default='Detect the objects in this image and return bounding boxes.',
        description='Detection prompt',
    )


@ai.flow()
async def detect_objects(input: ObjectDetectionInput) -> str:
    """Detect objects in an image using moondream:v2.

    Uses the moondream vision model for lightweight, local object detection.
    Moondream is a tiny but capable vision model that excels at describing
    image content and returning bounding box coordinates.

    Requires: ``ollama pull moondream:v2`` before running.

    Args:
        input: Input with image URL and detection prompt.

    Returns:
        Detection results with bounding box coordinates.
    """
    response = await ai.generate(
        model=ollama_name(MOONDREAM_MODEL),
        prompt=[
            Part(root=TextPart(text=input.prompt)),
            Part(root=MediaPart(media=Media(url=input.image_url, content_type='image/png'))),
        ],
    )
    return response.text


class ReasoningInput(BaseModel):
    """Input for the math reasoning flow.

    Attributes:
        problem: A math problem to solve with chain-of-thought reasoning.
    """

    problem: str = Field(
        default='What is the sum of the first 50 prime numbers?',
        description='A math problem to solve step-by-step',
    )


@ai.flow()
async def solve_math_problem(input: ReasoningInput) -> str:
    """Solve a math problem using Fathom-R1-14B with chain-of-thought reasoning.

    Uses Fractal AI's Fathom-R1-14B model, a 14B-parameter reasoning model
    fine-tuned from DeepSeek-R1-Distilled-Qwen-14B. It excels at olympiad-level
    mathematical reasoning and shows its work through chain-of-thought.

    Requires: ``ollama pull hf.co/Mungert/Fathom-R1-14B-GGUF``

    See: https://huggingface.co/FractalAIResearch/Fathom-R1-14B

    Args:
        input: Input with a math problem to solve.

    Returns:
        The model's step-by-step solution.
    """
    response = await ai.generate(
        model=ollama_name(FATHOM_MODEL),
        system='You are a math tutor. Show your reasoning step by step before giving the final answer.',
        prompt=input.problem,
    )
    return response.text


pokemon_list = [
    PokemonInfo(
        name='Pikachu',
        description='An Electric-type Pokemon known for its strong electric attacks.',
    ),
    PokemonInfo(
        name='Charmander',
        description='A Fire-type Pokemon that evolves into the powerful Charizard.',
    ),
    PokemonInfo(
        name='Bulbasaur',
        description='A Grass/Poison-type Pokemon that grows into a powerful Venusaur.',
    ),
    PokemonInfo(
        name='Squirtle',
        description='A Water-type Pokemon known for its water-based attacks and high defense.',
    ),
    PokemonInfo(
        name='Jigglypuff',
        description='A Normal/Fairy-type Pokemon with a hypnotic singing ability.',
    ),
]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Calculate the cosine distance between two vectors.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine distance (0 = identical, 2 = opposite).
    """
    if len(a) != len(b):
        raise ValueError('Input vectors must have the same length')
    dot_product = sum(ai_val * bi_val for ai_val, bi_val in zip(a, b, strict=True))
    magnitude_a = sqrt(sum(ai_val * ai_val for ai_val in a))
    magnitude_b = sqrt(sum(bi_val * bi_val for bi_val in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError('Invalid input: zero vector')

    return 1 - (dot_product / (magnitude_a * magnitude_b))


async def embed_pokemons() -> None:
    """Embed all Pokemon descriptions using the local embedding model."""
    embeddings = await ai.embed_many(
        embedder=ollama_name(EMBEDDER_MODEL),
        content=[pokemon.description for pokemon in pokemon_list],
    )
    for pokemon, embedding in zip(pokemon_list, embeddings, strict=True):
        pokemon.embedding = embedding.embedding


def find_nearest_pokemons(input_embedding: list[float], top_n: int = 3) -> list[PokemonInfo]:
    """Find the nearest Pokemon by cosine similarity.

    Args:
        input_embedding: The query embedding.
        top_n: Number of results to return.

    Returns:
        The most similar Pokemon.
    """
    if any(pokemon.embedding is None for pokemon in pokemon_list):
        raise AttributeError('Some Pokemon are not yet embedded')

    pokemon_distances = []
    for pokemon in pokemon_list:
        if pokemon.embedding is not None:
            distance = cosine_distance(input_embedding, pokemon.embedding)
            pokemon_distances.append((distance, pokemon))

    pokemon_distances.sort(key=lambda item: item[0])
    return [pokemon for _distance, pokemon in pokemon_distances[:top_n]]


async def generate_rag_response(question: str) -> GenerateResponseWrapper:
    """Generate a RAG response: embed the question, find context, generate.

    Args:
        question: The user's question.

    Returns:
        The model's response with retrieved context.
    """
    input_embedding = await ai.embed(
        embedder=ollama_name(EMBEDDER_MODEL),
        content=question,
    )
    nearest_pokemon = find_nearest_pokemons(input_embedding[0].embedding)
    pokemons_context = '\n'.join(f'{pokemon.name}: {pokemon.description}' for pokemon in nearest_pokemon)

    return await ai.generate(
        model=ollama_name(GEMMA_MODEL),
        prompt=f'Given the following context on Pokemon:\n{pokemons_context}\n\nQuestion: {question}\n\nAnswer:',
    )


@ai.flow(name='Pokedex')
async def pokedex(input: PokemonFlowInput) -> str:
    """Answer Pokemon questions using local RAG (embed → retrieve → generate).

    Args:
        input: A question about Pokemon.

    Returns:
        The generated answer.
    """
    await embed_pokemons()
    response = await generate_rag_response(question=input.question)
    if not response.message or not response.message.content:
        raise ValueError('No message content returned from model')
    text = response.message.content[0].root.text
    return str(text) if text is not None else ''


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
    return await generate_streaming_with_tools_logic(ai, input.location, ctx, model=ollama_name(MISTRAL_MODEL))


async def main() -> None:
    """Main function."""
    await logger.ainfo(await generate_greeting(GreetingInput(name='John Doe')))
    await logger.ainfo(str(await structured_menu_suggestion(MenuSuggestionInput(theme='pirate'))))
    await logger.ainfo(await calculate_gablorken(GablorkenFlowInput(value=33)))
    await logger.ainfo(await generate_weather(WeatherInput(location='San Francisco')))


if __name__ == '__main__':
    ai.run_main(main())
