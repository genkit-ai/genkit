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

"""Format demo sample - Output format options in Genkit.

This sample demonstrates the various output formats available in Genkit for
structured and unstructured AI-generated content.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Output Format       │ What shape the AI's answer takes. Plain text,      │
    │                     │ structured JSON, or something else.                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ text                │ Free-form text. Like a normal conversation.        │
    │                     │ "Once upon a time..."                              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ json                │ Structured data. AI returns objects you can        │
    │                     │ use in code: {name: "Bob", age: 25}                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ array               │ A list of items. ["item1", "item2", "item3"]       │
    │                     │ Great for generating multiple things.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ enum                │ One choice from a list. "What color?" → "blue"     │
    │                     │ Perfect for classification tasks.                  │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ jsonl               │ Multiple JSON objects, one per line.               │
    │                     │ For streaming or large datasets.                   │
    └─────────────────────┴────────────────────────────────────────────────────┘

Output Formats
==============
| Format   | Description                              | Use Case                    |
|----------|------------------------------------------|-----------------------------|
| `text`   | Raw text output (default)                | Free-form responses         |
| `json`   | Structured JSON object output            | API responses, data objects |
| `array`  | JSON array of items                      | Lists, collections          |
| `enum`   | Single value from a predefined set       | Classification, categories  |
| `jsonl`  | Newline-delimited JSON (streaming)       | Large datasets, streaming   |
"""

import asyncio
import os
from typing import Any, cast

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.logging import get_logger
from genkit.core.typing import OutputConfig
from genkit.plugins.google_genai import GoogleAI
from samples.shared.logging import setup_sample

setup_sample()

logger = get_logger(__name__)

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')


ai = Genkit(
    plugins=[GoogleAI(api_version='v1alpha')],
    model='googleai/gemini-3-flash-preview',
)


# --- 1. Text Format ---
class HaikuInput(BaseModel):
    """Input for generating a haiku."""

    topic: str = Field(default='coding', description='The topic for the haiku')


class Book(BaseModel):
    """A book with title and author."""

    title: str
    author: str


class Character(BaseModel):
    """A character in a story."""

    name: str
    role: str


class ClassifySentimentInput(BaseModel):
    """Input for sentiment classification."""

    review: str = Field(
        default='This product is terrible and broke after one day',
        description='Product review text to classify',
    )


class CountryInfo(BaseModel):
    """Information about a country."""

    name: str
    capital: str
    population: int


class CountryInfoInput(BaseModel):
    """Input for getting country information."""

    country: str = Field(default='Japan', description='Name of the country')


class CreateStoryCharactersInput(BaseModel):
    """Input for creating story characters."""

    theme: str = Field(default='Space Opera', description='Theme or genre of the story')


class RecommendBooksInput(BaseModel):
    """Input for book recommendations."""

    genre: str = Field(default='Fantasy', description='Book genre to recommend')


@ai.flow()
async def classify_sentiment_enum(input: ClassifySentimentInput) -> str:
    """Classify the sentiment of a product review.

    Uses the 'enum' format which constrains the model to output exactly one value
    from a predefined list. This is perfect for classification tasks.
    The response is cleaned (quotes stripped) automatically.

    Example inputs and outputs:
        "This product is terrible!" -> "NEGATIVE"
        "Works as expected, nothing special" -> "NEUTRAL"
        "Best purchase ever! Highly recommend!" -> "POSITIVE"
    """
    response = await ai.generate(
        prompt=f'Classify the sentiment of this review: "{input.review}"',
        output=OutputConfig(
            format='enum',
            schema={
                'type': 'string',
                'enum': ['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
            },
        ),
    )
    return cast(str, response.output)


@ai.flow()
async def create_story_characters_jsonl(input: CreateStoryCharactersInput) -> list[dict[str, object]]:
    """Create characters for a story with a given theme.

    Uses the 'jsonl' format which outputs newline-delimited JSON objects.
    This is particularly useful for streaming scenarios where you want
    to process each object as soon as it's generated, without waiting
    for the full array to complete.

    Each line is a complete JSON object that can be parsed independently.
    The schema must be 'array' type with 'object' items.

    Example output (raw):
        {"name": "Captain Zara", "role": "Ship Commander"}
        {"name": "Dr. Vex", "role": "Science Officer"}
        {"name": "K-7", "role": "Android Navigator"}
        {"name": "Luna", "role": "Engineer"}
        {"name": "The Broker", "role": "Mysterious Informant"}
    """
    response = await ai.generate(
        prompt=f'Generate 5 characters for a {input.theme} story',
        output=OutputConfig(
            format='jsonl',
            schema={
                'type': 'array',
                'items': Character.model_json_schema(),
            },
        ),
    )
    return cast(list[Any], response.output)


@ai.flow()
async def generate_haiku_text(input: HaikuInput) -> str:
    """Generate a haiku about a given topic.

    Uses the 'text' format which returns the model's response as a plain string.
    This is the simplest format, useful when you just need unstructured text.

    Example output:
        "Lines of code cascade,
        Bugs hide in the syntax maze,
        Debug brings the dawn."
    """
    response = await ai.generate(
        prompt=f'Write a haiku about {input.topic}',
        output=OutputConfig(format='text'),
    )
    return response.text


@ai.flow()
async def get_country_info_json(input: CountryInfoInput) -> dict[str, Any]:
    """Get structured information about a country.

    Uses the 'json' format which parses the model's response as a JSON object.
    You provide a schema (as a JSON Schema dict) to define the expected structure.
    Genkit uses constrained decoding when supported by the model.

    Example output:
        {"name": "Japan", "capital": "Tokyo", "population": 125000000}
    """
    response = await ai.generate(
        prompt=f'Give me information about {input.country}',
        output=OutputConfig(format='json', schema=CountryInfo.model_json_schema()),
    )
    return cast(dict[str, Any], response.output)


@ai.flow()
async def recommend_books_array(input: RecommendBooksInput) -> list[dict[str, object]]:
    """Recommend famous books in a given genre.

    Uses the 'array' format which parses the model's response as a JSON array of objects.
    This is useful for generating lists of structured items.
    The schema must be of type 'array' with 'items' defining the object structure.

    Example output:
        [
            {"title": "The Lord of the Rings", "author": "J.R.R. Tolkien"},
            {"title": "A Game of Thrones", "author": "George R.R. Martin"},
            {"title": "The Name of the Wind", "author": "Patrick Rothfuss"}
        ]
    """
    response = await ai.generate(
        prompt=f'List 3 famous {input.genre} books',
        output=OutputConfig(
            format='array',
            schema={
                'type': 'array',
                'items': Book.model_json_schema(),
            },
        ),
    )
    return cast(list[Any], response.output)


async def main() -> None:
    """Main function."""
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
