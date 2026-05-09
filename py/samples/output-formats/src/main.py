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

"""Output formats - text, enum, JSON object, array, and JSONL."""

from enum import Enum

from pydantic import BaseModel, Field, TypeAdapter

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI(api_version='v1alpha')], model='googleai/gemini-3-flash-preview')


class HaikuInput(BaseModel):
    """Input for plain text output."""

    topic: str = Field(default='coding', description='Topic for the haiku')


class ReviewInput(BaseModel):
    """Input for enum output."""

    review: str = Field(default='This product broke after one day.', description='Review to classify')


class Sentiment(str, Enum):
    """Allowed sentiment labels."""

    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'


class CountryInfo(BaseModel):
    """Structured country info."""

    name: str
    capital: str
    population: int


class CountryInput(BaseModel):
    """Input for JSON object output."""

    country: str = Field(default='Japan', description='Country to describe')


class Book(BaseModel):
    """Book recommendation schema."""

    title: str
    author: str


class GenreInput(BaseModel):
    """Input for array output."""

    genre: str = Field(default='Fantasy', description='Genre to recommend')


class Character(BaseModel):
    """Story character schema."""

    name: str
    role: str


class ThemeInput(BaseModel):
    """Input for JSONL output."""

    theme: str = Field(default='space opera', description='Story theme')


BOOK_LIST_SCHEMA = TypeAdapter(list[Book]).json_schema()
CHARACTER_LIST_SCHEMA = TypeAdapter(list[Character]).json_schema()


@ai.flow()
async def generate_haiku_text(input: HaikuInput) -> str:
    """Return plain text."""

    response = await ai.generate(prompt=f'Write a haiku about {input.topic}.', output_format='text')
    return response.text


@ai.flow()
async def classify_sentiment_enum(input: ReviewInput) -> Sentiment:
    """Return one value from a fixed set."""

    response = await ai.generate(
        prompt=f'Classify this review: {input.review}',
        output_format='enum',
        output_schema=Sentiment,
    )
    return response.output


@ai.flow()
async def get_country_info_json(input: CountryInput) -> CountryInfo:
    """Return one JSON object."""

    response = await ai.generate(
        prompt=f'Give quick facts about {input.country}.',
        output_format='json',
        output_schema=CountryInfo,
    )
    return response.output


@ai.flow()
async def recommend_books_array(input: GenreInput):
    """Return an array of objects."""

    response = await ai.generate(
        prompt=f'List 3 famous {input.genre} books.',
        output_format='array',
        output_schema=BOOK_LIST_SCHEMA,
    )
    return response.output


@ai.flow()
async def create_story_characters_jsonl(input: ThemeInput):
    """Return newline-delimited JSON objects."""

    response = await ai.generate(
        prompt=f'Generate 3 characters for a {input.theme} story.',
        output_format='jsonl',
        output_schema=CHARACTER_LIST_SCHEMA,
    )
    return response.output


async def main() -> None:
    """Run each output-format example once."""
    try:
        print(await generate_haiku_text(HaikuInput()))  # noqa: T201
        print(await classify_sentiment_enum(ReviewInput()))  # noqa: T201
        print(await get_country_info_json(CountryInput()))  # noqa: T201
        print(await recommend_books_array(GenreInput()))  # noqa: T201
        print(await create_story_characters_jsonl(ThemeInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
