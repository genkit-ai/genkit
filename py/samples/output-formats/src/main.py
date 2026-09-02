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

"""output_schema is the Pydantic model you get back on response.output."""

from enum import Enum

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, TypeAdapter

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


class Sentiment(str, Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'


class Country(BaseModel):
    name: str
    capital: str
    population: int


class Book(BaseModel):
    title: str
    author: str


async def main() -> None:
    # Default: plain text.
    haiku = await ai.generate(prompt='Write a haiku about coding.')
    print(haiku.text)

    # Enum: the model picks one of the values.
    review = await ai.generate(
        prompt='Classify this review: This product broke after one day.',
        output_format='enum',
        output_schema=Sentiment,
    )
    print(review.output)

    # JSON: response.output is a Country, not a string you parse.
    country = await ai.generate(
        prompt='Give quick facts about Japan.',
        output_schema=Country,
    )
    print(country.output)

    # Array / jsonl take an items schema; TypeAdapter is how you get one
    # from list[T]. validate_python turns the raw list into Book models.
    books = await ai.generate(
        prompt='List 3 famous fantasy books.',
        output_format='array',
        output_schema=TypeAdapter(list[Book]).json_schema(),
    )
    print(TypeAdapter(list[Book]).validate_python(books.output))


if __name__ == '__main__':
    ai.run_main(main())
