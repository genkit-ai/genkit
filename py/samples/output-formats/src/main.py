#!/usr/bin/env python3
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

"""Structured output formats — receive validated Pydantic objects, enums, or arrays directly from models.

Requires GEMINI_API_KEY. Run:
    uv run src/main.py
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, TypeAdapter

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit with Google GenAI plugin
ai = Genkit(plugins=[GoogleAI(api_version='v1alpha')], model='googleai/gemini-flash-latest')


# 2. Define structured schemas (JSON object, Enum, and Array shapes)
class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int


class Sentiment(str, Enum):
    POSITIVE = 'POSITIVE'
    NEGATIVE = 'NEGATIVE'
    NEUTRAL = 'NEUTRAL'


class Book(BaseModel):
    title: str
    author: str


async def main() -> None:
    """Run each output-format generation directly without intermediate flow wrappers."""
    try:
        # --- 1. Structured JSON Object Output (`output_format='json'`) ---
        country_res = await ai.generate(
            prompt='Give quick facts about Japan.',
            output_format='json',
            output_schema=CountryInfo,
        )
        print(country_res.output)
        # => CountryInfo(name='Japan', capital='Tokyo', population=125100000)

        # --- 2. Constrained Enum Output (`output_format='enum'`) ---
        sentiment_res = await ai.generate(
            prompt='Classify this review: Best purchase ever!',
            output_format='enum',
            output_schema=Sentiment,
        )
        print(sentiment_res.output)
        # => Sentiment.POSITIVE

        # --- 3. Structured JSON Array Output (`output_format='array'`) ---
        books_res = await ai.generate(
            prompt='List 2 famous Sci-Fi books.',
            output_format='array',
            output_schema=TypeAdapter(list[Book]).json_schema(),
        )
        print(books_res.output)
        # => [{'title': 'Dune', 'author': 'Frank Herbert'}, {'title': 'Neuromancer', 'author': 'William Gibson'}]
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
