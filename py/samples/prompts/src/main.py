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

"""Dotprompt templates — load `.prompt` files from disk, format with Handlebars, and receive typed output.

Requires GEMINI_API_KEY. Run:
    uv run src/main.py
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit with prompt directory pointing to `.prompt` files
ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-flash-latest',
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)


# 2. Define custom Handlebars helper and bind schemas for `.prompt` templates
def list_helper(data: object, *args: object, **kwargs: object) -> str:
    """Format a list as bullet points inside Handlebars templates."""
    if not isinstance(data, list):
        return ''
    return '\n'.join(f'- {item}' for item in data)


ai.define_helper('list', list_helper)


class Ingredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str = Field(..., description='recipe title')
    ingredients: list[Ingredient]
    steps: list[str] = Field(..., description='the steps required to complete the recipe')


ai.define_schema('Recipe', Recipe)


async def main() -> None:
    """Run dotprompt rendering and generation directly without intermediate flow wrappers."""
    try:
        # --- 1. Call `.prompt` template directly and receive typed Pydantic output ---
        recipe_res = await ai.prompt('recipe')(input={'food': 'banana bread'})
        if recipe_res.output:
            print(Recipe.model_validate(recipe_res.output))
        # => Recipe(title='Classic Banana Bread', ingredients=[Ingredient(name='ripe bananas', quantity='3')])

        # --- 2. Call the `robot` variant (`recipe.robot.prompt`) of the same prompt ---
        robot_res = await ai.prompt('recipe', variant='robot')(input={'food': 'banana bread'})
        if robot_res.output:
            print(Recipe.model_validate(robot_res.output))
        # => Recipe(title='UNIT-7 BANANA SYNTHESIS', ingredients=[Ingredient(name='BIOLOGICAL BANANA', quantity='3X')])

        # --- 3. Stream dotprompt results live chunk by chunk ---
        stream_res = ai.prompt('story').stream(input={'subject': 'a brave little toaster', 'personality': 'courageous'})
        async for chunk in stream_res.stream:
            if chunk.text:
                print(chunk.text, end='', flush=True)
        print()
        # => Streams chunks: "In the quiet kitchen...", "...little Toaster glowed...", "...ready for breakfast!"
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
