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

"""Prompts - load `.prompt` files, helpers, variants, and streaming."""

from pathlib import Path

from pydantic import BaseModel, Field

from genkit import Genkit
from genkit._core._action import ActionRunContext
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-flash-preview',
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)


def list_helper(data: object, *args: object, **kwargs: object) -> str:
    """Format a list as bullet points for prompt templates."""

    if not isinstance(data, list):
        return ''
    return '\n'.join(f'- {item}' for item in data)


ai.define_helper('list', list_helper)


class Ingredient(BaseModel):
    """An ingredient in a recipe."""

    name: str
    quantity: str


class Recipe(BaseModel):
    """A recipe."""

    title: str = Field(..., description='recipe title')
    ingredients: list[Ingredient]
    steps: list[str] = Field(..., description='the steps required to complete the recipe')


ai.define_schema('Recipe', Recipe)


class ChefInput(BaseModel):
    """Input for the chef flow."""

    food: str = Field(default='banana bread', description='The food to create a recipe for')


@ai.flow(name='generate_recipe')
async def chef_flow(input: ChefInput) -> Recipe:
    """Call the default `recipe.prompt` template."""

    response = await ai.prompt('recipe')(input={'food': input.food})
    if not response.output:
        raise ValueError('Model did not return a recipe.')
    return Recipe.model_validate(response.output)


@ai.flow(name='generate_robot_recipe')
async def robot_chef_flow(input: ChefInput) -> Recipe:
    """Call the `robot` variant of the same prompt."""

    response = await ai.prompt('recipe', variant='robot')(input={'food': input.food})
    if not response.output:
        raise ValueError('Model did not return a recipe.')
    return Recipe.model_validate(response.output)


class StoryInput(BaseModel):
    """Input for the story flow."""

    subject: str = Field(default='a brave little toaster', description='The subject of the story')
    personality: str | None = Field(default='courageous', description='Optional personality trait')


@ai.flow(name='tell_story')
async def tell_story(input: StoryInput, ctx: ActionRunContext) -> str:
    """Stream a prompt result chunk by chunk."""

    result = ai.prompt('story').stream(input={'subject': input.subject, 'personality': input.personality})
    full_text = ''
    async for chunk in result.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)
            full_text += chunk.text
    return full_text


async def main() -> None:
    """Run the prompt demos once."""
    try:
        print(await chef_flow(ChefInput()))  # noqa: T201
        print(await robot_chef_flow(ChefInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
