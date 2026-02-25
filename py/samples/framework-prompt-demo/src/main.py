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

"""Prompt demo sample - Loading and executing .prompt files.

This sample demonstrates Genkit's prompt management system, which allows you
to define prompts in separate .prompt files with templates, schemas, and config.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ .prompt File        │ A special file that holds your prompt template.    │
    │                     │ Like a recipe card the AI follows.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Template Variables  │ Placeholders like {{name}} that get filled in.     │
    │                     │ "Hello {{name}}" → "Hello Alice"                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Input Schema        │ Rules for what variables are allowed.              │
    │                     │ name: string, age: number, etc.                    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Output Schema       │ Rules for what the AI must return.                 │
    │                     │ Ensures structured, predictable responses.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Partials            │ Reusable prompt snippets (files starting with _).  │
    │                     │ Like a shared paragraph you paste into prompts.    │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Helpers             │ Custom functions usable in templates.              │
    │                     │ {{#list items}}...{{/list}}                        │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Prompt Management (Loading)             | `ai = Genkit(..., prompt_dir=...)`  |
| Prompt Execution                        | `recipe_prompt(input=...)`          |
| Custom Helpers                          | `ai.define_helper('list', ...)`     |
| Prompt Output Schema Validation         | `Recipe.model_validate(...)`        |
| Streaming Prompts (with partials)       | `story_prompt.stream()`             |
| Schema Registration                     | `ai.define_schema('Recipe', Recipe)`|

Testing Instructions
====================
1. Set ``GEMINI_API_KEY`` environment variable.
2. Run ``./run.sh`` from this sample directory.
3. Open the DevUI at http://localhost:4000.
4. Run ``chef_flow`` to generate a recipe (structured output).
5. Run ``robot_chef_flow`` to generate a robot-themed recipe (prompt variant).
6. Run ``tell_story`` to stream a story (uses partials + streaming).

See README.md for more details.
"""

import os
from pathlib import Path

from pydantic import BaseModel, Field

from genkit.ai import ActionKind, Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import GoogleAI
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

logger = get_logger(__name__)


current_dir = Path(__file__).resolve().parent
prompts_path = current_dir.parent / 'prompts'

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-3-flash-preview', prompt_dir=prompts_path)


def list_helper(data: object, *args: object, **kwargs: object) -> str:
    """Format a list of strings as bullet points.

    Args:
        data: List of items to format.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Formatted string or empty string if not a list.
    """
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


@ai.flow(name='chef_flow')
async def chef_flow(input: ChefInput) -> Recipe:
    """Generate a recipe for the given food.

    Args:
        input: Input containing the food item.

    Returns:
        A formatted recipe.

    Example:
        >>> await chef_flow(ChefInput(food='banana bread'))
        Recipe(title='Banana Bread', ...)
    """
    await logger.ainfo(f'chef_flow called with input: {input}')
    recipe_prompt = ai.prompt('recipe')

    response = await recipe_prompt(input={'food': input.food})
    # Ensure we return a Pydantic model as expected by the type hint and caller
    if not response.output:
        raise ValueError('Model did not return a recipe.')
    result = Recipe.model_validate(response.output)
    await logger.ainfo(f'chef_flow result: {result}')
    return result


@ai.flow(name='robot_chef_flow')
async def robot_chef_flow(input: ChefInput) -> Recipe:
    """Generate a robot-themed recipe for the given food.

    This flow demonstrates using prompt variants. The 'robot' variant
    of the recipe prompt generates recipes suitable for robots.

    Args:
        input: Input containing the food item.

    Returns:
        A formatted robot recipe.

    Example:
        >>> await robot_chef_flow(ChefInput(food='banana bread'))
        Recipe(title='Robotic Banana Bread', ...)
    """
    await logger.ainfo(f'robot_chef_flow called with input: {input}')
    robot_recipe_prompt = ai.prompt('recipe', variant='robot')

    response = await robot_recipe_prompt(input={'food': input.food})
    # Ensure we return a Pydantic model as expected by the type hint and caller
    if not response.output:
        raise ValueError('Model did not return a recipe.')
    result = Recipe.model_validate(response.output)
    await logger.ainfo(f'robot_chef_flow result: {result}')
    return result


class StoryInput(BaseModel):
    """Input for the story flow."""

    subject: str = Field(default='a brave little toaster', description='The subject of the story')
    personality: str | None = Field(default='courageous', description='Optional personality trait')


@ai.flow(name='tell_story')
async def tell_story(input: StoryInput, ctx: ActionRunContext) -> str:
    """Tell a story about the given subject.

    Args:
        input: Input containing the subject and optional personality.
        ctx: Action context for streaming.

    Returns:
        The generated story text.

    Example:
        >>> await tell_story(StoryInput(subject='toaster', personality='courageous'), ctx)
        'Once upon a time...'
    """
    await logger.ainfo(f'tell_story called with input: {input}')
    story_prompt = ai.prompt('story')
    result = story_prompt.stream(input={'subject': input.subject, 'personality': input.personality})

    full_text = ''
    async for chunk in result.stream:
        if chunk.text:
            ctx.send_chunk(chunk.text)
            full_text += chunk.text

    await logger.ainfo(f'tell_story completed, returning length: {len(full_text)}')
    return full_text


async def main() -> None:
    """Run the sample flows."""
    prompts = await ai.registry.resolve_actions_by_kind(ActionKind.PROMPT)
    executable_prompts = await ai.registry.resolve_actions_by_kind(ActionKind.EXECUTABLE_PROMPT)
    all_prompts = list(prompts.keys()) + list(executable_prompts.keys())

    await logger.ainfo('Registry Status', loaded_prompts=all_prompts)

    if not all_prompts:
        await logger.awarning('No prompts found! Check directory structure.')
        return

    # Chef Flow
    await logger.ainfo('--- Running Chef Flow ---')
    chef_result = await chef_flow(ChefInput(food='banana bread'))
    await logger.ainfo('Chef Flow Result', result=chef_result.model_dump())

    # Robot Chef Flow
    await logger.ainfo('--- Running Robot Chef Flow ---')
    robot_chef_result = await robot_chef_flow(ChefInput(food='banana bread'))
    await logger.ainfo('Robot Chef Flow Result', result=robot_chef_result.model_dump())

    # Tell Story Flow (Streaming)
    await logger.ainfo('--- Running Tell Story Flow ---')
    # To demonstrate streaming, we'll iterate over the streamer if calling directly like a flow would be consumed.
    story_stream, _ = tell_story.stream(StoryInput(subject='a brave little toaster', personality='courageous'))

    async for _chunk in story_stream:
        pass

    await logger.ainfo('Tell Story Flow Completed')


if __name__ == '__main__':
    ai.run_main(main())
