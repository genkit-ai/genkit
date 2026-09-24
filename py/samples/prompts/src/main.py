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

"""Copy lives in prompts/*.prompt. Change the wording there, not here."""

from pathlib import Path

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit

ai = Genkit(
    plugins=[GoogleAI()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
    prompt_dir=Path(__file__).resolve().parent.parent / 'prompts',
)


def list_helper(data: object, *args: object, **kwargs: object) -> str:
    # recipe.prompt writes `{{list ingredients}}` — the helper is how a
    # bullet list gets into the prompt without Python string-joining.
    if not isinstance(data, list):
        return ''
    return '\n'.join(f'- {item}' for item in data)


ai.define_helper('list', list_helper)


class Ingredient(BaseModel):
    name: str
    quantity: str


class Recipe(BaseModel):
    title: str
    ingredients: list[Ingredient]
    steps: list[str]


# The .prompt file names this schema; define_schema is what wires it up.
ai.define_schema('Recipe', Recipe)


class ChefInput(BaseModel):
    food: str = 'banana bread'
    ingredients: list[str] | None = Field(default=None)


recipe = ai.prompt('recipe', input_schema=ChefInput, output_schema=Recipe)
# Same name, different file: prompts/recipe.robot.prompt. Swap the voice
# without touching the call site.
robot_recipe = ai.prompt('recipe', variant='robot', input_schema=ChefInput, output_schema=Recipe)
story = ai.prompt('story')


async def main() -> None:
    pantry = ChefInput(ingredients=['ripe bananas', 'walnuts'])
    print((await recipe(input=pantry)).output)
    print((await robot_recipe(input=ChefInput())).output)

    # story.prompt includes the _style.prompt partial. stream() is the
    # same object as generate_stream — chunks first, then the full text.
    streamed = story.stream(input={'subject': 'a brave little toaster'})
    async for chunk in streamed.stream:
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    await streamed.response


if __name__ == '__main__':
    ai.run_main(main())
