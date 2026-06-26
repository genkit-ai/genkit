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

"""Anthropic's Claude Opus models generation samples."""

from genkit_anthropic import Anthropic
from pydantic import BaseModel, Field

from genkit import Genkit

ai = Genkit(plugins=[Anthropic()], model='anthropic/claude-opus-4-8')


class TopicInput(BaseModel):
    """Input for a plain-text generation."""

    topic: str = Field(default='coding', description='Topic for the haiku')


class CatInput(BaseModel):
    """Input for a structured generation."""

    name: str = Field(default='Mittens', description='Name of the cat to invent')


class Cat(BaseModel):
    """Structured cat profile — proves output=['text','json'] + constrained."""

    name: str
    breed: str
    age: int
    personality: str


# --- claude-opus-4-7 -------------------------------------------------------


@ai.flow()
async def haiku_opus_4_7(data: TopicInput) -> str:
    """Plain-text generate against claude-opus-4-7."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-7',
        prompt=f'Write a haiku about {data.topic}.',
    )
    return response.text


@ai.flow()
async def cat_opus_4_7(data: CatInput) -> Cat:
    """Structured/JSON generate against claude-opus-4-7."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-7',
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
    )
    return response.output


# --- claude-opus-4-8 -------------------------------------------------------


@ai.flow()
async def haiku_opus_4_8(data: TopicInput) -> str:
    """Plain-text generate against claude-opus-4-8."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-8',
        prompt=f'Write a haiku about {data.topic}.',
    )
    return response.text


@ai.flow()
async def cat_opus_4_8(data: CatInput) -> Cat:
    """Structured/JSON generate against claude-opus-4-8."""
    response = await ai.generate(
        model='anthropic/claude-opus-4-8',
        prompt=f'Invent a cat named {data.name}.',
        output_format='json',
        output_schema=Cat,
    )
    return response.output


async def main() -> None:
    """Run each flow once from the CLI."""
    try:
        print(await haiku_opus_4_7(TopicInput()))  # noqa: T201
        print(await cat_opus_4_7(CatInput()))  # noqa: T201
        print(await haiku_opus_4_8(TopicInput()))  # noqa: T201
        print(await cat_opus_4_8(CatInput()))  # noqa: T201
    except Exception as error:
        print(f'Set ANTHROPIC_API_KEY to a valid value before running this sample.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
