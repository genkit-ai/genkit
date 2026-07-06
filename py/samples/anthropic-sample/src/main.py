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

"""Anthropic's Claude Opus models generation samples. Requires ANTHROPIC_API_KEY.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

from pydantic import BaseModel

from genkit import Genkit
from genkit.plugins.anthropic import Anthropic

# 1. Initialize Genkit with the Anthropic plugin and default model
ai = Genkit(plugins=[Anthropic()], model='anthropic/claude-opus-4-8')


# 2. Define structured schema for JSON object generation
class Cat(BaseModel):
    name: str
    breed: str
    age: int
    personality: str


async def main() -> None:
    """Run generation directly without intermediate flow wrappers."""
    try:
        # --- 1. Plain-Text Generation ---
        haiku_res = await ai.generate(
            model='anthropic/claude-opus-4-8',
            prompt='Write a haiku about coding.',
        )
        print(haiku_res.text)
        # => Lines of logic flow,
        #    Silently building the world,
        #    Bugs fade in the light.

        # --- 2. Structured JSON Object Generation ---
        cat_res = await ai.generate(
            model='anthropic/claude-opus-4-8',
            prompt='Invent a cat named Mittens.',
            output_format='json',
            output_schema=Cat,
        )
        print(cat_res.output)
        # => Cat(name='Mittens', breed='Russian Blue', age=3, personality='Curious and calm')
    except Exception as error:
        print(f'Set ANTHROPIC_API_KEY to a valid value before running this sample.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
