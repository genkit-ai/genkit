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

"""Context caching - reuse a large source document across follow-up prompts. Requires GEMINI_API_KEY.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

import pathlib

import httpx

from genkit import Genkit, Message, Part, Role, TextPart
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit with Google GenAI plugin and default model
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')

DEFAULT_TEXT_FILE = 'https://www.gutenberg.org/cache/epub/74/pg74.txt'


async def _load_text(path: str) -> str:
    """Load text from either a URL or a local file."""
    if path.startswith('http'):
        async with httpx.AsyncClient() as client:
            response = await client.get(path)
            response.raise_for_status()
            return response.text
    return pathlib.Path(path).read_text(encoding='utf-8')


async def main() -> None:
    """Run context caching directly without intermediate flow wrappers."""
    try:
        source_text = await _load_text(DEFAULT_TEXT_FILE)

        # 2. Cache a large document in conversation history using metadata TTL
        cached_history = [
            Message(role=Role.USER, content=[Part(TextPart(text=source_text))]),
            Message(
                role=Role.MODEL,
                content=[Part(TextPart(text='Source document cached for follow-up questions.'))],
                metadata={'cache': {'ttl_seconds': 300}},
            ),
        ]

        # 3. Follow-up `generate()` calls reuse the cached tokens across turns automatically
        answer = await ai.generate(
            messages=cached_history,
            prompt='What do Tom Sawyer and Huck Finn value differently?',
        )
        short_answer = await ai.generate(
            messages=answer.messages,
            prompt='Now answer again in one sentence.',
        )

        print(f'Answer:\n{answer.text}\n\nOne sentence version:\n{short_answer.text}')
        # => Answer: Tom values status and societal play, whereas Huck prizes personal freedom and loyalty.
        # => One sentence version: Tom Sawyer seeks recognition in society, while Huck values independence.
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
