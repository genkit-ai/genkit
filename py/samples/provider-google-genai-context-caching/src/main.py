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

"""Context caching sample - Efficient handling of large contexts.

This sample demonstrates Genkit's context caching feature, which allows
expensive large context (like entire books) to be cached and reused across
multiple queries, significantly improving response time and cost.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Context Caching     │ Save expensive work to reuse later. Like           │
    │                     │ pre-cooking a big meal and reheating portions.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Large Context       │ Big documents (books, manuals, codebases).         │
    │                     │ Expensive to process every time.                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ TTL (Time-To-Live)  │ How long the cache stays valid. After 5 min,       │
    │                     │ it expires and must be re-created.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Cache Hit           │ Using cached context - FAST! Saves money           │
    │                     │ and time by reusing previous work.                 │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Cache Miss          │ No cache available - must process from scratch.    │
    │                     │ Slower, but creates cache for next time.           │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (Context Caching)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                   HOW CONTEXT CACHING SAVES TIME                        │
    │                                                                         │
    │    FIRST REQUEST (Cache Miss - creates cache)                           │
    │    ─────────────────────────────────────────                            │
    │    Query + Entire Book                                                  │
    │         │                                                               │
    │         │  (1) Send full context (slow, expensive)                      │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Gemini API     │   Processes book, caches internally              │
    │    │  (with cache)   │   Returns: response + cache_id                   │
    │    └────────┬────────┘                                                  │
    │             │   Time: ~10 seconds, Cost: $$$                            │
    │                                                                         │
    │    FOLLOW-UP REQUESTS (Cache Hit - uses cache)                          │
    │    ───────────────────────────────────────────                          │
    │    Query + cache_id (no book needed!)                                   │
    │         │                                                               │
    │         │  (2) Just send new question                                   │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  Gemini API     │   Uses cached context, answers quickly           │
    │    │  (cache hit!)   │                                                  │
    │    └────────┬────────┘                                                  │
    │             │   Time: ~1 second, Cost: $                                │
    │                                                                         │
    │    Savings: 90% faster, 90% cheaper for follow-up questions!            │
    └─────────────────────────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Plugin Initialization                   | `ai = Genkit(plugins=[GoogleAI()])` |
| Default Model Configuration             | `ai = Genkit(model=...)`            |
| Context Caching Config                  | `metadata={'cache': {'ttl_seconds': 300}}` |
| URL Content Fetching                    | `httpx.AsyncClient().get()`         |
| Message History Reuse                   | `messages=messages_history`         |
| Large Context Handling                  | Processing full book text           |

How Context Caching Works
=========================
1. First request: Full book is sent to model, context is cached (TTL: 5 min)
2. Follow-up requests: Use cached context, much faster response
3. Cache expires after TTL, next request re-caches

See README.md for testing instructions.
"""

import asyncio
import os
import pathlib

import httpx
from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import GoogleAI
from genkit.types import GenerationCommonConfig, Message, Part, Role, TextPart
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

logger = get_logger(__name__)


ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-pro-preview',
)


# Tom Sawyer is taken as a sample book here
DEFAULT_TEXT_FILE = 'https://www.gutenberg.org/cache/epub/74/pg74.txt'
DEFAULT_QUERY = "What are Huck Finn's views on society, and how do they contrast with Tom’s"


class BookContextInputSchema(BaseModel):
    """Input for book context flow."""

    query: str = Field(default=DEFAULT_QUERY, description='A question about the supplied text file')
    text_file_path: str = Field(
        default=DEFAULT_TEXT_FILE, description='Incoming reference to the text file (local or web)'
    )


@ai.flow(name='text_context_flow')
async def text_context_flow(_input: BookContextInputSchema) -> str:
    """Flow demonstrating context caching.

    Args:
        _input: The input schema.

    Returns:
        The generated response.
    """
    logger.info(f'Starting flow with file: {_input.text_file_path}')

    if _input.text_file_path.startswith('http'):
        async with httpx.AsyncClient() as client:
            res = await client.get(_input.text_file_path)
            res.raise_for_status()
            content_part = Part(root=TextPart(text=res.text))
    else:
        # Fallback for local text files
        with pathlib.Path(_input.text_file_path).open(encoding='utf-8') as text_file:
            content_part = Part(root=TextPart(text=text_file.read()))

    llm_response = await ai.generate(
        messages=[
            Message(
                role=Role.USER,
                content=[content_part],
            ),
            Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Here is some analysis based on the text provided.'))],
                metadata={
                    'cache': {
                        'ttl_seconds': 300,
                    }
                },
            ),
        ],
        config=GenerationCommonConfig(
            temperature=0.7,
            max_output_tokens=1000,
            top_k=50,
            top_p=0.9,
            stop_sequences=['END'],
        ),
        prompt=_input.query,
        return_tool_requests=False,
    )

    messages_history = llm_response.messages

    llm_response2 = await ai.generate(
        messages=messages_history,
        prompt=(
            'Rewrite the previous summary as if a pirate wrote it. '
            'Structure it exactly like this:\n'
            '### 1. Section Name\n'
            '*   **Key Concept:** Description...\n'
            'Keep it concise, use pirate slang, but maintain the helpful advice.'
        ),
        config=GenerationCommonConfig(
            version='gemini-3-flash-preview',
            temperature=0.7,
            max_output_tokens=1000,
            top_k=50,
            top_p=0.9,
            stop_sequences=['END'],
        ),
    )

    separator = '-' * 80
    return (
        f'{separator}\n'
        f'### Standard Analysis\n\n{llm_response.text}\n\n'
        f'{separator}\n'
        f'### Pirate Version\n\n{llm_response2.text}\n'
        f'{separator}'
    )


async def main() -> None:
    """Main entry point for the context caching sample - keep alive for Dev UI.

    This function demonstrates how to use context caching in Genkit for
    improved performance.
    """
    await logger.ainfo('Genkit server running. Press Ctrl+C to stop.')
    # Keep the process alive for Dev UI
    await asyncio.Event().wait()


if __name__ == '__main__':
    ai.run_main(main())
