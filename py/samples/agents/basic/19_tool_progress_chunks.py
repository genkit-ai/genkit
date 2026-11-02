#!/usr/bin/env python3
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

"""Stream typed progress chunks from a tool while the model is thinking.

Tools can stream intermediate progress updates back to the caller while they run,
before returning their final output. These chunks arrive as partial tool_responses
on the stream and are not recorded in the conversation history, making them perfect
for UI loading states, progress bars, or "thinking" steps.
"""

import asyncio

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Genkit, ToolContext

ai = Genkit(plugins=[GoogleAI()])


class SearchProgress(BaseModel):
    step: str
    percent: int


@ai.tool(name='deepSearch')
async def deep_search(query: str, ctx: ToolContext[SearchProgress]) -> str:
    """Run a deep search that takes a while and emits progress updates."""
    ctx.send_chunk(SearchProgress(step='Querying primary index', percent=10))
    await asyncio.sleep(0.5)

    ctx.send_chunk(SearchProgress(step='Fetching documents', percent=40))
    await asyncio.sleep(0.5)

    ctx.send_chunk(SearchProgress(step='Extracting answers', percent=80))
    await asyncio.sleep(0.5)

    return f"Search results for '{query}': Found 42 related documents."


async def main() -> None:
    print('Starting deep search flow with tool progress streaming...')

    # Use generate_stream to receive the chunks as they happen
    stream = await ai.generate_stream(
        model=GoogleAI.gemini('gemini-flash-latest'),
        prompt='Run a deep search for quantum computing breakthroughs in 2024.',
        tools=[deep_search],
    )

    async for chunk in stream.stream:
        # Check if the chunk contains partial tool responses
        if chunk.tool_responses:
            for tr in chunk.tool_responses:
                # is_partial distinguishes streaming chunks from the final tool return value
                if tr.is_partial:
                    progress = SearchProgress.model_validate(tr.tool_response.output)
                    print(f"🔄 Progress: {progress.step} ({progress.percent}%)")
                else:
                    print(f"✅ Final Tool Result: {tr.tool_response.output}")
        elif chunk.text:
            print(chunk.text, end='', flush=True)

    print()


if __name__ == '__main__':
    ai.run_main(main())
