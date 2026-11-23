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

"""Stream typed progress chunks from a tool through an agent turn.

Annotate a tool's context parameter as ``ToolContext[ChunkT]`` and call
``ctx.send_chunk(...)`` while the tool runs. During ``chat.send_stream(...)``,
each update arrives on ``AgentChunk.tool_responses`` with ``part.is_partial``
set to ``True``, while persisted session history keeps only the final tool
response. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Role, ToolContext
from genkit.exp import Genkit
from genkit.exp.agent import InMemorySessionStore


class PrepProgress(BaseModel):
    stage_name: str
    percent: int


ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


@ai.tool(name='prepareOrder', description='Prepare a kitchen order and stream stage progress.')
async def prepare_order(dish: str, ctx: ToolContext[PrepProgress]) -> str:
    ctx.send_chunk(PrepProgress(stage_name='searing', percent=40))
    await asyncio.sleep(0.2)
    ctx.send_chunk(PrepProgress(stage_name='plating', percent=90))
    await asyncio.sleep(0.2)
    return f'{dish} ready'


agent = ai.define_agent(
    name='kitchenPrepAgent',
    model=GoogleAI.gemini('gemini-flash-latest'),
    system='Kitchen expeditor. Always call prepareOrder when asked to prepare a dish.',
    tools=[prepare_order],
    store=store,
)


async def main() -> None:
    chat = agent.chat()
    turn = chat.send_stream('Prepare smoked salmon tartine.')

    seen_progress: list[PrepProgress] = []
    async for chunk in turn.stream:
        for part in chunk.tool_responses:
            if part.tool_response is None:
                continue
            if part.is_partial:
                progress = PrepProgress.model_validate(part.tool_response.output)
                seen_progress.append(progress)
                print(f'[{progress.percent}%] {progress.stage_name}')
            else:
                print(f'tool output: {part.tool_response.output}')

    res = await turn.response
    assert seen_progress
    assert all(not part.is_partial for msg in chat.messages if msg.role == Role.TOOL for part in msg.content)
    print(res.text)


if __name__ == '__main__':
    ai.run_main(main())
