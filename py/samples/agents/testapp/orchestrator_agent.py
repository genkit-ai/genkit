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

"""One agent that delegates to specialist sub-agents.

The trick is that an agent is just something you can ``chat()`` with — so a
delegation tool can spin up a sub-agent, run a turn, and hand its answer back as
the tool result. The orchestrator picks which specialist to call; the specialists
(researcher, coder) stay focused. This is multi-agent composition with nothing
but tools and chats.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from _ai import ai
from pydantic import BaseModel

from genkit import ActionRunContext

# Specialists. They're ordinary agents — the orchestrator reaches them through
# the delegation tools below, which just run a one-shot chat against each.
researcher = ai.define_agent(
    name='researcher',
    description='A thorough research assistant that gives well-organized, factual answers.',
    system='You are a thorough research assistant. Answer clearly and factually in a few short paragraphs.',
)

coder = ai.define_agent(
    name='coder',
    description='An expert programmer that writes clean, well-commented code.',
    system='You are an expert programmer. Write clean, well-commented code with a short explanation.',
)


class Task(BaseModel):
    task: str


@ai.tool(name='delegate_to_researcher', description='Hand a research question to the researcher specialist.')
async def delegate_to_researcher(input: Task) -> str:
    return (await researcher.chat().send(input.task)).text


@ai.tool(name='delegate_to_coder', description='Hand a programming task to the coder specialist.')
async def delegate_to_coder(input: Task) -> str:
    return (await coder.chat().send(input.task)).text


orchestrator_agent = ai.define_agent(
    name='orchestratorAgent',
    system=(
        'You are a project lead. Analyze the request and delegate: use delegate_to_researcher for '
        'research and delegate_to_coder for code. If a request needs both, call them in turn. Then '
        "synthesize the specialists' results into one final answer for the user."
    ),
    tools=[delegate_to_researcher, delegate_to_coder],
)


@ai.flow()
async def test_orchestrator_agent(text: str, ctx: ActionRunContext) -> str:
    chat = orchestrator_agent.chat()
    turn = chat.send(text or 'Research quicksort, then write a Python implementation of it.')
    async for chunk in turn:
        for call in chunk.tool_requests:
            ctx.send_chunk(f'[delegating] {call.tool_request.name}')
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    return res.text


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
