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

"""Fork a conversation into parallel what-ifs from a shared snapshot.

Every settled turn leaves a snapshot id. Point a new chat at an earlier snapshot
and you get an independent branch that shares history up to that point but
diverges after — so you can explore "what if I'd said X instead" without
disturbing the original. This is the primitive behind regenerate, variants, and
time-travel.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import ai

from genkit import ActionRunContext
from genkit.agent import InMemorySessionStore

branching_agent = ai.define_agent(
    name='branchingAgent',
    system='You are a concise, friendly assistant. Answer in one short sentence.',
    store=InMemorySessionStore(),
)


@ai.flow()
async def test_branching_agent(text: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Establish a root turn, then fork two branches off the same snapshot."""
    root = branching_agent.chat()
    res1 = await root.send(text or 'Hello!')
    fork_point = res1.snapshot_id
    ctx.send_chunk(f'[fork point] {fork_point}')

    # Branch A shares history up to fork_point, then learns a different fact…
    branch_a = branching_agent.chat(snapshot_id=fork_point)
    await branch_a.send('My name is Bob.')
    res_a = await branch_a.send('What is my name? One word.')

    # …Branch B forks from the SAME snapshot and never hears about Bob.
    branch_b = branching_agent.chat(snapshot_id=fork_point)
    await branch_b.send('My name is John.')
    res_b = await branch_b.send('What is my name? One word.')

    return {'fork_point': fork_point, 'branch_a': res_a.text, 'branch_b': res_b.text}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
