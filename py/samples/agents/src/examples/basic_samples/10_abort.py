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

"""Backend: abort a detached invocation via store.abort_snapshot().

Python has no conn.abort() — after detach(), call abort on the agent's store
with the snapshot_id from conn.output(). That flips the pending snapshot to
aborted and signals abort_signal inside the still-running agent fn.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import AgentInit, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI


async def main() -> None:
    ai = Genkit(plugins=[GoogleAI()])

    store = InMemorySessionStore()

    @ai.tool(name='slowWork', description='Simulate long background work.')
    async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
        for _ in range(30):
            if ctx.abort_signal.is_set():
                raise GenkitError(status='ABORTED', message='Task aborted')
            await asyncio.sleep(0.5)
        return {'done': True}

    agent = ai.define_agent(
        name='longTaskAgent',
        model='googleai/gemini-flash-latest',
        system='When asked for a long task, call slowWork.',
        tools=[slow_work],
        store=store,
    )

    conn = await agent.stream_bidi(AgentInit(session_id=str(uuid4())))
    await conn.send_text('Please run a long task using slowWork.')
    await conn.detach()

    async for chunk in conn.receive():
        print('chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out = await conn.output()
    snap_id = out.snapshot_id
    print('detached snapshot_id:', snap_id)

    status = await store.abort_snapshot(snap_id)
    print('abort_snapshot returned:', status)


if __name__ == '__main__':
    asyncio.run(main())
