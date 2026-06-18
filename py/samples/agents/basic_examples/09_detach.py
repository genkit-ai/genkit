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

"""Backend: session.detach() — early return + poll background execution using AgentAPI."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import InMemoryLatestStateStore, AgentInit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    for i in range(10):
        if ctx.abort_signal.is_set():
            raise GenkitError(status='ABORTED', message='Task aborted')
        print(f'[slowWork] working step {i+1}/10...')
        await asyncio.sleep(0.5)
    return {'done': True}


agent = ai.define_agent(
    name='longTaskAgent',
    model='googleai/gemini-flash-latest',
    system='When asked for a long task, call slowWork.',
    tools=[slow_work],
    store=store,
)


async def main() -> None:
    session = agent.connect(AgentInit(session_id=str(uuid4())))
    print("--- SUBMITTING DETACHED TASK ---")
    task = await session.detach('Please run a long task using slowWork.')
    print(f'Task detached! Snapshot ID: {task.snapshot_id}')

    # We can poll or wait for the task to complete
    print("\n--- POLLING TASK STATUS ---")
    async for snap in task.poll(interval_ms=1000):
        print(f'Task Status: {snap.status}, Messages: {len(snap.state.messages or [])}')

    print("Finished!")
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
