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

"""Backend: abort a detached invocation via task.abort() using AgentAPI."""

from __future__ import annotations

import asyncio

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import InMemoryLatestStateStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    try:
        for i in range(30):
            if ctx.abort_signal.is_set():
                print('[slowWork] Abort signal detected!')
                raise GenkitError(status='ABORTED', message='Task aborted')
            print(f'[slowWork] working step {i + 1}/30...')
            await asyncio.sleep(0.5)
        return {'done': True}
    except asyncio.CancelledError:
        print('[slowWork] cancelled!')
        raise


agent = ai.define_agent(
    name='longTaskAgent',
    model='googleai/gemini-flash-latest',
    system='When asked for a long task, call slowWork.',
    tools=[slow_work],
    store=store,
)


async def main() -> None:
    session = agent.chat()
    print('--- SUBMITTING DETACHED TASK ---')
    task = await session.detach('Please run a long task using slowWork.')
    print(f'Task detached! Snapshot ID: {task.snapshot_id}')

    await asyncio.sleep(2.0)

    print('\n--- ABORTING TASK ---')
    status = await task.abort()
    print('abort returned status:', status)

    # Wait a little to let the background logs print
    await asyncio.sleep(1.0)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
