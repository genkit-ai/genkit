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

"""Cancel a long-running detached turn with task.abort().

Detach a turn, let it run for a moment, then abort it. The abort signal reaches the
running tool so it can stop cleanly, and the snapshot settles in an ABORTED state —
the cancel button for background agent work. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import InMemoryLatestStateStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    for _i in range(30):
        if ctx.abort_signal.is_set():  # the abort propagates here so we can bail out cleanly
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


async def main() -> None:
    session = agent.chat()

    # Kick off the background turn and let it run for a moment.
    task = await session.detach('Please run a long task using slowWork.')
    assert task.snapshot_id
    await asyncio.sleep(2.0)

    # → abort_signal fires inside slowWork; the snapshot settles as ABORTED
    await task.abort()
    await asyncio.sleep(1.0)  # let the background task unwind
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
