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

"""Fire a long turn into the background and poll it to completion.

session.detach() submits a turn and returns immediately with a snapshot id instead of
streaming — the work runs server-side. You poll that snapshot until it reaches a
terminal status. This is the shape of a job-queue / async-task API on top of an agent.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit import Genkit, GenkitError, ToolRunContext
from genkit.agent import InMemoryLatestStateStore, SnapshotStatus
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()


@ai.tool(name='slowWork', description='Simulate long background work.')
async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
    for _i in range(10):
        if ctx.abort_signal.is_set():
            raise GenkitError(status='ABORTED', message='Task aborted')
        await asyncio.sleep(0.5)  # pretend each step is real work
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

    # Submit the turn and return right away — the work continues in the background.
    task = await session.detach('Please run a long task using slowWork.')
    assert task.snapshot_id  # the handle you poll on, hand off, or persist

    # Poll the snapshot until it reaches a terminal status.
    while True:
        snap = await task.poll()
        if snap and snap.status in (SnapshotStatus.COMPLETED, SnapshotStatus.FAILED, SnapshotStatus.ABORTED):
            break  # → settles COMPLETED once slowWork finishes its steps
        await asyncio.sleep(1.0)

    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
