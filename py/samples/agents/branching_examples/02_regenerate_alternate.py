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

"""Regenerate a turn without discarding the original answer.

The user walks the main line through two turns, then you fork from the snapshot
after turn 1 and replay turn 2 with different input. Both versions stick around
as siblings — the flow you'd wire up for "try again" or edit-and-resubmit when
you don't want to overwrite what they already saw.
"""

from __future__ import annotations

from genkit import Genkit
from genkit.agent import InMemoryBranchingSessionStore
from genkit.plugins.google_genai import GoogleAI

# Initialize Genkit with GoogleAI plugin and the purpose-built branching store
ai = Genkit(plugins=[GoogleAI()])
store = InMemoryBranchingSessionStore()

# Define the agent natively using Gemini Flash
agent = ai.define_agent(
    name='branchEcho',
    model='googleai/gemini-flash-latest',
    store=store,
)


async def main() -> None:
    # 1. Run Turn 1 (Checkpoint Root)
    print('--- TURN 1 ---')
    session = agent.chat()
    async for chunk in session.send('Plan a landing page'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    snap1 = session.snapshot_id
    assert snap1
    print(f'Turn 1 Snapshot Saved: {snap1}\n')

    # 2. Main Line: Continue linearly on the same session
    print('--- TURN 2 (MAIN LINE) ---')
    async for chunk in session.send('Make it minimal'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    snap2 = session.snapshot_id
    print(f'Main Line Turn 2 Snapshot: {snap2}\n')

    # 3. Regenerate: Fork from before turn 2 and send alternate input
    print('--- TURN 2 (ALTERNATE BRANCH) ---')
    session_alt = await agent.load_chat(snap1)
    async for chunk in session_alt.send('Make it bold instead'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    snap2_alt = session_alt.snapshot_id
    print(f'Alternate Turn 2 Snapshot: {snap2_alt}\n')

    # 4. Verify parentage
    assert snap2_alt != snap2
    alt_snap = await store.get_snapshot(snapshot_id=snap2_alt)
    assert alt_snap and alt_snap.parent_id == snap1
    print(' Parent validation succeeded: Both turns branch from Turn 1!')


if __name__ == '__main__':
    ai.run_main(main())
