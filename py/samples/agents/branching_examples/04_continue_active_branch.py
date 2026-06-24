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

"""Continue the conversation on whichever branch the user chose.

After sibling branches exist, the next turn resumes from that branch's latest
snapshotId — not sessionId alone. That's the step from "compare alternatives"
back to a normal linear chat on the path they picked.
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
    # 1. Run the initial root turn
    print('--- TURN 1 (ROOT SETUP) ---')
    session = agent.chat()
    async for chunk in session.send('Start workspace'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    root_snap = session.snapshot_id
    assert root_snap

    # 2. Fork sibling branches: Minimal and Bold
    print('--- TURN 2 (PATH A: MINIMAL) ---')
    session_minimal = await agent.load_chat(root_snap)
    async for chunk in session_minimal.send('Direction: minimal'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    active_leaf = session_minimal.snapshot_id
    assert active_leaf

    print('--- TURN 2 (PATH B: BOLD) ---')
    session_bold = await agent.load_chat(root_snap)
    async for chunk in session_bold.send('Direction: bold'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()

    # 3. User chose the minimal branch; continue linearly from its leaf snapshot
    print('--- TURN 3 (CONTINUE CHOSEN MINIMAL BRANCH) ---')
    session_turn3 = await agent.load_chat(active_leaf)
    async for chunk in session_turn3.send('Add a pricing section'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    turn3_snap_id = session_turn3.snapshot_id

    # 4. Verify parentage chain (root → minimal → turn 3)
    turn3_snap = await store.get_snapshot(snapshot_id=turn3_snap_id)
    assert turn3_snap and turn3_snap.parent_id == active_leaf
    print('\n Parent chain validation succeeded: turn 3 parent is minimal branch!')
    print(' Complete path: root → minimal → turn 3')


if __name__ == '__main__':
    ai.run_main(main())
