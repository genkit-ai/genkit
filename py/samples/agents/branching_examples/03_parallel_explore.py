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

"""Explore several directions from one shared starting point.

After an initial turn, kick off multiple invocations from the same snapshotId
with different prompts — minimal, bold, corporate takes on the same brief.
Each one becomes its own leaf you can show as tabs or side-by-side columns.

Good when you want the user to browse options before committing to a single
timeline.
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
    # 1. Run the shared starting point
    print('--- TURN 1 (SHARED BRIEF) ---')
    session = agent.chat()
    async for chunk in session.send('Design a hero section'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    root_snap = session.snapshot_id
    assert root_snap
    print(f'Shared Checkpoint Saved: {root_snap}\n')

    # 2. Explore multiple directions in parallel from that checkpoint
    directions = ['Direction: minimal', 'Direction: bold', 'Direction: corporate']
    branches = []

    for label in directions:
        print(f'--- EXPLORING BRANCH [{label}] ---')
        session_branch = await agent.load_chat(root_snap)
        async for chunk in session_branch.send(label):
            if chunk.text:
                print(chunk.text, end='', flush=True)
        print()
        snap_id = session_branch.snapshot_id

        # Verify parentage in database
        snap = await store.get_snapshot(snapshot_id=snap_id)
        assert snap and snap.parent_id == root_snap

        branches.append((label, snap_id))
        print(f'Branch Saved: {snap_id}\n')

    # 3. Print the resulting tree leaves
    print(f'Explored {len(branches)} parallel branches successfully:')
    for label, snap_id in branches:
        print(f'  - [{label}] snapshot={snap_id}')


if __name__ == '__main__':
    ai.run_main(main())
