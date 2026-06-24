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

"""Fork one conversation into two paths from the same checkpoint.

Run a first turn normally, then start two invocations from that turn's
snapshotId with different follow-up messages. You get sibling snapshots
instead of one linear history — handy when someone wants to compare directions
(e.g. "minimal" vs "bold") without losing the shared setup.

Once the tree branches, "latest for this session" is ambiguous; the client
needs to track snapshotId per path.
"""

from __future__ import annotations

from genkit import Genkit, GenkitError, SessionErrorType, StatusCodes
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
    # 1. Run the initial setup turn normally
    print('--- TURN 1 (ROOT SETUP) ---')
    session = agent.chat()
    async for chunk in session.send('Plan a landing page'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()

    root_snap = session.snapshot_id
    session_id = session.session_id
    assert root_snap
    print(f'Root Snapshot Saved: {root_snap}\n')

    # 2. Fork Path A (Minimal) from the root checkpoint
    print('--- TURN 2 (PATH A: MINIMAL) ---')
    session_minimal = await agent.load_chat(root_snap)
    async for chunk in session_minimal.send('Direction: minimal'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    min_snap_id = session_minimal.snapshot_id
    print(f'Minimal Branch Snapshot: {min_snap_id}\n')

    # 3. Fork Path B (Bold) from the exact same root checkpoint
    print('--- TURN 2 (PATH B: BOLD) ---')
    session_bold = await agent.load_chat(root_snap)
    async for chunk in session_bold.send('Direction: bold'):
        if chunk.text:
            print(chunk.text, end='', flush=True)
    print()
    bold_snap_id = session_bold.snapshot_id
    print(f'Bold Branch Snapshot: {bold_snap_id}\n')

    # 4. Verify parentage relationships in the database
    min_snap = await store.get_snapshot(snapshot_id=min_snap_id)
    bold_snap = await store.get_snapshot(snapshot_id=bold_snap_id)
    assert min_snap and bold_snap
    assert min_snap.parent_id == root_snap
    assert bold_snap.parent_id == root_snap
    print(' Parent validation succeeded: Both branches fork from root_snap!')

    # 5. Handle the branching conflict (ambiguousBranch) as a first-class recovery flow
    print('--- CONFLICT RESOLUTION & RECOVERY ---')
    try:
        # Looking up by sessionId is ambiguous because the session has branched (two leaf snapshots)
        await store.get_snapshot(session_id=session_id)
    except GenkitError as exc:
        # A. Assert that we got the expected precondition failure and structured details
        assert exc.status == StatusCodes.FAILED_PRECONDITION
        assert exc.details and exc.details.get('type') == SessionErrorType.AMBIGUOUS_BRANCH

        # B. Extract the conflicting leaves from the structured metadata
        leaves = exc.details.get('leaves')
        print(f'⚠️ Detected Ambiguous Branch Conflict for Session: {session_id}')
        print(f'   Conflicting Leaf Snapshots in DB: {leaves}')

        # C. Programmatically resolve the conflict by choosing one of the branches to resume!
        # In this case, we choose to resume Path B (the bold branch) by choosing its snapshot ID
        chosen_leaf = leaves[1] if leaves[1] == bold_snap_id else leaves[0]
        print(f'✅ Programmatically resolving conflict: Resuming from Bold Branch ({chosen_leaf})...')

        resolved_session = await agent.load_chat(chosen_leaf)
        async for chunk in resolved_session.send('Add a pricing section to the landing page'):
            if chunk.text:
                print(chunk.text, end='', flush=True)
        print()
        print(f'🚀 Recovery Flow Completed! New active snapshot: {resolved_session.snapshot_id}')
    else:
        raise AssertionError('Expected sessionId lookup to fail once the session has branched')


if __name__ == '__main__':
    ai.run_main(main())
