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

from uuid import uuid4

from _helpers import define_echo_agent, model_text, run_turn

from genkit import Genkit, GenkitError
from genkit.agent import AgentInit, InMemoryBranchingSessionStore

ai = Genkit()
store = InMemoryBranchingSessionStore()
agent = define_echo_agent(ai, store)


async def main() -> None:
    session_id = str(uuid4())
    root = await run_turn(agent, AgentInit(session_id=session_id), 'Plan a landing page')
    root_snap = root.snapshot_id
    assert root_snap
    print('root snapshot:', root_snap)
    print('root reply:', model_text(root))

    minimal = await run_turn(agent, AgentInit(snapshot_id=root_snap), 'Direction: minimal')
    bold = await run_turn(agent, AgentInit(snapshot_id=root_snap), 'Direction: bold')
    assert minimal.snapshot_id and bold.snapshot_id and minimal.snapshot_id != bold.snapshot_id

    min_snap = await store.get_snapshot(snapshot_id=minimal.snapshot_id)
    bold_snap = await store.get_snapshot(snapshot_id=bold.snapshot_id)
    assert min_snap and bold_snap
    assert min_snap.parent_id == root_snap
    assert bold_snap.parent_id == root_snap

    print('minimal branch:', minimal.snapshot_id, '→', model_text(minimal))
    print('bold branch:', bold.snapshot_id, '→', model_text(bold))

    try:
        await store.get_snapshot(session_id=session_id)
    except GenkitError as exc:
        print('sessionId lookup after fork (expected failure):', exc)
    else:
        raise AssertionError('expected sessionId lookup to fail once the session has branched')


if __name__ == '__main__':
    ai.run_main(main())
