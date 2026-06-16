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

from uuid import uuid4

from _helpers import define_echo_agent, model_text, run_turn

from genkit import Genkit
from genkit.agent import AgentInit, InMemorySessionStore

ai = Genkit()
store = InMemorySessionStore()
agent = define_echo_agent(ai, store)


async def main() -> None:
    session_id = str(uuid4())
    root = await run_turn(agent, AgentInit(session_id=session_id), 'Start workspace')
    root_snap = root.snapshot_id
    assert root_snap

    minimal = await run_turn(agent, AgentInit(snapshot_id=root_snap), 'Direction: minimal')
    await run_turn(agent, AgentInit(snapshot_id=root_snap), 'Direction: bold')

    # User chose the minimal branch; continue from its leaf snapshot.
    active_leaf = minimal.snapshot_id
    assert active_leaf
    print('active branch (minimal):', active_leaf, '→', model_text(minimal))

    turn3 = await run_turn(
        agent,
        AgentInit(snapshot_id=active_leaf),
        'Add a pricing section',
    )
    turn3_snap = await store.get_snapshot(snapshot_id=turn3.snapshot_id)
    assert turn3_snap and turn3_snap.parent_id == active_leaf

    print('continued on chosen branch:', turn3.snapshot_id, '→', model_text(turn3))
    print('parent chain: root → minimal → turn 3')


if __name__ == '__main__':
    ai.run_main(main())
