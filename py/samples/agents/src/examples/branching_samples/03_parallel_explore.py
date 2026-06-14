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

import asyncio
from uuid import uuid4

from _helpers import define_echo_agent, model_text, run_turn

from genkit import Genkit
from genkit._core._typing import AgentInit
from genkit.agent import InMemorySessionStore


async def main() -> None:
    ai = Genkit()
    store = InMemorySessionStore()
    agent = define_echo_agent(ai, store)

    session_id = str(uuid4())
    root = await run_turn(agent, AgentInit(session_id=session_id), 'Design a hero section')
    root_snap = root.snapshot_id
    assert root_snap
    print('shared checkpoint:', root_snap)

    directions = ['Direction: minimal', 'Direction: bold', 'Direction: corporate']
    branches = []
    for label in directions:
        out = await run_turn(agent, AgentInit(snapshot_id=root_snap), label)
        snap = await store.get_snapshot(snapshot_id=out.snapshot_id)
        assert snap and snap.parent_id == root_snap
        branches.append((label, out.snapshot_id, model_text(out)))

    print(f'explored {len(branches)} parallel branches:')
    for label, snap_id, reply in branches:
        print(f'  [{label}] snapshot={snap_id} reply={reply!r}')

    leaf_ids = {snap_id for _, snap_id, _ in branches}
    assert len(leaf_ids) == len(directions)


if __name__ == '__main__':
    asyncio.run(main())
