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

from uuid import uuid4

from _helpers import define_echo_agent, model_text, run_turn

from genkit import Genkit
from genkit.agent import AgentInit, InMemorySessionStore

ai = Genkit()
store = InMemorySessionStore()
agent = define_echo_agent(ai, store)


async def main() -> None:
    session_id = str(uuid4())

    turn1 = await run_turn(agent, AgentInit(session_id=session_id), 'Plan a landing page')
    snap1 = turn1.snapshot_id
    assert snap1
    print('turn 1 snapshot:', snap1)

    # Main line: user continues linearly on the session.
    main_turn2 = await run_turn(agent, AgentInit(session_id=session_id), 'Make it minimal')
    snap2 = main_turn2.snapshot_id
    print('main line turn 2:', snap2, '→', model_text(main_turn2))

    # Regenerate: fork from *before* turn 2 and send alternate user text.
    alt_turn2 = await run_turn(agent, AgentInit(snapshot_id=snap1), 'Make it bold instead')
    snap2_alt = alt_turn2.snapshot_id
    assert snap2_alt and snap2_alt != snap2
    alt_snap = await store.get_snapshot(snapshot_id=snap2_alt)
    assert alt_snap and alt_snap.parent_id == snap1

    print('regenerated turn 2:', snap2_alt, '→', model_text(alt_turn2))
    print('sibling branches from snap₁:', snap2, snap2_alt)


if __name__ == '__main__':
    ai.run_main(main())
