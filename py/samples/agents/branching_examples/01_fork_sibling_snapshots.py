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

"""Fork one conversation into two siblings from the same checkpoint.

Run a shared setup turn, then start two branches from that turn's snapshot with
different follow-ups. You get sibling timelines instead of one linear history —
the move when someone wants to compare directions without losing the setup.

Once the tree forks, "the latest turn for this session" is ambiguous, so looking
the session up by id surfaces a structured, recoverable error instead of guessing.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit import Genkit, GenkitError, SessionErrorType
from genkit.agent import InMemoryBranchingSessionStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryBranchingSessionStore()

agent = ai.define_agent(
    name='designer',
    model='googleai/gemini-flash-latest',
    system='You help design a product landing page. Reply in two or three short sentences.',
    store=store,
)


async def main() -> None:
    # One shared setup turn. Its snapshot is the fork point for both siblings.
    root = agent.chat()
    await root.send('Plan a landing page for a note-taking app.')
    checkpoint = root.snapshot_id
    session_id = root.session_id
    assert checkpoint and session_id

    # Two branches off the same checkpoint; neither sees the other.
    # → minimal gets a whitespace-heavy take; bold gets a dark, high-contrast one.
    minimal = await agent.load_chat(checkpoint)
    await minimal.send('Direction: minimal.')
    bold = await agent.load_chat(checkpoint)
    await bold.send('Direction: bold.')
    bold_leaf = bold.snapshot_id
    assert bold_leaf

    # The tree has two leaves now, so a session-id lookup can't pick "the latest"
    # turn. Genkit raises AMBIGUOUS_BRANCH carrying the conflicting leaves; you
    # resolve it by continuing from the specific leaf you mean.
    try:
        await store.get_snapshot(session_id=session_id)
    except GenkitError as exc:
        # → exc.details holds type=AMBIGUOUS_BRANCH and the conflicting leaves
        assert exc.details and exc.details.get('type') == SessionErrorType.AMBIGUOUS_BRANCH
        resumed = await agent.load_chat(bold_leaf)
        # → continues the bold timeline, extending it with a pricing section
        await resumed.send('Add a pricing section.')


if __name__ == '__main__':
    ai.run_main(main())
