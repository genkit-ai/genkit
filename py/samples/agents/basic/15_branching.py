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

"""Branch a conversation: fork one turn into sibling timelines.

Every store-backed turn is a snapshot you can fork from. Run a shared setup turn,
then start two branches from that turn's snapshot with different follow-ups. You
get sibling timelines instead of one linear history — the move when someone wants
to compare directions, or hit "try again" on a turn, without losing the setup or
overwriting the first answer.

Once the tree forks, "the latest turn for this session" is ambiguous, so a
session-id lookup surfaces a structured, recoverable error instead of guessing.
You resolve it by continuing from the specific leaf you mean — and from there
it's a normal linear chat again. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI

from genkit import Genkit, GenkitError
from genkit.agent import InMemorySessionStore

ai = Genkit(plugins=[GoogleAI()])
# reject_ambiguous_session makes a session-id lookup over a forked history raise
# instead of silently picking the newest branch — that's what surfaces the
# ambiguous-branch error below.
store = InMemorySessionStore(reject_ambiguous_session=True)

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

    # Fork the checkpoint twice into sibling timelines; neither sees the other.
    # This is also the "try again" / edit-and-resubmit move: re-run the turn with
    # different input while the first answer stays put as its own sibling.
    # → minimal gets a whitespace-heavy take; bold gets a dark, high-contrast one.
    minimal = await agent.load_chat(snapshot_id=checkpoint)
    await minimal.send('Direction: minimal.')
    bold = await agent.load_chat(snapshot_id=checkpoint)
    await bold.send('Direction: bold.')
    chosen_leaf = bold.snapshot_id
    assert chosen_leaf

    # Two leaves now, so a session-id lookup can't pick "the latest" turn. Genkit
    # raises FAILED_PRECONDITION rather than silently guessing which branch you meant.
    try:
        await store.get_snapshot(session_id=session_id)
        raise AssertionError('expected an ambiguous-session lookup to fail')
    except GenkitError as exc:
        assert exc.status == 'FAILED_PRECONDITION'

    # Resolve by resuming the specific leaf you want. From here it's a normal
    # linear chat again — this extends the bold timeline with a pricing section,
    # and the minimal sibling is left untouched.
    resumed = await agent.load_chat(snapshot_id=chosen_leaf)
    await resumed.send('Add a pricing section.')


if __name__ == '__main__':
    ai.run_main(main())
