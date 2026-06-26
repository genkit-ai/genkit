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

"""Regenerate a turn without throwing away the original answer.

Walk the main line through two turns, then fork from the snapshot after turn 1
and replay turn 2 with different input. Both answers stick around as siblings —
the flow behind "try again" or edit-and-resubmit when you don't want to overwrite
what the user already saw. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit import Genkit
from genkit.agent import InMemoryBranchingSessionStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='designer',
    model='googleai/gemini-flash-latest',
    system='You help design a product landing page. Reply in two or three short sentences.',
    store=InMemoryBranchingSessionStore(),
)


async def main() -> None:
    # Turn 1 sets the scene; bookmark it, then keep going on the main line.
    session = agent.chat()
    await session.send('Plan a landing page for a note-taking app.')
    checkpoint = session.snapshot_id
    assert checkpoint
    await session.send('Make it minimal.')  # → the original "minimal" answer

    # Regenerate turn 2 from the bookmark with different input. The original
    # "minimal" answer is untouched; both versions now coexist as siblings.
    retry = await agent.load_chat(snapshot_id=checkpoint)
    await retry.send('Make it bold instead.')  # → a "bold" sibling of that same turn


if __name__ == '__main__':
    ai.run_main(main())
