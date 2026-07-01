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

"""Continue the conversation on whichever branch the user picked.

Once sibling branches exist, the next turn resumes from that branch's leaf
snapshot — not the session id, which is now ambiguous. That's the step from
"compare alternatives" back to a normal linear chat on the chosen path.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit import Genkit
from genkit.agent import InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='designer',
    model='googleai/gemini-flash-latest',
    system='You help design a product landing page. Reply in two or three short sentences.',
    store=InMemorySessionStore(),
)


async def main() -> None:
    # Shared setup turn — the checkpoint both branches fork from.
    root = agent.chat()
    await root.send('Plan a landing page for a note-taking app.')
    checkpoint = root.snapshot_id
    assert checkpoint

    # Fork two sibling directions. The user will pick one to keep building on.
    minimal = await agent.load_chat(snapshot_id=checkpoint)
    await minimal.send('Direction: minimal.')
    chosen_leaf = minimal.snapshot_id
    bold = await agent.load_chat(snapshot_id=checkpoint)
    await bold.send('Direction: bold.')
    assert chosen_leaf

    # User picked minimal — resume a normal linear chat from that leaf snapshot.
    resumed = await agent.load_chat(snapshot_id=chosen_leaf)
    # → continues the minimal timeline as a plain linear chat, adding pricing
    await resumed.send('Add a pricing section.')


if __name__ == '__main__':
    ai.run_main(main())
