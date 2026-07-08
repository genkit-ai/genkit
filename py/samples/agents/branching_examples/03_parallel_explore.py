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

"""Explore several directions from one checkpoint, all at once — the right way.

A chat is a single cursor over the snapshot tree: each turn resumes from the
chat's current snapshot and, when it finishes, drags the cursor to the new leaf.
So you can't fan one chat out into parallel branches. Fire ``chat.send()`` a few
times at once and all of them fork from the same point, then fight over the
cursor — it ends up on whichever turn happens to finish last, and their messages
interleave into one mangled timeline. One cursor can't be three branches.

To branch, give each direction its own cursor: fork the checkpoint into a
separate chat with ``load_chat``, then run those concurrently. Each becomes its
own leaf you can show as tabs or side-by-side columns. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit_google_genai import GoogleAI

from genkit import Genkit
from genkit.agent import InMemorySessionStore

DIRECTIONS = ['minimal', 'bold', 'corporate']

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='designer',
    model='googleai/gemini-flash-latest',
    system='You help design a product landing page. Reply in two or three short sentences.',
    store=InMemorySessionStore(),
)


async def main() -> None:
    # One shared brief. Its snapshot is the checkpoint every branch forks from.
    root = agent.chat()
    await root.send('Design a hero section for a landing page.')
    checkpoint = root.snapshot_id
    assert checkpoint

    # ✗ DON'T fan out one chat. `root` is a single cursor, so this makes all
    #   three sends fork from `checkpoint` and then race to drag the cursor —
    #   it lands on whichever finishes last and the messages interleave:
    #
    #       await asyncio.gather(*(root.send(f'Take it in a {d} direction.') for d in DIRECTIONS))
    #
    # ✓ DO give each direction its own cursor: fork the checkpoint into a
    #   separate chat, then those are safe to run concurrently.
    async def explore(direction: str) -> None:
        branch = await agent.load_chat(snapshot_id=checkpoint)
        await branch.send(f'Take it in a {direction} direction.')

    # → each direction develops its own hero take from the shared brief, in parallel,
    #   and none of the branches sees the others (or moves the root cursor).
    await asyncio.gather(*(explore(d) for d in DIRECTIONS))


if __name__ == '__main__':
    ai.run_main(main())
