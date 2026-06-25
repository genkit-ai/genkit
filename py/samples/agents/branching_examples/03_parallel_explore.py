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

"""Explore several directions from one checkpoint, all at once.

After a shared brief, fork the same snapshot into several branches and run them
concurrently with different prompts. Each becomes its own leaf you can show as
tabs or side-by-side columns — the move when you want the user to browse options
before committing to one timeline. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit import Genkit
from genkit.agent import InMemoryBranchingSessionStore
from genkit.plugins.google_genai import GoogleAI

DIRECTIONS = ['minimal', 'bold', 'corporate']

ai = Genkit(plugins=[GoogleAI()])

agent = ai.define_agent(
    name='designer',
    model='googleai/gemini-flash-latest',
    system='You help design a product landing page. Reply in two or three short sentences.',
    store=InMemoryBranchingSessionStore(),
)


async def main() -> None:
    # One shared brief. Its snapshot is the checkpoint every branch forks from.
    root = agent.chat()
    await root.send('Design a hero section for a landing page.')
    checkpoint = root.snapshot_id
    assert checkpoint

    # Fork the checkpoint into independent branches and explore them concurrently.
    async def explore(snapshot: str, direction: str) -> None:
        branch = await agent.load_chat(snapshot)
        await branch.send(f'Take it in a {direction} direction.')

    # → each direction develops its own hero take from the shared brief, in parallel,
    #   and none of the branches sees the others.
    await asyncio.gather(*(explore(checkpoint, d) for d in DIRECTIONS))


if __name__ == '__main__':
    ai.run_main(main())
