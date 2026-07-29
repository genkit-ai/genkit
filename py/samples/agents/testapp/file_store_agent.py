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

"""File-backed persistence: the server owns the conversation, on disk.

``fileStoreAgent`` is a logbook assistant whose history lives in a
``FileSessionStore``. A single ``chat`` persists after every turn and picks the
thread back up automatically, so a caller only ever needs the session's snapshot
id to resume — nothing about the conversation is round-tripped over the wire.

The second flow uses a store capped at three turns
(``max_persisted_chain_length=3``): each turn lands as its own
``<snapshot_id>.json`` file, and once the chain grows past the cap the oldest
snapshot is deleted so a long-lived session's history stays bounded on disk.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any

from _ai import LITE_MODEL, ai
from pydantic import BaseModel

from genkit import ActionRunContext
from genkit.agent import FileSessionStore

STORE_DIR = './.snapshots-filestore'
PRUNING_STORE_DIR = './.snapshots-pruning'
MAX_CHAIN = 3


# A logbook agent is cheap busywork, so it runs on the lite model.
file_store_agent = ai.define_agent(
    name='fileStoreAgent',
    model=LITE_MODEL,
    system='You are a personal logbook assistant.',
    store=FileSessionStore(STORE_DIR),
)


# Same logbook, but its on-disk history is capped: only the newest MAX_CHAIN
# turns survive, so the session can't grow files without bound.
pruning_agent = ai.define_agent(
    name='pruningAgent',
    model=LITE_MODEL,
    system='You are a personal logbook assistant.',
    store=FileSessionStore(PRUNING_STORE_DIR, max_persisted_chain_length=MAX_CHAIN),
)


class FileStoreResult(BaseModel):
    snapshot_id1: str | None = None
    reply1: str
    reply2: str


@ai.flow()
async def test_file_store_agent(user_name: str, ctx: ActionRunContext) -> FileStoreResult:
    """Two turns on one chat: the note logged in turn 1 is recalled in turn 2."""
    chat = file_store_agent.chat()

    turn1 = chat.send('Hello! Please log this note: I started studying Genkit today.')
    async for chunk in turn1:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res1 = await turn1

    turn2 = chat.send('What did I study today?')
    async for chunk in turn2:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res2 = await turn2

    return FileStoreResult(snapshot_id1=res1.snapshot_id, reply1=res1.text, reply2=res2.text)


@ai.flow()
async def test_file_store_chain_pruning(user_name: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Run four turns against the capped store and report what survived on disk.

    With the chain capped at three, the fourth turn evicts the first: its
    ``<snapshot_id>.json`` is gone while the newest three remain.
    """
    chat = pruning_agent.chat()

    snapshot_ids: list[str] = []
    for n in range(1, 5):
        turn = chat.send(f'Turn {n}')
        async for _ in turn:
            pass
        res = await turn
        if res.snapshot_id:
            snapshot_ids.append(res.snapshot_id)

    on_disk = {sid: os.path.exists(os.path.join(PRUNING_STORE_DIR, f'{sid}.json')) for sid in snapshot_ids}
    return {'snapshot_ids': snapshot_ids, 'on_disk': on_disk}


if __name__ == '__main__':
    # Lets you run this one agent on its own in the Dev UI:
    #   genkit start -- uv run testapp/file_store_agent.py
    import asyncio

    ai.run_main(asyncio.sleep(0))
