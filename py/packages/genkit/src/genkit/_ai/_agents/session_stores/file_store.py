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

"""File-backed, full-snapshot session store for local development and tests.

See :mod:`genkit._ai._agents.session_stores.util` for the flat, full-snapshot
design these stores share.
"""

from __future__ import annotations

import asyncio
import os
from typing import Generic

from genkit._ai._agents._session import (
    SessionStore,
    SnapshotSubscriber,
    StateT,
)
from genkit._ai._agents.session_stores.util import (
    SaveFn,
    Subs,
    apply_save,
    notify,
    require_one_selector,
    select_leaf,
    session_id_of,
    subscribe,
)
from genkit._core._typing import SessionSnapshot, SnapshotStatus


class FileSessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """File-backed snapshot store: one ``<snapshot_id>.json`` per snapshot."""

    def __init__(
        self,
        directory: str,
        *,
        reject_ambiguous_session: bool = False,
        max_persisted_chain_length: int | None = None,
    ) -> None:
        """Create the store, ensuring ``directory`` exists.

        See :class:`InMemorySessionStore` for ``reject_ambiguous_session``.

        ``max_persisted_chain_length`` caps how many snapshots of a chat's
        history stay on disk: once a chain grows past it, the oldest turns are
        deleted on each save so a long-lived conversation doesn't accumulate
        files forever. Resuming and continuing still work, but you lose the
        ability to rewind or branch past the retained window. Leave it unset to
        keep the full history.
        """
        self.reject_ambiguous = reject_ambiguous_session
        self.max_persisted_chain_length = max_persisted_chain_length
        self.directory = directory
        self.subs: Subs = {}
        os.makedirs(directory, exist_ok=True)

    def path(self, snapshot_id: str) -> str:
        """Return the file path for a snapshot ID."""
        return os.path.join(self.directory, f'{snapshot_id}.json')

    def read_sync(self, snapshot_id: str) -> SessionSnapshot | None:
        """Read and validate a snapshot from disk synchronously."""
        path = self.path(snapshot_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            return SessionSnapshot.model_validate_json(f.read())

    def write_sync(self, snapshot: SessionSnapshot) -> None:
        """Atomically write a snapshot to disk synchronously."""
        path = self.path(snapshot.snapshot_id)
        temp_path = path + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(snapshot.model_dump_json(indent=2))
        os.replace(temp_path, path)

    def delete_sync(self, snapshot_id: str) -> None:
        """Delete a snapshot file, tolerating one that's already gone."""
        try:
            os.remove(self.path(snapshot_id))
        except FileNotFoundError:
            pass

    def prune_chain_sync(self, leaf: SessionSnapshot) -> None:
        """Trim a chat's ancestry to the newest ``max_persisted_chain_length`` turns.

        Walks ``parent_id`` back from the just-written snapshot and deletes the
        oldest links past the cap. The retained oldest snapshot keeps pointing at
        its (now-deleted) parent, so history reconstruction simply stops at the
        window's edge.
        """
        cap = self.max_persisted_chain_length
        if not cap or cap <= 0:
            return
        chain: list[str] = []
        seen: set[str] = set()
        cur: SessionSnapshot | None = leaf
        # `seen` stops a corrupt/cyclic parent chain from looping forever (each
        # hop is a disk read), the same guard walk_back_to_resumable uses.
        while cur is not None and cur.snapshot_id not in seen:
            seen.add(cur.snapshot_id)
            chain.append(cur.snapshot_id)
            cur = self.read_sync(cur.parent_id) if cur.parent_id else None
        for snapshot_id in chain[cap:]:
            self.delete_sync(snapshot_id)

    def read_session_sync(self, session_id: str) -> list[SessionSnapshot]:
        """Read all snapshots for a session from disk synchronously."""
        out: list[SessionSnapshot] = []
        if not os.path.isdir(self.directory):
            return out
        for name in os.listdir(self.directory):
            if not name.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.directory, name), encoding='utf-8') as f:
                    snap = SessionSnapshot.model_validate_json(f.read())
            except (OSError, ValueError):
                continue
            if session_id_of(snap) == session_id:
                out.append(snap)
        return out

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Return a snapshot by id, or the session's latest leaf, read from disk."""
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        async with self.lock:
            if snapshot_id is not None:
                return await asyncio.to_thread(self.read_sync, snapshot_id)

            assert session_id is not None
            owned = await asyncio.to_thread(self.read_session_sync, session_id)
            return select_leaf(snapshots=owned, session_id=session_id, reject_ambiguous=self.reject_ambiguous)

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
        """Read-modify-write a snapshot on disk, prune the chain, and notify subscribers."""
        async with self.lock:
            existing = await asyncio.to_thread(self.read_sync, snapshot_id) if snapshot_id is not None else None
            next_snapshot = apply_save(existing=existing, snapshot_id=snapshot_id, fn=fn)
            if next_snapshot is None:
                return None
            await asyncio.to_thread(self.write_sync, next_snapshot)
            if self.max_persisted_chain_length:
                await asyncio.to_thread(self.prune_chain_sync, next_snapshot)
            notify(subs=self.subs, snapshot_id=next_snapshot.snapshot_id, status=next_snapshot.status)
            return next_snapshot

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Return a queue that receives this snapshot's status changes."""
        async with self.lock:
            current = await asyncio.to_thread(self.read_sync, snapshot_id)
            return await subscribe(subs=self.subs, snapshot_id=snapshot_id, current=current)
