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

"""Flat, full-snapshot session stores for local development and tests.

Each turn is persisted whole, keyed by its snapshot id, and the ``parent_id``
links between snapshots form the conversation tree. That single shape covers a
linear chat, a "just give me the latest turn" lookup, and a forked/branching
history without any diffing — so it's the right default for local dev, tests,
and single-process apps. For a multi-instance production deployment, back the
same ``SessionStore`` protocol with a real database (where it's worth trading
this simplicity for incremental, diff-based persistence).

``InMemorySessionStore`` and ``FileSessionStore`` are standalone — they share
the storage-agnostic bits below as plain functions rather than a common base
class, so each store is a self-contained read of how its backend works.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Generic
from uuid import uuid4

from genkit._ai._agents._session import (
    SessionStore,
    SnapshotSubscriber,
    StateT,
    select_leaf_snapshot,
)
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

SaveFn = Callable[[SessionSnapshot | None], SessionSnapshot | None]
Subs = dict[str, list['asyncio.Queue[SnapshotStatus | None]']]


def session_id_of(snapshot: SessionSnapshot) -> str | None:
    """Session a snapshot belongs to, preferring the top-level id over state's."""
    if snapshot.session_id:
        return snapshot.session_id
    return snapshot.state.session_id if snapshot.state is not None else None


def require_one_selector(*, snapshot_id: str | None, session_id: str | None) -> None:
    """Enforce that a get_snapshot call names exactly one of snapshot_id / session_id."""
    if bool(snapshot_id) == bool(session_id):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                "get_snapshot requires exactly one of 'snapshot_id' or "
                f"'session_id' (got {'snapshot_id' if snapshot_id else 'neither'}"
                f'{" and session_id" if session_id else ""}).'
            ),
        )


def select_leaf(
    *,
    snapshots: list[SessionSnapshot],
    session_id: str,
    reject_ambiguous: bool,
) -> SessionSnapshot | None:
    """Resolve a session's current leaf from all its snapshots.

    A leaf is a snapshot no other snapshot names as a parent. A linear chat has
    exactly one; a forked history has several. When opted in we reject the
    ambiguous case, otherwise the most recently created leaf wins so a sibling
    left behind by an aborted/failed turn never shadows the live one.
    """
    if not snapshots:
        return None

    if reject_ambiguous:
        return select_leaf_snapshot(snapshots=snapshots, session_id=session_id)

    parent_ids = {snap.parent_id for snap in snapshots if snap.parent_id}
    leaves = [snap for snap in snapshots if snap.snapshot_id not in parent_ids]
    if not leaves:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Session '{session_id}' has no leaf snapshot (corrupt or cyclic "
                'history). Resume by snapshot_id instead.'
            ),
        )
    # created_at is an ISO-8601 string, so lexicographic max is chronological;
    # snapshot_id breaks exact ties deterministically.
    return max(leaves, key=lambda snap: (snap.created_at, snap.snapshot_id))


def stamp_store_fields(*, snapshot: SessionSnapshot, snapshot_id: str | None) -> None:
    """Fill in the fields the store owns on a snapshot about to be written."""
    snapshot.snapshot_id = snapshot_id or str(uuid4())
    if not snapshot.created_at:
        snapshot.created_at = datetime.now(timezone.utc).isoformat()
    if not snapshot.status:
        snapshot.status = SnapshotStatus.COMPLETED
    # Mirror the session id up to the top level so session lookups and callers
    # reading snapshot.session_id don't have to dig into state.
    if not snapshot.session_id and snapshot.state is not None:
        snapshot.session_id = snapshot.state.session_id


def apply_save(*, existing: SessionSnapshot | None, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
    """Run a save mutator and stamp the result, or None to skip the write."""
    if snapshot_id is not None and existing is None:
        return None
    next_snapshot = fn(existing.model_copy(deep=True) if existing is not None else None)
    if next_snapshot is None:
        return None
    stamp_store_fields(snapshot=next_snapshot, snapshot_id=snapshot_id)
    return next_snapshot


def notify(*, subs: Subs, snapshot_id: str, status: SnapshotStatus | None) -> None:
    """Push a status change to everyone subscribed to a snapshot."""
    # Subscriber queues are unbounded, so put_nowait can't fail here.
    for q in subs.get(snapshot_id, []):
        q.put_nowait(status)


async def subscribe(
    *,
    subs: Subs,
    snapshot_id: str,
    current: SessionSnapshot | None,
) -> asyncio.Queue[SnapshotStatus | None]:
    """Register a status-change queue, seeding it with the current status."""
    q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
    if current is None:
        await q.put(None)
        return q
    await q.put(current.status)
    subs.setdefault(snapshot_id, []).append(q)
    return q


class InMemorySessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """In-memory snapshot store. State is lost when the process exits."""

    def __init__(self, *, reject_ambiguous_session: bool = False) -> None:
        """Create the store.

        When ``reject_ambiguous_session`` is set, a ``session_id`` lookup on a
        history that has forked (more than one leaf) raises instead of picking
        the most recent branch — useful when accidental branching should surface
        loudly.
        """
        self.reject_ambiguous = reject_ambiguous_session
        self.snapshots: dict[str, SessionSnapshot] = {}
        self.subs: Subs = {}

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        async with self.lock:
            if snapshot_id is not None:
                snap = self.snapshots.get(snapshot_id)
                return snap.model_copy(deep=True) if snap is not None else None

            assert session_id is not None
            owned = [snap for snap in self.snapshots.values() if session_id_of(snap) == session_id]
            leaf = select_leaf(snapshots=owned, session_id=session_id, reject_ambiguous=self.reject_ambiguous)
            return leaf.model_copy(deep=True) if leaf is not None else None

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
        async with self.lock:
            existing = self.snapshots.get(snapshot_id) if snapshot_id is not None else None
            next_snapshot = apply_save(existing=existing, snapshot_id=snapshot_id, fn=fn)
            if next_snapshot is None:
                return None
            self.snapshots[next_snapshot.snapshot_id] = next_snapshot.model_copy(deep=True)
            notify(subs=self.subs, snapshot_id=next_snapshot.snapshot_id, status=next_snapshot.status)
            return next_snapshot

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        async with self.lock:
            return await subscribe(subs=self.subs, snapshot_id=snapshot_id, current=self.snapshots.get(snapshot_id))


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
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        async with self.lock:
            if snapshot_id is not None:
                return await asyncio.to_thread(self.read_sync, snapshot_id)

            assert session_id is not None
            owned = await asyncio.to_thread(self.read_session_sync, session_id)
            return select_leaf(snapshots=owned, session_id=session_id, reject_ambiguous=self.reject_ambiguous)

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
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
        async with self.lock:
            current = await asyncio.to_thread(self.read_sync, snapshot_id)
            return await subscribe(subs=self.subs, snapshot_id=snapshot_id, current=current)
