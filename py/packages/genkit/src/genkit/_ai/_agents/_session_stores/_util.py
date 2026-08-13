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

"""Shared utilities for building and extending Genkit agent session stores.

Provides common helpers for state persistence, invariant enforcement, and
real-time status streaming across both built-in (in-memory, file) and custom
durable backends (such as Firestore).
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime, timezone

from genkit._ai._agents._session import SnapshotStatusStream, select_leaf_snapshot
from genkit._ai._agents._snapshot import parse_snapshot_lookup_kw
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

SaveFn = Callable[[SessionSnapshot | None], SessionSnapshot | None]
Subs = dict[str, list['asyncio.Queue[SnapshotStatus | None]']]
OnClose = Callable[[], Awaitable[None]]

# Persisted statuses that end a snapshot's store lifecycle. apply_save treats
# these as absorbing; status streams end after one. ``expired`` is a read-time
# overlay and is never written, so it is not included here.
TERMINAL_STATUSES = frozenset({
    SnapshotStatus.COMPLETED,
    SnapshotStatus.FAILED,
    SnapshotStatus.ABORTED,
})


def assert_safe_snapshot_id(*, snapshot_id: str) -> None:
    """Reject snapshot ids that could escape a store directory when used as a filename.

    Snapshot ids can arrive straight off the wire (abort/getSnapshot take a bare
    string), so without this an id like ``../../foo`` would let a caller read or
    write outside the store directory.
    """
    if (
        not snapshot_id
        or '/' in snapshot_id
        or '\\' in snapshot_id
        or '\0' in snapshot_id
        or snapshot_id in ('.', '..')
        or os.path.basename(snapshot_id) != snapshot_id
    ):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                f'Invalid snapshotId: "{snapshot_id}". '
                'A snapshotId must be a plain file name (no path separators or "..").'
            ),
        )


def session_id_of(snapshot: SessionSnapshot) -> str | None:
    """Session a snapshot belongs to, preferring the top-level id over state's."""
    if snapshot.session_id:
        return snapshot.session_id
    return snapshot.state.session_id if snapshot.state is not None else None


def require_one_selector(*, snapshot_id: str | None, session_id: str | None) -> None:
    """Enforce that a get_snapshot call names exactly one of snapshot_id / session_id."""
    parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)


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


def stamp_store_fields(*, snapshot: SessionSnapshot, snapshot_id: str) -> None:
    """Fill in the fields the store owns on a snapshot about to be written."""
    snapshot.snapshot_id = snapshot_id
    if not snapshot.created_at:
        snapshot.created_at = datetime.now(timezone.utc).isoformat()
    if not snapshot.status:
        snapshot.status = SnapshotStatus.COMPLETED
    # Mirror the session id up to the top level so session lookups and callers
    # reading snapshot.session_id don't have to dig into state.
    if not snapshot.session_id and snapshot.state is not None:
        snapshot.session_id = snapshot.state.session_id


def apply_save(*, existing: SessionSnapshot | None, snapshot_id: str, fn: SaveFn) -> SessionSnapshot | None:
    """Run a save mutator and stamp the result under ``snapshot_id``, or None to skip.

    When ``existing`` is None this creates under the reserved id. Mutators that
    only update (abort, heartbeat) return None when ``existing`` is missing.

    Terminal statuses are absorbing: once a snapshot is completed, failed, or
    aborted, a mutation may still rewrite its state or metadata, but changing
    its status raises ``FAILED_PRECONDITION`` — an aborted run must never be
    rewritten into a completed one.
    """
    next_snapshot = fn(existing.model_copy(deep=True) if existing is not None else None)
    if inspect.iscoroutine(next_snapshot):
        next_snapshot.close()
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=("save mutator must be a synchronous function; got a coroutine — remove 'async' from the mutator"),
        )
    if next_snapshot is None:
        return None
    if (
        existing is not None
        and existing.session_id
        and next_snapshot.session_id
        and existing.session_id != next_snapshot.session_id
    ):
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Snapshot '{snapshot_id}' belongs to session '{existing.session_id}' "
                f"and cannot be rewritten by session '{next_snapshot.session_id}'. "
                'Snapshot ids share one namespace per store prefix; use ids that are '
                'unique across sessions.'
            ),
        )
    if existing is not None and existing.status in TERMINAL_STATUSES and next_snapshot.status != existing.status:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                f"Snapshot '{snapshot_id}' already has terminal status "
                f"'{existing.status and existing.status.value}' and cannot transition to "
                f"'{next_snapshot.status and next_snapshot.status.value}'."
            ),
        )
    stamp_store_fields(snapshot=next_snapshot, snapshot_id=snapshot_id)
    return next_snapshot


def notify(*, subs: Subs, snapshot_id: str, status: SnapshotStatus | None) -> None:
    """Push a status change to everyone subscribed to a snapshot.

    A persisted terminal status is followed by the end-of-stream sentinel so
    consumers can finish their ``async for`` instead of waiting forever.
    """
    # Subscriber queues are unbounded, so put_nowait can't fail here.
    for q in subs.get(snapshot_id, []):
        q.put_nowait(status)
        if status in TERMINAL_STATUSES:
            q.put_nowait(None)


async def _status_queue_values(q: asyncio.Queue[SnapshotStatus | None]) -> AsyncGenerator[SnapshotStatus, None]:
    """Yield statuses until the end-of-stream sentinel."""
    while True:
        status = await q.get()
        if status is None:
            return
        yield status


class _StatusStream:
    """Closable async iterator over a status queue.

    Follow-until-done (like a channel range / callback subscription): consume
    with ``async for``, and call ``aclose()`` (or use ``contextlib.aclosing``)
    to unsubscribe. Python does not tear down async generators on ``break``,
    so teardown is explicit — same idea as an unsubscribe handle.
    """

    def __init__(
        self,
        q: asyncio.Queue[SnapshotStatus | None],
        *,
        on_close: OnClose | None = None,
    ) -> None:
        self._agen: AsyncGenerator[SnapshotStatus, None] = _status_queue_values(q)
        self._on_close = on_close
        self._closed = False

    def __aiter__(self) -> SnapshotStatusStream:
        return self

    async def __anext__(self) -> SnapshotStatus:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._agen.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._agen.aclose()
        if self._on_close is not None:
            await self._on_close()


def iterate_statuses(
    q: asyncio.Queue[SnapshotStatus | None],
    *,
    on_close: OnClose | None = None,
) -> SnapshotStatusStream:
    """Expose a status queue as a closable async stream."""
    return _StatusStream(q, on_close=on_close)


async def subscribe(
    *,
    subs: Subs,
    snapshot_id: str,
    current: SessionSnapshot | None,
) -> asyncio.Queue[SnapshotStatus | None]:
    """Register a status-change queue, seeding it with the current status.

    A missing snapshot stays subscribed with an empty queue so a later create
    (or realtime watch) can deliver the first status. ``None`` is reserved as
    the end-of-stream sentinel after a terminal status, not as "not found".
    If the current status is already terminal, seed that status and then
    ``None`` so the subscriber does not block waiting for a future event.
    """
    q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
    if current is not None:
        await q.put(current.status)
        if current.status in TERMINAL_STATUSES:
            await q.put(None)
    subs.setdefault(snapshot_id, []).append(q)
    return q
