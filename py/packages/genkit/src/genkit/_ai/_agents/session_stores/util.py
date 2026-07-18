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

"""Storage-agnostic helpers shared by the flat, full-snapshot session stores.

Each turn is persisted whole, keyed by its snapshot id, and the ``parent_id``
links between snapshots form the conversation tree. That single shape covers a
linear chat, a "just give me the latest turn" lookup, and a forked/branching
history without any diffing — so it's the right default for local dev, tests,
and single-process apps. For a multi-instance production deployment, back the
same ``SessionStore`` protocol with a real database (where it's worth trading
this simplicity for incremental, diff-based persistence).

``InMemorySessionStore`` and ``FileSessionStore`` are standalone — they share
the bits here as plain functions rather than a common base class, so each store
is a self-contained read of how its backend works.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from genkit._ai._agents._session import select_leaf_snapshot
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
