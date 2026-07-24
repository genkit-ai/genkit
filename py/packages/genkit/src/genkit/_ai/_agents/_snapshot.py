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

"""Snapshot read/abort helpers shared by agents, transports, and registered actions."""

from __future__ import annotations

from datetime import datetime, timezone

from genkit._ai._agents._session import SessionStore
from genkit._ai._agents._types import StateTransform
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

DEFAULT_HEARTBEAT_TIMEOUT_MS = 60_000


async def walk_back_to_resumable(
    *,
    store: SessionStore,
    snapshot: SessionSnapshot | None,
) -> SessionSnapshot | None:
    """Falls back from a session leaf to the last resumable (completed) snapshot.

    A session's newest snapshot can be a failed, aborted, or still-pending turn,
    and none of those are a place you can pick the conversation back up from. So
    a non-completed leaf walks its parent chain back to the last good turn —
    landing a reload on the same spot a live chat would resume from, instead of a
    dead handle. A visited set guards a corrupt or cyclic chain.
    """
    visited: set[str] = set()
    while snapshot is not None and snapshot.status != SnapshotStatus.COMPLETED:
        if snapshot.snapshot_id in visited:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    f'Snapshot parent chain for {snapshot.snapshot_id!r} is cyclic '
                    '(a snapshot was visited twice). Resume by snapshot_id instead.'
                ),
            )
        visited.add(snapshot.snapshot_id)
        snapshot = await store.get_snapshot(snapshot_id=snapshot.parent_id) if snapshot.parent_id else None
    return snapshot


def parse_snapshot_lookup_kw(
    *,
    snapshot_id: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Require exactly one of ``snapshot_id`` or ``session_id``.

    A bad selector is a caller mistake, so it raises ``INVALID_ARGUMENT`` — over a
    transport that surfaces as a 400, not a 500 the way a bare ``ValueError`` would.
    """
    if bool(snapshot_id) == bool(session_id):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                "get_snapshot requires exactly one of 'snapshot_id' or 'session_id' "
                f'(got snapshot_id={snapshot_id!r}, session_id={session_id!r}).'
            ),
        )
    return snapshot_id, session_id


def lookup_label(*, snapshot_id: str | None = None, session_id: str | None = None) -> str:
    if snapshot_id:
        return snapshot_id
    assert session_id is not None
    return f'session {session_id}'


def is_heartbeat_expired(
    snapshot: SessionSnapshot,
    *,
    timeout_ms: int = DEFAULT_HEARTBEAT_TIMEOUT_MS,
) -> bool:
    if snapshot.status != SnapshotStatus.PENDING or not snapshot.heartbeat_at:
        return False
    try:
        # 3.10's fromisoformat rejects the 'Z' UTC suffix, so normalize it first.
        last = datetime.fromisoformat(snapshot.heartbeat_at.replace('Z', '+00:00'))
    except ValueError:
        # Can't read the timestamp, so don't declare the turn dead: expiring flips
        # a pending turn to EXPIRED, and we'd rather leave a live turn alone than
        # kill it over a garbled heartbeat.
        return False
    age_ms = (datetime.now(timezone.utc) - last).total_seconds() * 1000
    return age_ms > timeout_ms


def to_client_snapshot(
    *,
    snapshot: SessionSnapshot,
    state_transform: StateTransform | None,
) -> SessionSnapshot:
    if state_transform is None or snapshot.state is None:
        return snapshot
    transformed = state_transform(snapshot.state)
    if transformed is snapshot.state:
        return snapshot
    # Only this outbound copy is reshaped; the stored snapshot is untouched.
    return snapshot.model_copy(update={'state': transformed})


async def resolve_snapshot(
    *,
    store: SessionStore,
    snapshot_id: str | None = None,
    session_id: str | None = None,
    state_transform: StateTransform | None = None,
) -> SessionSnapshot | None:
    snapshot_id, session_id = parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)
    if snapshot_id is not None:
        snapshot = await store.get_snapshot(snapshot_id=snapshot_id)
    else:
        assert session_id is not None
        # Resolving a session means "where do I continue from", so skip a
        # failed/aborted/pending leaf back to the last resumable turn.
        snapshot = await walk_back_to_resumable(store=store, snapshot=await store.get_snapshot(session_id=session_id))
    if snapshot is None:
        return None
    effective = (
        snapshot.model_copy(update={'status': SnapshotStatus.EXPIRED}) if is_heartbeat_expired(snapshot) else snapshot
    )
    return to_client_snapshot(snapshot=effective, state_transform=state_transform)


def abort_if_pending(existing: SessionSnapshot | None) -> SessionSnapshot | None:
    """save_snapshot mutator: flip a still-pending snapshot to aborted, else skip."""
    if existing is None or existing.status != SnapshotStatus.PENDING:
        return None
    return existing.model_copy(update={'status': SnapshotStatus.ABORTED})


async def abort_snapshot_in_store(*, store: SessionStore, snapshot_id: str) -> SnapshotStatus | None:
    """Abort a running snapshot by flipping it to aborted.

    There's no dedicated store abort call: aborting is an ordinary atomic
    snapshot write whose mutator flips a still-pending turn to aborted and leaves
    an already-finished one untouched, so a late abort never rewrites a
    completed/failed result. The write also notifies any status subscribers,
    which is how a detached turn learns it was aborted. Returns the snapshot's
    resulting status (aborted when this call did the flip), or None if it doesn't
    exist.
    """
    saved = await store.save_snapshot(snapshot_id, abort_if_pending)
    if saved is not None:
        return saved.status
    # The mutator skipped the write: either the snapshot is gone or already
    # terminal. Report its current status without touching it.
    current = await store.get_snapshot(snapshot_id=snapshot_id)
    return current.status if current is not None else None
