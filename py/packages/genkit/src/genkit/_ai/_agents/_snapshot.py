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
from typing import Any

from genkit._ai._agents._session import SessionStore
from genkit._ai._agents._types import StateTransform
from genkit._core._action import get_current_context
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

DEFAULT_HEARTBEAT_TIMEOUT_MS = 60_000


async def walk_back_to_resumable(
    *,
    store: SessionStore,
    snapshot: SessionSnapshot | None,
) -> SessionSnapshot | None:
    """Skip a failed / aborted / pending leaf back to the last completed snapshot.

    That's the only kind of row you can continue a conversation from. If the
    parent chain loops, we fail instead of reading forever. Parent hops use
    the ambient request context so tenant-scoped stores keep the caller's auth.
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
        snapshot = (
            await store.get_snapshot(
                snapshot_id=snapshot.parent_id,
                context=get_current_context(),
            )
            if snapshot.parent_id
            else None
        )
    return snapshot


def parse_snapshot_lookup_kw(
    *,
    snapshot_id: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Require exactly one of ``snapshot_id`` or ``session_id``.

    A bad selector is a caller mistake, so it raises ``INVALID_ARGUMENT`` — over a
    transport that surfaces as a 400, not a 500 the way a bare ``ValueError`` would.
    Whitespace-only ids count as empty: they are not usable as document keys.
    """
    for name, value in (('snapshot_id', snapshot_id), ('session_id', session_id)):
        if value is not None and not value.strip():
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{name} must not be empty or whitespace-only.',
            )
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
    context: dict[str, Any] | None = None,
) -> SessionSnapshot | None:
    snapshot_id, session_id = parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)
    if snapshot_id is not None:
        snapshot = await store.get_snapshot(snapshot_id=snapshot_id, context=context)
    else:
        assert session_id is not None
        # Return the stored leaf as-is — including a failed, aborted, or
        # still-pending turn — so you can see why the last turn died.
        # chat(session_id=) / load_session skip a dead leaf separately.
        snapshot = await store.get_snapshot(session_id=session_id, context=context)
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


class _AbortRecorder:
    """save_snapshot mutator for abort that also records the row it saw.

    Abort reports the turn's prior status (was it still running, or already
    finished?), and the only trustworthy source for that is the row the winning
    write actually observed — a separate read could see a newer state. Stores
    may invoke the mutator more than once under contention; the last
    observation wins, matching what got committed.
    """

    def __init__(self) -> None:
        self.previous: SnapshotStatus | None = None

    def __call__(self, existing: SessionSnapshot | None) -> SessionSnapshot | None:
        if existing is not None:
            self.previous = existing.status
        return abort_if_pending(existing)


async def abort_snapshot_in_store(
    *,
    store: SessionStore,
    snapshot_id: str,
    context: dict[str, Any] | None = None,
) -> SnapshotStatus | None:
    """Abort a running snapshot by flipping it to aborted.

    There's no separate abort API on the store. Aborting is an ordinary atomic
    snapshot write: the mutator flips a still-pending turn to aborted and leaves
    an already-finished one untouched, so a late abort never overwrites a
    completed or failed result. The write also notifies status subscribers,
    which is how a detached turn learns it was aborted.

    Returns the last status the mutator saw on the existing row — typically
    ``pending`` when this call cancelled in-flight work, or the unchanged
    terminal status if the turn had already finished. ``None`` if the mutator
    never ran (missing row, or the store skipped the write). We do not re-read
    the row afterward to invent a previous status: if the mutator never ran,
    there is nothing to report.
    """
    recorder = _AbortRecorder()
    await store.save_snapshot(snapshot_id, recorder, context=context)
    return recorder.previous
