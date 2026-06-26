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

from genkit._ai._agents._session import SessionStore, SnapshotAborter
from genkit._ai._agents._types import ClientTransform
from genkit._core._typing import SessionSnapshot, SnapshotStatus

DEFAULT_HEARTBEAT_TIMEOUT_MS = 60_000


def parse_snapshot_lookup_kw(
    *,
    snapshot_id: str | None = None,
    session_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Require exactly one of ``snapshot_id`` or ``session_id``."""
    if bool(snapshot_id) == bool(session_id):
        raise ValueError(
            "get_snapshot requires exactly one of 'snapshot_id' or 'session_id' "
            f'(got snapshot_id={snapshot_id!r}, session_id={session_id!r}).'
        )
    return snapshot_id, session_id


def lookup_label(*, snapshot_id: str | None = None, session_id: str | None = None) -> str:
    if snapshot_id:
        return snapshot_id
    assert session_id is not None
    return f'session {session_id}'


def _snapshot_id_from_input(input_val: Any) -> str | None:  # noqa: ANN401
    if isinstance(input_val, str):
        return input_val
    if isinstance(input_val, dict):
        val = input_val.get('snapshotId') or input_val.get('snapshot_id')
        return val if isinstance(val, str) else None
    val = getattr(input_val, 'snapshotId', None) or getattr(input_val, 'snapshot_id', None)
    return val if isinstance(val, str) else None


def parse_snapshot_lookup_input(input_val: Any) -> tuple[str | None, str | None]:  # noqa: ANN401
    """Parse getSnapshot action input (camelCase or snake_case)."""
    snapshot_id: str | None = None
    session_id: str | None = None
    if isinstance(input_val, dict):
        snapshot_id = input_val.get('snapshotId') or input_val.get('snapshot_id')
        session_id = input_val.get('sessionId') or input_val.get('session_id')
    else:
        snapshot_id = getattr(input_val, 'snapshotId', None) or getattr(input_val, 'snapshot_id', None)
        session_id = getattr(input_val, 'sessionId', None) or getattr(input_val, 'session_id', None)
    if isinstance(snapshot_id, str):
        snapshot_id = snapshot_id or None
    else:
        snapshot_id = None
    if isinstance(session_id, str):
        session_id = session_id or None
    else:
        session_id = None
    return parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)


def parse_abort_input(input_val: Any) -> str:  # noqa: ANN401
    """Parse abort action input; returns snapshot_id."""
    snapshot_id = _snapshot_id_from_input(input_val)
    if snapshot_id is None:
        raise ValueError(f"'snapshot_id' is required and must be a string, got {type(input_val).__name__}.")
    return snapshot_id


def is_heartbeat_expired(
    snapshot: SessionSnapshot,
    timeout_ms: int = DEFAULT_HEARTBEAT_TIMEOUT_MS,
) -> bool:
    if snapshot.status != SnapshotStatus.PENDING or not snapshot.heartbeat_at:
        return False
    try:
        last = datetime.fromisoformat(snapshot.heartbeat_at.replace('Z', '+00:00'))
    except ValueError:
        return False
    age_ms = (datetime.now(timezone.utc) - last).total_seconds() * 1000
    return age_ms > timeout_ms


def to_client_snapshot(
    snapshot: SessionSnapshot,
    client_transform: ClientTransform | None,
) -> SessionSnapshot:
    state_fn = client_transform.get('state') if client_transform else None
    if state_fn is None or snapshot.state is None:
        return snapshot
    transformed = state_fn(snapshot.state)
    if transformed is snapshot.state:
        return snapshot
    return snapshot.model_copy(update={'state': transformed})


async def resolve_snapshot(
    store: SessionStore,
    *,
    snapshot_id: str | None = None,
    session_id: str | None = None,
    client_transform: ClientTransform | None = None,
) -> SessionSnapshot | None:
    snapshot_id, session_id = parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)
    if snapshot_id is not None:
        snapshot = await store.get_snapshot(snapshot_id=snapshot_id)
    else:
        assert session_id is not None
        snapshot = await store.get_snapshot(session_id=session_id)
    if snapshot is None:
        return None
    effective = (
        snapshot.model_copy(update={'status': SnapshotStatus.EXPIRED}) if is_heartbeat_expired(snapshot) else snapshot
    )
    return to_client_snapshot(effective, client_transform)


async def run_abort(store: SessionStore, snapshot_id: str) -> SnapshotStatus | None:
    if not isinstance(store, SnapshotAborter):
        return None
    return await store.abort_snapshot(snapshot_id)
