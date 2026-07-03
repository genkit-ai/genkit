# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Firestore-backed session store for agent snapshots.

Layout mirrors the JS ``FirestoreSessionStore`` collection structure (snapshots +
pointers under a per-tenant prefix). This Python implementation stores each
snapshot as a full document for now — diff/checkpoint sharding can land later
for very long sessions.

Paths (default collection ``genkit-sessions``, prefix ``global``):

  genkit-sessions/{prefix}/snapshots/{snapshotId}
  genkit-sessions-pointers/{prefix}/pointers/{sessionId}
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from genkit._ai._agents._session import SessionStore, SnapshotSubscriber
from genkit._ai._agents._session_stores import (
    SaveFn,
    Subs,
    _apply_save,
    _notify,
    _require_one_selector,
    _select_leaf,
    _session_id_of,
    _subscribe,
)
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

StateT = TypeVar('StateT')

DEFAULT_COLLECTION = 'genkit-sessions'
DEFAULT_PREFIX = 'global'


def _sanitize(value: Any) -> Any:  # noqa: ANN401
    """Drop None members so Firestore accepts the payload."""
    return json.loads(json.dumps(value, default=str))


class FirestoreSessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """Persist agent snapshots in Cloud Firestore.

    Uses Application Default Credentials (or ``FIRESTORE_EMULATOR_HOST`` for
    the emulator). Pass ``snapshot_path_prefix`` to isolate tenants when session
    ids may collide across users.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,  # noqa: ANN401 — google.cloud.firestore.Client
        collection: str = DEFAULT_COLLECTION,
        snapshot_path_prefix: Callable[[], str] | None = None,
        reject_ambiguous_session: bool = False,
    ) -> None:
        """Initialize the Firestore session store."""
        if client is None:
            from google.cloud import firestore  # noqa: PLC0415

            client = firestore.Client()
        self._client = client
        self._collection = collection
        self._prefix_fn = snapshot_path_prefix or (lambda: DEFAULT_PREFIX)
        self._reject_ambiguous = reject_ambiguous_session
        self._lock = asyncio.Lock()
        self._subs: Subs = {}

    def _prefix(self) -> str:
        return self._prefix_fn()

    def _snapshots_col(self) -> Any:  # noqa: ANN401
        return self._client.collection(self._collection).document(self._prefix()).collection('snapshots')

    def _pointers_col(self) -> Any:  # noqa: ANN401
        return self._client.collection(f'{self._collection}-pointers').document(self._prefix()).collection('pointers')

    def _snapshot_ref(self, snapshot_id: str) -> Any:  # noqa: ANN401
        return self._snapshots_col().document(snapshot_id)

    def _pointer_ref(self, session_id: str) -> Any:  # noqa: ANN401
        return self._pointers_col().document(session_id)

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Retrieve a session snapshot by snapshot ID or session ID."""
        _require_one_selector(snapshot_id, session_id)
        if snapshot_id is not None:
            snap = await asyncio.to_thread(self._read_snapshot_sync, snapshot_id)
            return snap.model_copy(deep=True) if snap is not None else None

        assert session_id is not None
        pointer = await asyncio.to_thread(self._read_pointer_sync, session_id)
        if pointer is not None:
            snap = await asyncio.to_thread(self._read_snapshot_sync, pointer['currentSnapshotId'])
            return snap.model_copy(deep=True) if snap is not None else None

        # Fallback: scan snapshots for this session (dev/small deployments).
        owned = await asyncio.to_thread(self._list_session_snapshots_sync, session_id)
        leaf = _select_leaf(owned, session_id, reject_ambiguous=self._reject_ambiguous)
        return leaf.model_copy(deep=True) if leaf is not None else None

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
        """Save or update a session snapshot in Firestore."""
        async with self._lock:
            existing = (
                await asyncio.to_thread(self._read_snapshot_sync, snapshot_id) if snapshot_id is not None else None
            )
            next_snapshot = _apply_save(existing, snapshot_id, fn)
            if next_snapshot is None:
                return None

            sid = next_snapshot.snapshot_id
            session_id = _session_id_of(next_snapshot)
            if not session_id:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message="FirestoreSessionStore requires 'sessionId' on the snapshot.",
                )

            await asyncio.to_thread(self._write_snapshot_sync, next_snapshot)
            await asyncio.to_thread(
                self._maybe_update_pointer_sync,
                session_id,
                sid,
                is_new=existing is None,
            )
            _notify(self._subs, sid, next_snapshot.status)
            return next_snapshot

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Subscribe to status changes for a session snapshot."""
        async with self._lock:
            current = await asyncio.to_thread(self._read_snapshot_sync, snapshot_id)
            q = await _subscribe(self._subs, snapshot_id, current)
        # Firestore listener keeps cross-instance abort working for detached turns.
        self._start_listener(snapshot_id, q)
        return q

    def _read_snapshot_sync(self, snapshot_id: str) -> SessionSnapshot | None:
        doc = self._snapshot_ref(snapshot_id).get()
        if not doc.exists:
            return None
        return SessionSnapshot.model_validate(doc.to_dict())

    def _write_snapshot_sync(self, snapshot: SessionSnapshot) -> None:
        assert snapshot.snapshot_id is not None
        self._snapshot_ref(snapshot.snapshot_id).set(_sanitize(snapshot.model_dump(by_alias=True, exclude_none=True)))

    def _read_pointer_sync(self, session_id: str) -> dict[str, str] | None:
        doc = self._pointer_ref(session_id).get()
        if not doc.exists:
            return None
        data = doc.to_dict() or {}
        current = data.get('currentSnapshotId')
        return {'currentSnapshotId': current} if current else None

    def _maybe_update_pointer_sync(self, session_id: str, snapshot_id: str, *, is_new: bool) -> None:
        ref = self._pointer_ref(session_id)
        existing = ref.get()
        pointer = existing.to_dict() if existing.exists else None
        if is_new or not pointer or pointer.get('currentSnapshotId') == snapshot_id:
            ref.set(
                _sanitize({
                    'currentSnapshotId': snapshot_id,
                    'updatedAt': datetime.now(timezone.utc).isoformat(),
                })
            )

    def _list_session_snapshots_sync(self, session_id: str) -> list[SessionSnapshot]:
        # Prefix scan via collection listing — fine for moderate session counts.
        snaps: list[SessionSnapshot] = []
        for doc in self._snapshots_col().stream():
            try:
                snap = SessionSnapshot.model_validate(doc.to_dict())
            except (ValueError, TypeError):
                continue
            if _session_id_of(snap) == session_id:
                snaps.append(snap)
        return snaps

    def _start_listener(self, snapshot_id: str, q: asyncio.Queue[SnapshotStatus | None]) -> None:
        ref = self._snapshot_ref(snapshot_id)
        loop = asyncio.get_event_loop()

        def on_snapshot(doc_snapshot: Any) -> None:  # noqa: ANN401
            if not doc_snapshot.exists:
                return
            data = doc_snapshot.to_dict() or {}
            status_val = data.get('status')
            if status_val is None:
                return
            try:
                status = SnapshotStatus(status_val)
            except ValueError:
                return
            loop.call_soon_threadsafe(lambda: _notify(self._subs, snapshot_id, status))

        ref.on_snapshot(on_snapshot)
