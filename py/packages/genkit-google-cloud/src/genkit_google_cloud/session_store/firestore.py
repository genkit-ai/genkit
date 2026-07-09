# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Firestore-backed session store for agent snapshots.

Each snapshot is stored as a full document under a per-tenant prefix.
Diff/checkpoint sharding can land later for very long sessions.

Paths (default collection ``genkit-sessions``, prefix ``global``):

  genkit-sessions/{prefix}/snapshots/{snapshotId}
  genkit-sessions-pointers/{prefix}/pointers/{sessionId}
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from google.cloud import firestore
from google.cloud.firestore import AsyncClient

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

TERMINAL_STATUSES = frozenset({
    SnapshotStatus.COMPLETED,
    SnapshotStatus.FAILED,
    SnapshotStatus.ABORTED,
    SnapshotStatus.EXPIRED,
})


def copy_snapshot(snapshot: SessionSnapshot | None) -> SessionSnapshot | None:
    """Return a deep copy of the session snapshot if present."""
    return snapshot.model_copy(deep=True) if snapshot is not None else None


def status_from_doc(doc_snapshot: Any) -> SnapshotStatus | None:  # noqa: ANN401
    """Extract and validate the snapshot status from a Firestore document."""
    if not doc_snapshot.exists:
        return None
    status_val = (doc_snapshot.to_dict() or {}).get('status')
    if status_val is None:
        return None
    try:
        out: Any = SnapshotStatus(status_val)
    except ValueError:
        return None
    return out


class FirestoreSessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """Persist agent snapshots in Cloud Firestore.

    Uses Application Default Credentials (or ``FIRESTORE_EMULATOR_HOST`` for
    the emulator). Pass ``snapshot_path_prefix`` to isolate tenants when session
    ids may collide across users.
    """

    def __init__(
        self,
        *,
        client: AsyncClient | None = None,
        collection: str = DEFAULT_COLLECTION,
        snapshot_path_prefix: Callable[[], str] | None = None,
        reject_ambiguous_session: bool = False,
    ) -> None:
        """Initialize the Firestore session store."""
        self._client = client or firestore.AsyncClient()
        self._collection = collection
        self._prefix_fn = snapshot_path_prefix or (lambda: DEFAULT_PREFIX)
        self._reject_ambiguous = reject_ambiguous_session
        self._lock = asyncio.Lock()
        self._subs: Subs = {}
        self._sync_client: firestore.Client | None = None

    def _snapshots_col(self) -> Any:  # noqa: ANN401
        prefix = self._prefix_fn()
        return self._client.collection(self._collection).document(prefix).collection('snapshots')

    def _pointers_col(self) -> Any:  # noqa: ANN401
        prefix = self._prefix_fn()
        return self._client.collection(f'{self._collection}-pointers').document(prefix).collection('pointers')

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
            return copy_snapshot(await self._read_snapshot(snapshot_id))

        assert session_id is not None
        current_id = await self._read_pointer(session_id)
        if current_id is not None:
            return copy_snapshot(await self._read_snapshot(current_id))

        owned = await self._list_session_snapshots(session_id)
        return copy_snapshot(_select_leaf(owned, session_id, reject_ambiguous=self._reject_ambiguous))

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
        """Save or update a session snapshot in Firestore."""
        async with self._lock:
            existing = await self._read_snapshot(snapshot_id) if snapshot_id is not None else None
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

            await self._write_snapshot(next_snapshot)
            await self._maybe_update_pointer(session_id, sid, is_new=existing is None)
            _notify(self._subs, sid, next_snapshot.status)
            return next_snapshot

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Subscribe to status changes for a session snapshot."""
        async with self._lock:
            current = await self._read_snapshot(snapshot_id)
            is_first = snapshot_id not in self._subs
            q = await _subscribe(self._subs, snapshot_id, current)
            # Firestore listener keeps cross-instance abort working for detached turns.
            if is_first:
                self._start_listener(snapshot_id)
        return q

    async def _read_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        doc = await self._snapshot_ref(snapshot_id).get()
        if not doc.exists:
            return None
        return SessionSnapshot.model_validate(doc.to_dict())

    async def _write_snapshot(self, snapshot: SessionSnapshot) -> None:
        assert snapshot.snapshot_id is not None
        payload = snapshot.model_dump(by_alias=True, exclude_none=True, mode='json')
        await self._snapshot_ref(snapshot.snapshot_id).set(payload)

    async def _read_pointer(self, session_id: str) -> str | None:
        doc = await self._pointer_ref(session_id).get()
        if not doc.exists:
            return None
        current = (doc.to_dict() or {}).get('currentSnapshotId')
        return current if isinstance(current, str) else None

    async def _maybe_update_pointer(self, session_id: str, snapshot_id: str, *, is_new: bool) -> None:
        ref = self._pointer_ref(session_id)
        transaction = self._client.transaction()

        @firestore.async_transactional
        async def update_in_transaction(transaction: Any) -> None:  # noqa: ANN401
            snapshot = await ref.get(transaction=transaction)
            pointer = snapshot.to_dict() if snapshot.exists else None
            if is_new or not pointer or pointer.get('currentSnapshotId') == snapshot_id:
                transaction.set(
                    ref,
                    {
                        'currentSnapshotId': snapshot_id,
                        'updatedAt': firestore.SERVER_TIMESTAMP,
                    },
                )

        await update_in_transaction(transaction)

    async def _list_session_snapshots(self, session_id: str) -> list[SessionSnapshot]:
        snaps: list[SessionSnapshot] = []
        async for doc in self._snapshots_col().where('sessionId', '==', session_id).stream():
            try:
                snaps.append(SessionSnapshot.model_validate(doc.to_dict()))
            except (ValueError, TypeError):
                continue
        return snaps

    def _start_listener(self, snapshot_id: str) -> None:
        ref = self._snapshot_ref(snapshot_id)
        if not hasattr(ref, 'on_snapshot'):
            if self._sync_client is None:
                self._sync_client = firestore.Client(
                    project=self._client.project,
                    credentials=getattr(self._client, '_credentials', None),
                    database=getattr(self._client, '_database', getattr(self._client, 'database', None)),
                )
            prefix = self._prefix_fn()
            ref = (
                self._sync_client
                .collection(self._collection)
                .document(prefix)
                .collection('snapshots')
                .document(snapshot_id)
            )
        loop = asyncio.get_running_loop()
        watch_holder: list[Any] = []

        def on_snapshot(doc_snapshots: list[Any], changes: Any, read_time: Any) -> None:  # noqa: ANN401
            if not doc_snapshots:
                return
            doc_snapshot = doc_snapshots[0]
            status = status_from_doc(doc_snapshot)
            if status is None:
                return
            loop.call_soon_threadsafe(lambda: _notify(self._subs, snapshot_id, status))
            if status not in TERMINAL_STATUSES:
                return

            loop.call_soon_threadsafe(lambda: _notify(self._subs, snapshot_id, None))

            async def cleanup() -> None:
                async with self._lock:
                    self._subs.pop(snapshot_id, None)

            asyncio.run_coroutine_threadsafe(cleanup(), loop)

            def unsubscribe_safely() -> None:
                if watch_holder:
                    watch_holder[0].unsubscribe()

            loop.call_soon_threadsafe(unsubscribe_safely)

        watch_holder.append(ref.on_snapshot(on_snapshot))
