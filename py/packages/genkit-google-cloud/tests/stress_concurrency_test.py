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

"""Stress and concurrency test suite for FirestoreSessionStore."""

from __future__ import annotations

import asyncio
import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_google_cloud import FirestoreSessionStore
from google.api_core.exceptions import Aborted
from google.cloud import firestore

from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SessionState, SnapshotStatus


def _doc(
    *,
    path: str,
    exists: bool = True,
    data: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.exists = exists
    snap.id = doc_id or path.rsplit('/', 1)[-1]
    snap.reference.path = path
    snap.to_dict.return_value = data
    return snap


class TransactionalFakeStoreHarness:
    """In-memory Firestore stand-in with optimistic concurrency control (OCC).

    Tracks document versions. If a document read within a transaction is updated
    by another transaction before commit, the commit raises google.api_core.exceptions.Aborted,
    causing @firestore.async_transactional to retry the transaction.
    """

    def __init__(self, simulate_delays: bool = True) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.doc_versions: dict[str, int] = {}
        self.deleted: list[str] = []
        self.simulate_delays = simulate_delays
        self._lock = asyncio.Lock()

        # Mock sync client for status watches
        self.sync_client = MagicMock()
        self.listeners: dict[str, list[Any]] = {}

        def sync_collection(col_name: str) -> MagicMock:
            col = MagicMock()

            def sync_document(doc_id: str) -> MagicMock:
                doc_ref = MagicMock()

                def sync_subcol(sub_name: str) -> MagicMock:
                    subcol = MagicMock()

                    def sync_item_doc(item_id: str) -> MagicMock:
                        path = f'{col_name}/{doc_id}/{sub_name}/{item_id}'
                        item_ref = MagicMock()

                        def on_snapshot(cb: Any) -> MagicMock:
                            watch = MagicMock()
                            self.listeners.setdefault(path, []).append(cb)

                            def unsubscribe() -> None:
                                if path in self.listeners and cb in self.listeners[path]:
                                    self.listeners[path].remove(cb)

                            watch.unsubscribe = unsubscribe
                            return watch

                        item_ref.on_snapshot.side_effect = on_snapshot
                        return item_ref

                    subcol.document.side_effect = sync_item_doc
                    return subcol

                doc_ref.collection.side_effect = sync_subcol
                return doc_ref

            col.document.side_effect = sync_document
            return col

        self.sync_client.collection.side_effect = sync_collection

        # Async Client and Transaction creation
        self.client = MagicMock()

        def create_transaction(read_only: bool = False, **kwargs: Any) -> MagicMock:
            txn = MagicMock()
            txn._max_attempts = 20
            txn._read_only = read_only
            txn._begin = AsyncMock()
            txn._rollback = AsyncMock()
            txn._read_versions = {}
            txn._pending_writes = {}
            txn._pending_deletes = set()

            async def txn_get_all(refs: list[Any]) -> Any:
                if self.simulate_delays:
                    await asyncio.sleep(random.uniform(0.001, 0.005))
                async with self._lock:
                    for ref in refs:
                        path = ref.path
                        ver = self.doc_versions.get(path, 0)
                        txn._read_versions[path] = ver
                        if path in self.docs:
                            yield _doc(path=path, exists=True, data=dict(self.docs[path]), doc_id=ref.id)
                        else:
                            yield _doc(path=path, exists=False, data=None, doc_id=ref.id)

            txn.get_all = txn_get_all

            def txn_set(ref: Any, data: dict[str, Any]) -> None:
                txn._pending_writes[ref.path] = dict(data)
                txn._pending_deletes.discard(ref.path)

            def txn_update(ref: Any, data: dict[str, Any]) -> None:
                current: dict[str, Any] = dict(txn._pending_writes.get(ref.path, self.docs.get(ref.path, {})))
                for key, value in data.items():
                    if value is firestore.DELETE_FIELD:
                        current.pop(key, None)
                    elif value is firestore.SERVER_TIMESTAMP:
                        current[key] = 'SERVER_TIMESTAMP'
                    else:
                        current[key] = value
                txn._pending_writes[ref.path] = current

            def txn_delete(ref: Any) -> None:
                txn._pending_deletes.add(ref.path)
                txn._pending_writes.pop(ref.path, None)

            txn.set = txn_set
            txn.update = txn_update
            txn.delete = txn_delete

            async def _commit() -> None:
                if self.simulate_delays:
                    await asyncio.sleep(random.uniform(0.001, 0.005))
                async with self._lock:
                    if not txn._read_only:
                        # Validate OCC: check if any read doc was modified since read
                        for path, read_ver in txn._read_versions.items():
                            current_ver = self.doc_versions.get(path, 0)
                            if current_ver != read_ver:
                                raise Aborted(f'Transaction aborted due to concurrent write on {path}')

                        # Commit writes
                        for path, data in txn._pending_writes.items():
                            self.docs[path] = data
                            self.doc_versions[path] = self.doc_versions.get(path, 0) + 1
                            self._trigger_listeners(path, data)

                        for path in txn._pending_deletes:
                            self.docs.pop(path, None)
                            self.doc_versions[path] = self.doc_versions.get(path, 0) + 1
                            self.deleted.append(path)
                            self._trigger_listeners(path, None)

            txn._commit = AsyncMock(side_effect=_commit)
            return txn

        self.client.transaction.side_effect = create_transaction

        def collection(name: str) -> MagicMock:
            col = MagicMock()
            col_name = name

            def document(doc_id: str) -> MagicMock:
                prefix_ref = MagicMock()

                def subcollection(sub: str) -> MagicMock:
                    sub_col = MagicMock()

                    def item_document(item_id: str) -> MagicMock:
                        path = f'{col_name}/{doc_id}/{sub}/{item_id}'
                        ref = MagicMock(spec=['get', 'path', 'id', 'collection'])
                        ref.path = path
                        ref.id = item_id

                        async def get(*, transaction: Any = None) -> MagicMock:
                            if self.simulate_delays:
                                await asyncio.sleep(random.uniform(0.001, 0.003))
                            async with self._lock:
                                if transaction is not None and hasattr(transaction, '_read_versions'):
                                    transaction._read_versions[path] = self.doc_versions.get(path, 0)
                                if path in self.docs:
                                    return _doc(path=path, exists=True, data=dict(self.docs[path]), doc_id=item_id)
                                return _doc(path=path, exists=False, data=None, doc_id=item_id)

                        ref.get = get
                        return ref

                    sub_col.document.side_effect = item_document
                    return sub_col

                prefix_ref.collection.side_effect = subcollection
                return prefix_ref

            col.document.side_effect = document
            return col

        self.client.collection.side_effect = collection

    def _trigger_listeners(self, path: str, data: dict[str, Any] | None) -> None:
        if path in self.listeners:
            doc_snap = _doc(path=path, exists=data is not None, data=data)
            for cb in list(self.listeners[path]):
                cb([doc_snap], None, None)

    def store(self, **kwargs: Any) -> FirestoreSessionStore:
        return FirestoreSessionStore(client=self.client, sync_client=self.sync_client, **kwargs)


def _snap_path(snapshot_id: str, *, prefix: str = 'global') -> str:
    return f'genkit-sessions/{prefix}/snapshots/{snapshot_id}'


def _pointer_path(session_id: str, *, prefix: str = 'global') -> str:
    return f'genkit-sessions-pointers/{prefix}/pointers/{session_id}'


# -----------------------------------------------------------------------------
# 1. Concurrent save_snapshot calls on the exact same snapshot_id
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_save_snapshot_same_snapshot_id_initial_creation() -> None:
    """Multiple tasks concurrently save a brand new snapshot under the exact same snapshot_id."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store = h.store()
    num_tasks = 10
    snapshot_id = 'snap-concurrent-init'

    async def save_worker(worker_id: int) -> SessionSnapshot | None:
        def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
            if existing is None:
                return SessionSnapshot(
                    snapshot_id=snapshot_id,
                    session_id='sess-same-id',
                    created_at='2026-08-03T00:00:00Z',
                    state=SessionState(session_id='sess-same-id', custom={'workers': [worker_id]}),
                )
            state_custom = existing.state.custom if existing.state and isinstance(existing.state.custom, dict) else {}
            workers = list(state_custom.get('workers', []))
            workers.append(worker_id)
            return existing.model_copy(
                update={'state': SessionState(session_id='sess-same-id', custom={'workers': workers})}
            )

        return await store.save_snapshot(snapshot_id, save_fn)

    results = await asyncio.gather(*(save_worker(i) for i in range(num_tasks)))

    assert all(res is not None for res in results)

    final_snap = await store.get_snapshot(snapshot_id=snapshot_id)
    assert final_snap is not None
    assert final_snap.snapshot_id == snapshot_id
    assert final_snap.state is not None

    workers_registered = final_snap.state.custom.get('workers', []) if isinstance(final_snap.state.custom, dict) else []
    assert len(workers_registered) == num_tasks
    assert set(workers_registered) == set(range(num_tasks))


@pytest.mark.asyncio
async def test_concurrent_save_snapshot_same_snapshot_id_updates() -> None:
    """Multiple tasks concurrently update an existing snapshot's state and status."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store = h.store()
    snapshot_id = 'snap-concurrent-update'

    await store.save_snapshot(
        snapshot_id,
        lambda _e: SessionSnapshot(
            snapshot_id=snapshot_id,
            session_id='sess-same-id-update',
            created_at='2026-08-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-same-id-update', custom={'counter': 0}),
        ),
    )

    num_tasks = 15

    async def increment_counter(task_id: int) -> SessionSnapshot | None:
        def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
            assert existing is not None and existing.state is not None and isinstance(existing.state.custom, dict)
            current_val = existing.state.custom.get('counter', 0)
            return existing.model_copy(
                update={
                    'state': SessionState(
                        session_id='sess-same-id-update',
                        custom={'counter': current_val + 1},
                    )
                }
            )

        return await store.save_snapshot(snapshot_id, save_fn)

    results = await asyncio.gather(*(increment_counter(i) for i in range(num_tasks)))
    assert all(res is not None for res in results)

    final_snap = await store.get_snapshot(snapshot_id=snapshot_id)
    assert final_snap is not None
    assert final_snap.state is not None
    assert isinstance(final_snap.state.custom, dict)
    assert final_snap.state.custom.get('counter') == num_tasks


# -----------------------------------------------------------------------------
# 2. Concurrent save_snapshot calls on exact same session_id creating branching turns
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_save_snapshot_branching_session() -> None:
    """Multiple tasks concurrently save new child snapshots off the same root snapshot, creating branching turns."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store_reject = h.store(reject_ambiguous_session=True)
    session_id = 'sess-branching-stress'
    root_id = 'snap-root-branch'

    await store_reject.save_snapshot(
        root_id,
        lambda _e: SessionSnapshot(
            snapshot_id=root_id,
            session_id=session_id,
            created_at='2026-08-03T00:00:00Z',
            state=SessionState(session_id=session_id, custom={'branch': 'root'}),
        ),
    )

    num_branches = 10

    async def create_branch(branch_idx: int) -> SessionSnapshot | None:
        branch_snap_id = f'snap-branch-{branch_idx}'
        timestamp = f'2026-08-03T00:00:{branch_idx + 1:02d}Z'
        return await store_reject.save_snapshot(
            branch_snap_id,
            lambda _e: SessionSnapshot(
                snapshot_id=branch_snap_id,
                parent_id=root_id,
                session_id=session_id,
                created_at=timestamp,
                state=SessionState(session_id=session_id, custom={'branch': f'branch-{branch_idx}'}),
            ),
        )

    branch_results = await asyncio.gather(*(create_branch(i) for i in range(num_branches)))
    assert all(res is not None for res in branch_results)

    pointer_doc = h.docs[_pointer_path(session_id)]
    assert pointer_doc['isAmbiguous'] is True
    assert len(pointer_doc['leaves']) == num_branches
    expected_leaves = {f'snap-branch-{i}' for i in range(num_branches)}
    assert set(pointer_doc['leaves'].keys()) == expected_leaves

    with pytest.raises(GenkitError) as exc_info:
        await store_reject.get_snapshot(session_id=session_id)
    assert exc_info.value.status == 'FAILED_PRECONDITION'

    store_permissive = h.store(reject_ambiguous_session=False)
    resolved = await store_permissive.get_snapshot(session_id=session_id)
    assert resolved is not None
    assert resolved.snapshot_id == f'snap-branch-{num_branches - 1}'
    assert resolved.state is not None
    assert resolved.state.custom == {'branch': f'branch-{num_branches - 1}'}


@pytest.mark.asyncio
async def test_concurrent_branching_subsequent_child_resolves_one_branch() -> None:
    """Creating a child of one branch removes that branch tip from leaves while keeping remaining branches."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store = h.store(reject_ambiguous_session=False)
    session_id = 'sess-branch-subsequent'

    await store.save_snapshot(
        'root',
        lambda _e: SessionSnapshot(
            snapshot_id='root',
            session_id=session_id,
            created_at='2026-08-03T00:00:00Z',
            state=SessionState(session_id=session_id),
        ),
    )

    await asyncio.gather(
        store.save_snapshot(
            'b1',
            lambda _e: SessionSnapshot(
                snapshot_id='b1',
                parent_id='root',
                session_id=session_id,
                created_at='2026-08-03T00:00:01Z',
                state=SessionState(session_id=session_id, custom={'b': 1}),
            ),
        ),
        store.save_snapshot(
            'b2',
            lambda _e: SessionSnapshot(
                snapshot_id='b2',
                parent_id='root',
                session_id=session_id,
                created_at='2026-08-03T00:00:02Z',
                state=SessionState(session_id=session_id, custom={'b': 2}),
            ),
        ),
    )

    pointer_before = h.docs[_pointer_path(session_id)]
    assert pointer_before['isAmbiguous'] is True
    assert set(pointer_before['leaves'].keys()) == {'b1', 'b2'}

    await store.save_snapshot(
        'b1-child',
        lambda _e: SessionSnapshot(
            snapshot_id='b1-child',
            parent_id='b1',
            session_id=session_id,
            created_at='2026-08-03T00:00:03Z',
            state=SessionState(session_id=session_id, custom={'b': '1-child'}),
        ),
    )

    pointer_after = h.docs[_pointer_path(session_id)]
    assert pointer_after['isAmbiguous'] is True
    assert set(pointer_after['leaves'].keys()) == {'b2', 'b1-child'}


# -----------------------------------------------------------------------------
# 3. Concurrent read_snapshot and save_snapshot while status notifications fire
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_read_save_and_status_notifications() -> None:
    """Stress test concurrent read_snapshot, save_snapshot, and status notification delivery."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store = h.store()
    snapshot_id = 'snap-status-stream'

    await store.save_snapshot(
        snapshot_id,
        lambda _e: SessionSnapshot(
            snapshot_id=snapshot_id,
            session_id='sess-status',
            created_at='2026-08-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-status', custom={'step': 0}),
        ),
    )

    status_queue = await store.on_snapshot_status_change(snapshot_id)
    received_statuses: list[SnapshotStatus | None] = []

    async def queue_consumer() -> None:
        while True:
            st = await status_queue.get()
            received_statuses.append(st)
            if st is None:
                break

    consumer_task = asyncio.create_task(queue_consumer())

    num_updates = 10

    async def writer_task() -> None:
        for i in range(1, num_updates + 1):
            await asyncio.sleep(random.uniform(0.002, 0.01))
            status = SnapshotStatus.PENDING if i < num_updates else SnapshotStatus.COMPLETED

            def update_status_fn(
                existing: SessionSnapshot | None, step: int = i, st: SnapshotStatus = status
            ) -> SessionSnapshot:
                assert existing is not None
                return existing.model_copy(
                    update={
                        'status': st,
                        'state': SessionState(session_id='sess-status', custom={'step': step}),
                    }
                )

            await store.save_snapshot(snapshot_id, update_status_fn)

    async def reader_task() -> None:
        for _ in range(15):
            await asyncio.sleep(random.uniform(0.002, 0.008))
            snap = await store.read_snapshot(snapshot_id)
            assert snap is not None
            assert snap.snapshot_id == snapshot_id

    async def get_by_session_task() -> None:
        for _ in range(15):
            await asyncio.sleep(random.uniform(0.002, 0.008))
            snap = await store.get_snapshot(session_id='sess-status')
            assert snap is not None
            assert snap.session_id == 'sess-status'

    await asyncio.gather(writer_task(), reader_task(), get_by_session_task())
    await consumer_task

    assert received_statuses[0] == SnapshotStatus.PENDING
    assert received_statuses[-1] is None
    assert received_statuses[-2] == SnapshotStatus.COMPLETED
    assert snapshot_id not in store.subs
    assert snapshot_id not in store._watches


@pytest.mark.asyncio
async def test_concurrent_reads_and_writes_multiple_snapshots() -> None:
    """Stress test multiple distinct snapshots being read and saved simultaneously across tasks."""
    h = TransactionalFakeStoreHarness(simulate_delays=True)
    store = h.store()
    num_snapshots = 5
    updates_per_snapshot = 5

    async def snapshot_lifecycle(sid_idx: int) -> None:
        sid = f'snap-multi-{sid_idx}'
        sess_id = f'sess-multi-{sid_idx}'

        await store.save_snapshot(
            sid,
            lambda _e: SessionSnapshot(
                snapshot_id=sid,
                session_id=sess_id,
                created_at='2026-08-03T00:00:00Z',
                status=SnapshotStatus.PENDING,
                state=SessionState(session_id=sess_id, custom={'val': 0}),
            ),
        )

        for step in range(1, updates_per_snapshot + 1):

            def update_fn(existing: SessionSnapshot | None, s: int = step) -> SessionSnapshot:
                assert existing is not None
                return existing.model_copy(
                    update={
                        'status': SnapshotStatus.PENDING if s < updates_per_snapshot else SnapshotStatus.COMPLETED,
                        'state': SessionState(session_id=sess_id, custom={'val': s}),
                    }
                )

            await store.save_snapshot(sid, update_fn)
            read_back = await store.get_snapshot(snapshot_id=sid)
            assert read_back is not None
            assert read_back.snapshot_id == sid

        final_by_sess = await store.get_snapshot(session_id=sess_id)
        assert final_by_sess is not None
        assert final_by_sess.state is not None
        assert final_by_sess.state.custom == {'val': updates_per_snapshot}

    await asyncio.gather(*(snapshot_lifecycle(i) for i in range(num_snapshots)))
