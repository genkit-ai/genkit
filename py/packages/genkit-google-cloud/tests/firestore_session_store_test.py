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

"""Tests for FirestoreSessionStore."""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from genkit_google_cloud import FirestoreSessionStore
from google.api_core import exceptions as google_exceptions
from google.cloud import firestore
from google.cloud.firestore_v1._helpers import ReadAfterWriteError

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


async def _next_status(statuses: Any, timeout: float = 2.0) -> Any:  # noqa: ANN401
    """Next status from a subscription stream, bounded."""
    return await asyncio.wait_for(anext(statuses), timeout)


async def _stream_ended(statuses: Any, timeout: float = 2.0) -> bool:  # noqa: ANN401
    """True if the stream ends (StopAsyncIteration) within the bound."""
    try:
        await asyncio.wait_for(anext(statuses), timeout)
        return False
    except StopAsyncIteration:
        return True


class FakeStoreHarness:
    """In-memory Firestore stand-in wired for AsyncClient-style access.

    Transaction writes are buffered until commit (like the real client), and a
    read after any buffered write raises ``ReadAfterWriteError``. ``_commit``
    can be told to raise ``Aborted`` a fixed number of times so retry behavior
    under ``@async_transactional`` is exercisable.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.deleted: list[str] = []
        self._pending: list[tuple[str, str, dict[str, Any] | None]] = []
        self.commit_aborts_remaining = 0
        self.commit_attempts = 0
        self.commit_raises: Exception | None = None

        self.client = MagicMock(spec=firestore.AsyncClient)
        self.transaction = MagicMock()
        self.transaction._max_attempts = 5
        self.transaction._read_only = False
        self.transaction._write_pbs = []
        self.transaction._id = b'fake-txn'
        self.transaction._in_progress = False
        self.client.transaction.return_value = self.transaction

        def clean_up() -> None:
            self.transaction._write_pbs = []
            self._pending = []
            self.transaction._in_progress = False

        async def begin(*, retry_id: Any = None) -> None:  # noqa: ANN401
            self.transaction._write_pbs = []
            self._pending = []
            self.transaction._in_progress = True

        async def commit() -> None:
            self.commit_attempts += 1
            if self.commit_raises is not None:
                exc = self.commit_raises
                self.commit_raises = None
                raise exc
            if self.commit_aborts_remaining > 0:
                self.commit_aborts_remaining -= 1
                raise google_exceptions.Aborted('injected abort')
            for op, path, data in self._pending:
                if op == 'set':
                    assert data is not None
                    self.docs[path] = dict(data)
                elif op == 'update':
                    assert data is not None
                    current = dict(self.docs.get(path, {}))
                    for key, value in data.items():
                        if value is firestore.DELETE_FIELD:
                            current.pop(key, None)
                        elif value is firestore.SERVER_TIMESTAMP:
                            current[key] = 'SERVER_TIMESTAMP'
                        else:
                            current[key] = value
                    self.docs[path] = current
                elif op == 'delete':
                    self.docs.pop(path, None)
                    self.deleted.append(path)
            self.transaction._write_pbs = []
            self._pending = []
            self.transaction._in_progress = False

        async def rollback() -> None:
            self.transaction._write_pbs = []
            self._pending = []
            self.transaction._in_progress = False

        self.transaction._clean_up = clean_up
        self.transaction._begin = begin
        self.transaction._commit = commit
        self.transaction._rollback = rollback

        def collection(name: str) -> MagicMock:
            col = MagicMock()
            col_name = name

            def document(doc_id: str) -> MagicMock:
                # prefix doc
                prefix_ref = MagicMock()

                def subcollection(sub: str) -> MagicMock:
                    sub_col = MagicMock()

                    def item_document(item_id: str) -> MagicMock:
                        path = f'{col_name}/{doc_id}/{sub}/{item_id}'
                        # Match AsyncDocumentReference: no on_snapshot (sync client owns watches).
                        ref = MagicMock(spec=['get', 'path', 'id', 'collection'])
                        ref.path = path
                        ref.id = item_id

                        async def get(*, transaction: Any = None) -> MagicMock:  # noqa: ANN401
                            self._reject_read_after_write(transaction)
                            if path in self.docs:
                                return _doc(path=path, exists=True, data=self.docs[path], doc_id=item_id)
                            return _doc(path=path, exists=False, data=None, doc_id=item_id)

                        ref.get = get
                        return ref

                    sub_col.document.side_effect = item_document

                    def where(field: str, op: str, value: Any) -> MagicMock:  # noqa: ANN401
                        stream_holder = MagicMock()
                        scoped_prefix = f'{col_name}/{doc_id}/{sub}/'

                        async def stream() -> Any:  # noqa: ANN401
                            for path, data in list(self.docs.items()):
                                if not path.startswith(scoped_prefix):
                                    continue
                                if data.get(field) == value:
                                    yield _doc(path=path, data=data)

                        stream_holder.stream = MagicMock(return_value=stream())
                        return stream_holder

                    sub_col.where.side_effect = where
                    return sub_col

                prefix_ref.collection.side_effect = subcollection
                return prefix_ref

            col.document.side_effect = document
            return col

        self.client.collection.side_effect = collection

        async def get_all(refs: list[Any], transaction: Any = None, **_kwargs: Any) -> Any:  # noqa: ANN401
            self._reject_read_after_write(transaction)
            for ref in refs:
                path = ref.path
                if path in self.docs:
                    yield _doc(path=path, exists=True, data=self.docs[path], doc_id=ref.id)
                else:
                    yield _doc(path=path, exists=False, data=None, doc_id=ref.id)

        # Production reads go through AsyncClient.get_all(transaction=...); keep
        # transaction.get_all as a thin alias for older call sites/tests.
        self.client.get_all = get_all
        self.transaction.get_all = get_all

        def txn_set(ref: Any, data: dict[str, Any]) -> None:  # noqa: ANN401
            self.transaction._write_pbs.append(('set', ref.path))
            self._pending.append(('set', ref.path, dict(data)))

        def txn_update(ref: Any, data: dict[str, Any]) -> None:  # noqa: ANN401
            self.transaction._write_pbs.append(('update', ref.path))
            self._pending.append(('update', ref.path, dict(data)))

        def txn_delete(ref: Any) -> None:  # noqa: ANN401
            self.transaction._write_pbs.append(('delete', ref.path))
            self._pending.append(('delete', ref.path, None))

        self.transaction.set = txn_set
        self.transaction.update = txn_update
        self.transaction.delete = txn_delete

    def _reject_read_after_write(self, transaction: Any) -> None:  # noqa: ANN401
        if transaction is not None and len(getattr(transaction, '_write_pbs', [])) > 0:
            raise ReadAfterWriteError('Attempted read after write in a transaction.')

    def store(self, **kwargs: Any) -> FirestoreSessionStore:  # noqa: ANN401
        return FirestoreSessionStore(client=self.client, **kwargs)


def _snap_path(snapshot_id: str, *, prefix: str = 'global') -> str:
    return f'genkit-sessions/{prefix}/snapshots/{snapshot_id}'


def _pointer_path(session_id: str, *, prefix: str = 'global') -> str:
    return f'genkit-sessions-pointers/{prefix}/pointers/{session_id}'


def _shard_path(checkpoint_id: str, index: int, *, prefix: str = 'global') -> str:
    return f'genkit-sessions-shards/{prefix}/shards/{checkpoint_id}_{index}'


@pytest.mark.asyncio
async def test_firestore_session_store_save_root_writes_checkpoint_shards() -> None:
    """Root snapshot (no parent) is stored as a sharded checkpoint."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)

    def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1', messages=[]),
        )

    saved = await store.save_snapshot('snap-1', save_fn)
    assert saved is not None
    assert saved.snapshot_id == 'snap-1'

    snap_doc = h.docs[_snap_path('snap-1')]
    assert snap_doc['kind'] == 'checkpoint'
    assert snap_doc['checkpointId'] == 'snap-1'
    assert snap_doc['segmentPath'] == []
    assert 'state' not in snap_doc
    assert 'statePatch' not in snap_doc

    shard = h.docs[_shard_path('snap-1', 0)]
    assert json.loads(shard['chunk'].decode('utf-8'))['sessionId'] == 'sess-1'

    pointer = h.docs[_pointer_path('sess-1')]
    assert pointer['currentSnapshotId'] == 'snap-1'
    assert pointer['checkpointId'] == 'snap-1'


@pytest.mark.asyncio
async def test_firestore_session_store_save_child_writes_diff() -> None:
    """Child under the checkpoint interval is stored as a JSON Patch diff."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-2',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-2',
            parent_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )

    child = h.docs[_snap_path('snap-2')]
    assert child['kind'] == 'diff'
    assert child['checkpointId'] == 'snap-1'
    assert child['segmentPath'] == ['snap-2']
    assert isinstance(child['statePatch'], str) and child['statePatch']

    loaded = await store.get_snapshot(snapshot_id='snap-2')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'n': 2}
    assert loaded.parent_id == 'snap-1'


@pytest.mark.asyncio
async def test_firestore_session_store_checkpoint_interval_promotes() -> None:
    """Crossing checkpoint_interval writes a new checkpoint instead of a long diff chain."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=2)

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-2',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-2',
            parent_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )
    # segmentPath length for snap-2 is 1; next child would be length 2 >= interval → checkpoint
    await store.save_snapshot(
        'snap-3',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-3',
            parent_id='snap-2',
            session_id='sess-1',
            created_at='2026-07-03T00:00:02Z',
            state=SessionState(session_id='sess-1', custom={'n': 3}),
        ),
    )

    third = h.docs[_snap_path('snap-3')]
    assert third['kind'] == 'checkpoint'
    assert third['checkpointId'] == 'snap-3'
    assert third['segmentPath'] == []
    assert _shard_path('snap-3', 0) in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_get_by_session_id() -> None:
    """session_id lookup reconstructs from pointer checkpoint metadata."""
    h = FakeStoreHarness()
    store = h.store()

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'hello': 'world'}),
        ),
    )

    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None
    assert loaded.snapshot_id == 'snap-1'
    assert loaded.state is not None
    assert loaded.state.custom == {'hello': 'world'}


@pytest.mark.asyncio
async def test_firestore_session_store_save_skips_when_mutator_returns_none() -> None:
    """Mutator returning None must not write snapshot or pointer docs."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.ABORTED,
            state=SessionState(session_id='sess-1'),
        ),
    )
    before = dict(h.docs)
    saved = await store.save_snapshot('snap-1', lambda _existing: None)
    assert saved is None
    assert h.docs == before


@pytest.mark.asyncio
async def test_firestore_session_store_heartbeat_updates_leaf_without_rebranch() -> None:
    """In-place leaf update refreshes pointer metadata and keeps a single leaf."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )

    def beat(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'heartbeat_at': '2026-07-03T00:00:05Z'})

    saved = await store.save_snapshot('snap-1', beat)
    assert saved is not None
    assert saved.heartbeat_at == '2026-07-03T00:00:05Z'
    pointer = h.docs[_pointer_path('sess-1')]
    assert 'isAmbiguous' not in pointer


@pytest.mark.asyncio
async def test_firestore_session_store_non_leaf_update_leaves_pointer_alone() -> None:
    """Updating an ancestor must not re-add it to leaves or mark the session ambiguous."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-A',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-A',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-B',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-B',
            parent_id='snap-A',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )
    pointer_before = dict(h.docs[_pointer_path('sess-1')])

    def rewrite_a(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'status': SnapshotStatus.COMPLETED})

    saved = await store.save_snapshot('snap-A', rewrite_a)
    assert saved is not None
    pointer_after = h.docs[_pointer_path('sess-1')]
    assert pointer_after == pointer_before  # non-current write: pointer fully untouched
    assert pointer_after['currentSnapshotId'] == 'snap-B'


@pytest.mark.asyncio
async def test_firestore_session_store_branching_last_writer_wins() -> None:
    """Forked sessions never error: session lookup follows the last-written leaf.

    Matches the JS Firestore store exactly — no ambiguity detection, no
    rejection option; concurrent branch extension makes the current snapshot
    race-dependent by design. Resume by snapshot_id when a branch matters.
    """
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', None, custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    await store.save_snapshot('snap-b', _mk('snap-b', 'snap-root', custom={'n': 2}))

    # Last NEW leaf wins.
    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None and loaded.snapshot_id == 'snap-b'

    # Heartbeating the OTHER branch's leaf does not move the pointer.
    def beat(existing: SessionSnapshot | None) -> SessionSnapshot | None:
        assert existing is not None
        return existing.model_copy(update={'heartbeat_at': '2026-07-03T00:09:00Z'})

    await store.save_snapshot('snap-a', beat)
    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None and loaded.snapshot_id == 'snap-b'

    # Extending the other branch DOES: its new leaf becomes current.
    await store.save_snapshot('snap-a2', _mk('snap-a2', 'snap-a', custom={'n': 3}))
    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None and loaded.snapshot_id == 'snap-a2'
    assert 'leaves' not in h.docs[_pointer_path('sess-1')]


@pytest.mark.asyncio
async def test_firestore_session_store_missing_pointer_returns_none() -> None:
    """Without a pointer, session_id lookup is None (no collection-query repair)."""
    h = FakeStoreHarness()
    h.docs[_snap_path('snap-1')] = {
        'snapshotId': 'snap-1',
        'sessionId': 'sess-unpointed',
        'createdAt': '2026-07-03T00:00:01Z',
        'kind': 'checkpoint',
        'checkpointId': 'snap-1',
        'checkpointShardCount': 1,
        'segmentPath': [],
    }
    h.docs[_shard_path('snap-1', 0)] = {
        'chunk': json.dumps({'sessionId': 'sess-unpointed'}).encode('utf-8'),
    }

    store = h.store()
    assert await store.get_snapshot(session_id='sess-unpointed') is None
    # Document-ID lookup still works; only the pointer-based path is unavailable.
    by_id = await store.get_snapshot(snapshot_id='snap-1')
    assert by_id is not None
    assert by_id.snapshot_id == 'snap-1'
    assert _pointer_path('sess-unpointed') not in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_corrupt_pointer_returns_none() -> None:
    """A pointer that can't reconstruct returns None without rewriting the pointer."""
    h = FakeStoreHarness()
    h.docs[_snap_path('snap-live')] = {
        'snapshotId': 'snap-live',
        'sessionId': 'sess-corrupt',
        'createdAt': '2026-07-03T00:00:01Z',
        'kind': 'checkpoint',
        'checkpointId': 'snap-live',
        'checkpointShardCount': 1,
        'segmentPath': [],
    }
    h.docs[_shard_path('snap-live', 0)] = {
        'chunk': json.dumps({'sessionId': 'sess-corrupt'}).encode('utf-8'),
    }
    # Structurally valid pointer whose leaf is gone; omit checkpoint meta so
    # lookup falls through to snapshot-id reconstruct (None), not DATA_LOSS
    # from missing shards.
    h.docs[_pointer_path('sess-corrupt')] = {
        'currentSnapshotId': 'snap-deleted',
        'segmentPath': [],
    }

    store = h.store()
    assert await store.get_snapshot(session_id='sess-corrupt') is None
    assert h.docs[_pointer_path('sess-corrupt')]['currentSnapshotId'] == 'snap-deleted'


@pytest.mark.asyncio
async def test_firestore_session_store_invalid_snapshot_doc_raises() -> None:
    """Schema-invalid snapshot docs fail loud instead of looking like a miss."""
    h = FakeStoreHarness()
    h.docs[_snap_path('snap-bad')] = {
        'snapshotId': 'snap-bad',
        'sessionId': 'sess-1',
        'createdAt': '2026-07-03T00:00:00Z',
        'kind': 'not-a-kind',
        'checkpointId': 'snap-bad',
        'checkpointShardCount': 1,
        'segmentPath': [],
    }
    before = dict(h.docs[_snap_path('snap-bad')])
    store = h.store()

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-bad')
    assert exc_info.value.status == 'DATA_LOSS'

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'snap-bad',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-bad',
                session_id='sess-1',
                created_at='2026-07-03T00:00:01Z',
                state=SessionState(session_id='sess-1'),
            ),
        )
    assert exc_info.value.status == 'DATA_LOSS'
    assert h.docs[_snap_path('snap-bad')] == before


@pytest.mark.asyncio
async def test_firestore_session_store_missing_segment_path_raises() -> None:
    """segmentPath is required on snapshot docs (empty list is ok; absent is not)."""
    h = FakeStoreHarness()
    h.docs[_snap_path('snap-no-seg')] = {
        'snapshotId': 'snap-no-seg',
        'sessionId': 'sess-1',
        'createdAt': '2026-07-03T00:00:00Z',
        'kind': 'checkpoint',
        'checkpointId': 'snap-no-seg',
        'checkpointShardCount': 1,
    }
    store = h.store()

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-no-seg')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_datetime_in_custom_round_trips() -> None:
    """SessionState mode='json' coerces datetimes before they are persisted."""
    h = FakeStoreHarness()
    store = h.store()
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await store.save_snapshot(
        'snap-dt',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-dt',
            session_id='sess-dt',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-dt', custom={'when': when}),
        ),
    )

    shard = h.docs[_shard_path('snap-dt', 0)]
    stored = json.loads(shard['chunk'].decode('utf-8'))
    assert stored['custom']['when'] == '2026-01-01T00:00:00Z'

    loaded = await store.get_snapshot(snapshot_id='snap-dt')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'when': '2026-01-01T00:00:00Z'}


@pytest.mark.asyncio
async def test_firestore_session_store_oversized_diff_promotes_to_checkpoint() -> None:
    """A patch larger than shard_size is stored as a sharded checkpoint."""
    h = FakeStoreHarness()
    store = h.store(shard_size=64, checkpoint_interval=25)

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-2',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-2',
            parent_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            state=SessionState(session_id='sess-1', custom={'blob': 'x' * 200}),
        ),
    )

    child = h.docs[_snap_path('snap-2')]
    assert child['kind'] == 'checkpoint'
    assert child['checkpointId'] == 'snap-2'
    assert 'statePatch' not in child
    loaded = await store.get_snapshot(snapshot_id='snap-2')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'blob': 'x' * 200}


@pytest.mark.asyncio
async def test_firestore_session_store_status_change_and_cleanup() -> None:
    """Test snapshot status subscription and thread-safe cleanup on terminal status."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-sub',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-sub',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )

    watches, captured_cb = _wire_sync_watch(store)
    queue = await store.on_snapshot_status_change('snap-sub')
    assert len(captured_cb) == 1
    assert len(store._subscriptions) == 1
    (sub,) = store._subscriptions
    assert sub.queue.empty()

    await _pump_watch(captured_cb, 0, _status_doc('snap-sub', 'pending'))
    assert await _next_status(queue) == SnapshotStatus.PENDING

    await _pump_watch(captured_cb, 0, _status_doc('snap-sub', 'aborted'))

    assert await _next_status(queue) == SnapshotStatus.ABORTED
    assert await _stream_ended(queue)
    watches[0].unsubscribe.assert_called_once()
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_terminal_cleanup_falls_back_when_create_task_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the loop can't schedule async teardown, still unsubscribe the watch."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-task', _pending_mk('snap-task'))
    watches, captured_cb = _wire_sync_watch(store)
    queue = await store.on_snapshot_status_change('snap-task')

    await _pump_watch(captured_cb, 0, _status_doc('snap-task', 'pending'))
    assert await _next_status(queue) == SnapshotStatus.PENDING

    def boom(_coro: Any) -> Any:  # noqa: ANN401
        raise RuntimeError('no running event loop')

    monkeypatch.setattr(asyncio, 'create_task', boom)
    await _pump_watch(captured_cb, 0, _status_doc('snap-task', 'completed'))

    assert await _next_status(queue) == SnapshotStatus.COMPLETED
    assert await _stream_ended(queue)
    watches[0].unsubscribe.assert_called_once()
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_close_stops_watches_and_sync_client() -> None:
    """close() unsubscribes watches and closes a lazily-created sync client."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-close',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-close',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )

    watches, _captured = _wire_sync_watch(store)
    mock_sync_client = cast(Any, store.client)._to_sync_copy.return_value
    await store.on_snapshot_status_change('snap-close')
    assert store.sync_client is mock_sync_client
    assert len(store._subscriptions) == 1
    assert all(s.watch is not None for s in store._subscriptions)

    store.close()
    watches[0].unsubscribe.assert_called_once()
    mock_sync_client.close.assert_called_once()
    assert store.sync_client is None
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_close_does_not_close_injected_sync_client() -> None:
    """Caller-owned sync_client is left open by close()."""
    h = FakeStoreHarness()
    injected = MagicMock()
    store = h.store(sync_client=injected)
    store.close()
    injected.close.assert_not_called()


@pytest.mark.asyncio
async def test_firestore_session_store_ensure_sync_client_raises_genkit_error() -> None:
    """_ensure_sync_client() raises GenkitError if client has no _to_sync_copy and no sync_client set."""
    custom_client = MagicMock(spec=[])
    store = FirestoreSessionStore(client=custom_client)
    with pytest.raises(GenkitError) as exc_info:
        store._ensure_sync_client()
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'Realtime status watches require a synchronous Firestore client' in str(exc_info.value)


def _by_user(context: dict[str, Any] | None = None) -> str:
    """Prefix from authenticated user id (tenant isolation)."""
    if not isinstance(context, dict):
        return 'anonymous'
    auth = context.get('auth')
    if not isinstance(auth, dict):
        return 'anonymous'
    uid = auth.get('uid')
    return uid if isinstance(uid, str) and uid else 'anonymous'


def _ctx(uid: str) -> dict[str, Any]:
    return {'auth': {'uid': uid}}


@pytest.mark.asyncio
async def test_firestore_session_store_isolates_snapshot_id_per_tenant() -> None:
    """Same snapshot id under different tenants must not cross-read."""
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=_by_user)

    await store.save_snapshot(
        'shared-id',
        lambda _e: SessionSnapshot(
            snapshot_id='shared-id',
            session_id='sess-a',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-a', custom={'counter': 1}),
        ),
        context=_ctx('alice'),
    )

    as_alice = await store.get_snapshot(snapshot_id='shared-id', context=_ctx('alice'))
    assert as_alice is not None
    assert as_alice.state is not None
    assert as_alice.state.custom == {'counter': 1}

    as_bob = await store.get_snapshot(snapshot_id='shared-id', context=_ctx('bob'))
    assert as_bob is None

    assert _snap_path('shared-id', prefix='alice') in h.docs
    assert _snap_path('shared-id', prefix='global') not in h.docs
    assert _snap_path('shared-id', prefix='bob') not in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_isolates_session_id_per_tenant() -> None:
    """Same session id resolves only within the caller's tenant prefix."""
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=_by_user)

    await store.save_snapshot(
        'a1',
        lambda _e: SessionSnapshot(
            snapshot_id='a1',
            session_id='shared-sess',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='shared-sess', custom={'counter': 11}),
        ),
        context=_ctx('alice'),
    )

    as_bob = await store.get_snapshot(session_id='shared-sess', context=_ctx('bob'))
    assert as_bob is None

    as_alice = await store.get_snapshot(session_id='shared-sess', context=_ctx('alice'))
    assert as_alice is not None
    assert as_alice.snapshot_id == 'a1'
    assert as_alice.state is not None
    assert as_alice.state.custom == {'counter': 11}


def _wire_sync_watch(store: FirestoreSessionStore) -> tuple[list[MagicMock], list[Any]]:
    """Attach mock sync on_snapshot watches; return (watches, captured_callbacks)."""
    watches: list[MagicMock] = []
    captured_cb: list[Any] = []

    def on_snapshot_side_effect(cb: Any) -> Any:  # noqa: ANN401
        captured_cb.append(cb)
        watch_mock = MagicMock()
        watches.append(watch_mock)
        return watch_mock

    mock_sync_doc_ref = MagicMock()
    mock_sync_doc_ref.on_snapshot.side_effect = on_snapshot_side_effect
    mock_sync_col = MagicMock()
    mock_sync_col.document.return_value = mock_sync_doc_ref
    mock_sync_doc_ref.collection.return_value = mock_sync_col
    mock_sync_client = MagicMock()
    mock_sync_client.collection.return_value = mock_sync_col
    cast(Any, store.client)._to_sync_copy.return_value = mock_sync_client
    return watches, captured_cb


def _status_doc(snapshot_id: str, status: str, *, session_id: str = 'sess-1') -> MagicMock:
    doc = MagicMock()
    doc.exists = True
    doc.to_dict.return_value = {
        'snapshotId': snapshot_id,
        'sessionId': session_id,
        'createdAt': '2026-07-03T00:00:00Z',
        'status': status,
    }
    return doc


async def _pump_watch(captured_cb: list[Any], index: int, doc: MagicMock) -> None:
    captured_cb[index]([doc], None, None)
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_firestore_session_store_upsert_diff_leaf_rewrites_patch() -> None:
    """Heartbeat/abort-style upsert of an existing diff leaf refreshes statePatch."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-2',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-2',
            parent_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )
    before = h.docs[_snap_path('snap-2')]
    assert before['kind'] == 'diff'
    assert before['statePatch']

    await store.save_snapshot(
        'snap-2',
        lambda existing: (
            existing.model_copy(
                update={
                    'heartbeat_at': '2026-07-03T00:00:05Z',
                    'state': SessionState(session_id='sess-1', custom={'n': 3}),
                }
            )
            if existing
            else None
        ),
    )

    after = h.docs[_snap_path('snap-2')]
    assert after['kind'] == 'diff'
    assert after['checkpointId'] == 'snap-1'
    assert after['segmentPath'] == ['snap-2']
    assert after['statePatch']
    assert after['statePatch'] != before['statePatch']
    loaded = await store.get_snapshot(snapshot_id='snap-2')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'n': 3}
    assert loaded.heartbeat_at == '2026-07-03T00:00:05Z'


@pytest.mark.asyncio
async def test_firestore_session_store_upsert_oversized_diff_promotes_to_checkpoint() -> None:
    """Upserting a diff leaf with an oversized patch rewrites it as a checkpoint."""
    h = FakeStoreHarness()
    store = h.store(shard_size=64, checkpoint_interval=25)

    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-1', custom={'n': 1}),
        ),
    )
    await store.save_snapshot(
        'snap-2',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-2',
            parent_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )
    assert h.docs[_snap_path('snap-2')]['kind'] == 'diff'

    await store.save_snapshot(
        'snap-2',
        lambda existing: (
            existing.model_copy(update={'state': SessionState(session_id='sess-1', custom={'blob': 'x' * 200})})
            if existing
            else None
        ),
    )

    child = h.docs[_snap_path('snap-2')]
    assert child['kind'] == 'checkpoint'
    assert child['checkpointId'] == 'snap-2'
    assert 'statePatch' not in child
    loaded = await store.get_snapshot(snapshot_id='snap-2')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'blob': 'x' * 200}


@pytest.mark.asyncio
async def test_firestore_session_store_missing_snapshot_subscribe_waits_for_create() -> None:
    """A missing snapshot stays subscribed; each caller gets its own watch."""
    h = FakeStoreHarness()
    store = h.store()
    watches, captured_cb = _wire_sync_watch(store)

    q1 = await store.on_snapshot_status_change('missing')
    assert all(s.queue.empty() for s in store._subscriptions)
    assert len(captured_cb) == 1
    assert len(store._subscriptions) == 1

    q2 = await store.on_snapshot_status_change('missing')
    assert len(captured_cb) == 2
    assert len(store._subscriptions) == 2
    assert all(s.queue.empty() for s in store._subscriptions)
    watches[0].unsubscribe.assert_not_called()
    watches[1].unsubscribe.assert_not_called()

    await store.save_snapshot(
        'missing',
        lambda _e: SessionSnapshot(
            snapshot_id='missing',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )
    pending = _status_doc('missing', 'pending')
    await _pump_watch(captured_cb, 0, pending)
    await _pump_watch(captured_cb, 1, pending)
    assert await _next_status(q1) == SnapshotStatus.PENDING
    assert await _next_status(q2) == SnapshotStatus.PENDING


@pytest.mark.asyncio
async def test_firestore_session_store_retries_on_aborted_commit() -> None:
    """@async_transactional retries the RMW closure when commit raises Aborted."""
    h = FakeStoreHarness()
    store = h.store()
    h.commit_aborts_remaining = 2

    saved = await store.save_snapshot(
        'snap-retry',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-retry',
            session_id='sess-retry',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-retry', custom={'n': 1}),
        ),
    )
    assert saved is not None
    assert h.commit_attempts == 3
    loaded = await store.get_snapshot(snapshot_id='snap-retry')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'n': 1}


@pytest.mark.asyncio
async def test_fake_harness_rejects_read_after_write() -> None:
    """Harness mirrors the real client: buffered writes forbid later reads."""
    h = FakeStoreHarness()
    store = h.store()
    ref = store._snapshot_ref('snap-x', 'global')
    txn = h.client.transaction()
    await txn._begin()
    txn.set(ref, {'snapshotId': 'snap-x'})
    with pytest.raises(ReadAfterWriteError):
        await ref.get(transaction=txn)


@pytest.mark.asyncio
async def test_firestore_session_store_upsert_restores_deleted_pointer() -> None:
    """A leaf upsert restores a missing pointer so session lookup keeps working."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )
    del h.docs[_pointer_path('sess-1')]

    def beat(existing: SessionSnapshot | None) -> SessionSnapshot | None:
        assert existing is not None
        return existing.model_copy(update={'heartbeat_at': '2026-07-03T00:00:05Z'})

    await store.save_snapshot('snap-1', beat)
    pointer = h.docs[_pointer_path('sess-1')]
    assert pointer['currentSnapshotId'] == 'snap-1'
    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None
    assert loaded.snapshot_id == 'snap-1'


@pytest.mark.asyncio
async def test_firestore_session_store_forked_pointer_keeps_current_snapshot_id() -> None:
    """After a fork the pointer is the plain last-writer shape: no branch metadata."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', None, custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    await store.save_snapshot('snap-b', _mk('snap-b', 'snap-root', custom={'n': 2}))

    pointer = h.docs[_pointer_path('sess-1')]
    assert pointer['currentSnapshotId'] == 'snap-b'
    assert 'isAmbiguous' not in pointer
    assert 'leaves' not in pointer
    assert isinstance(pointer['updatedAt'], str)


@pytest.mark.asyncio
async def test_firestore_session_store_null_value_survives_patch_wire() -> None:
    """Clearing a field to null keeps ``value: null`` on the stored patch op."""
    h = FakeStoreHarness()
    store = h.store()

    def mk(sid: str, parent: str | None, custom: dict) -> object:  # noqa: ANN401
        return lambda _e: SessionSnapshot(
            snapshot_id=sid,
            session_id='sess-1',
            parent_id=parent,
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1', custom=custom),
        )

    await store.save_snapshot('snap-1', mk('snap-1', None, {'x': 1}))
    await store.save_snapshot('snap-2', mk('snap-2', 'snap-1', {'x': None}))

    doc = h.docs[_snap_path('snap-2')]
    assert doc['kind'] == 'diff'
    assert isinstance(doc['statePatch'], str)
    patch = json.loads(doc['statePatch'])
    ops = [o for o in patch if o['path'] == '/custom/x']
    assert ops, f'no op for /custom/x in {doc["statePatch"]}'
    assert ops[0]['op'] == 'replace'
    assert 'value' in ops[0]
    assert ops[0]['value'] is None

    loaded = await store.get_snapshot(snapshot_id='snap-2')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'x': None}


@pytest.mark.asyncio
async def test_firestore_session_store_cjk_diff_not_prematurely_promoted() -> None:
    """Multilingual diffs are sized in raw UTF-8, not Unicode-escape length."""
    text = '中' * 200
    raw = len(json.dumps(text, ensure_ascii=False).encode('utf-8'))  # ~602
    escaped = len(json.dumps(text).encode('utf-8'))  # ~1202
    shard_size = (raw + escaped) // 2  # diff fits raw, not escaped
    h = FakeStoreHarness()
    store = h.store(shard_size=shard_size)

    def mk(sid: str, parent: str | None, custom: dict) -> object:  # noqa: ANN401
        return lambda _e: SessionSnapshot(
            snapshot_id=sid,
            session_id='sess-1',
            parent_id=parent,
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1', custom=custom),
        )

    await store.save_snapshot('snap-1', mk('snap-1', None, {}))
    await store.save_snapshot('snap-2', mk('snap-2', 'snap-1', {'text': text}))
    assert h.docs[_snap_path('snap-2')]['kind'] == 'diff'


@pytest.mark.asyncio
async def test_firestore_session_store_stamps_parseable_timestamps() -> None:
    """Store-stamped timestamps are valid ISO-8601 (exact format is not pinned)."""
    from datetime import datetime as _dt

    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='',  # force the store to stamp
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1'),
        ),
    )
    assert _dt.fromisoformat(h.docs[_snap_path('snap-1')]['createdAt'])
    assert _dt.fromisoformat(h.docs[_pointer_path('sess-1')]['updatedAt'])


def _mk(sid: str, parent: str | None = None, custom: dict | None = None) -> Any:  # noqa: ANN401
    return lambda _e: SessionSnapshot(
        snapshot_id=sid,
        session_id='sess-1',
        parent_id=parent,
        created_at='2026-07-03T00:00:00Z',
        status=SnapshotStatus.COMPLETED,
        state=SessionState(session_id='sess-1', custom=custom or {}),
    )


@pytest.mark.asyncio
async def test_firestore_session_store_missing_shard_raises_data_loss() -> None:
    """A checkpoint whose shard document vanished fails loud, not with partial state."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'x': 1}))
    del h.docs[_shard_path('snap-1', 0)]

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_corrupt_shard_raises_data_loss() -> None:
    """A shard document that fails schema validation is DATA_LOSS, not garbage state."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'x': 1}))
    h.docs[_shard_path('snap-1', 0)] = {'not_chunk': 1}

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_missing_checkpoint_doc_returns_none() -> None:
    """Pointer names a checkpoint whose snapshot doc is gone: lookup degrades to None."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    del h.docs[_snap_path('snap-1')]

    assert await store.get_snapshot(session_id='sess-1') is None


@pytest.mark.asyncio
async def test_firestore_session_store_corrupt_checkpoint_doc_raises_data_loss() -> None:
    """Pointer fast-path re-reads the checkpoint doc; corruption there is DATA_LOSS."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    h.docs[_snap_path('snap-1')]['kind'] = 'not-a-kind'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_checkpoint_id_mismatch_raises_data_loss() -> None:
    """A doc disagreeing with its own address is corruption, not not-found.

    Pointer and snapshot metadata commit in one transaction, so no legitimate
    staleness can produce this shape.
    """
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    h.docs[_snap_path('snap-1')]['snapshotId'] = 'snap-other'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_missing_segment_raises_data_loss() -> None:
    """A deleted segment referenced by the target's chain is DATA_LOSS, matching missing shards."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    await store.save_snapshot('snap-b', _mk('snap-b', 'snap-a', custom={'n': 2}))
    del h.docs[_snap_path('snap-a')]

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-b')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_corrupt_segment_raises_data_loss() -> None:
    """A diff chain with an invalid interior segment fails loud."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    await store.save_snapshot('snap-b', _mk('snap-b', 'snap-a', custom={'n': 2}))
    h.docs[_snap_path('snap-a')]['kind'] = 'not-a-kind'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-b')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_missing_parent_makes_child_a_checkpoint() -> None:
    """A new child of a nonexistent parent self-heals forward as a full checkpoint."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)

    await store.save_snapshot('snap-child', _mk('snap-child', 'snap-ghost', custom={'n': 1}))

    doc = h.docs[_snap_path('snap-child')]
    assert doc['kind'] == 'checkpoint'
    assert doc['segmentPath'] == []
    loaded = await store.get_snapshot(snapshot_id='snap-child')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'n': 1}


@pytest.mark.asyncio
async def test_firestore_session_store_extending_corrupt_current_leaf_fails_loud() -> None:
    """A new child of the corrupt *current* leaf raises DATA_LOSS (no diff against garbage).

    The pointer fast-path supplies the parent's chain metadata, so the corruption
    is discovered during parent reconstruction and surfaces loudly.
    """
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-parent', _mk('snap-parent', custom={'n': 0}))
    h.docs[_snap_path('snap-parent')]['kind'] = 'not-a-kind'

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-child', _mk('snap-child', 'snap-parent', custom={'n': 1}))
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_forking_corrupt_interior_parent_self_heals() -> None:
    """Forking off a corrupt *non-current* parent self-heals forward as a checkpoint.

    Off the pointer fast-path the parent doc must be read directly; when it fails
    validation the child is written as a full checkpoint with no dependence on the
    corrupt chain.
    """
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    h.docs[_snap_path('snap-root')]['kind'] = 'not-a-kind'

    await store.save_snapshot('snap-fork', _mk('snap-fork', 'snap-root', custom={'n': 9}))

    doc = h.docs[_snap_path('snap-fork')]
    assert doc['kind'] == 'checkpoint'
    assert doc['segmentPath'] == []
    loaded = await store.get_snapshot(snapshot_id='snap-fork')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'n': 9}


@pytest.mark.asyncio
async def test_firestore_session_store_shrinking_checkpoint_prunes_orphan_shards() -> None:
    """Re-checkpointing with smaller state deletes shards beyond the new count."""
    h = FakeStoreHarness()
    store = h.store(shard_size=64)

    await store.save_snapshot('snap-1', _mk('snap-1', custom={'blob': 'x' * 200}))
    assert _shard_path('snap-1', 0) in h.docs
    assert _shard_path('snap-1', 2) in h.docs  # >= 3 shards at size 64

    await store.save_snapshot(
        'snap-1', lambda e: e.model_copy(update={'state': SessionState(session_id='sess-1', custom={})})
    )

    assert _shard_path('snap-1', 0) in h.docs
    assert _shard_path('snap-1', 1) not in h.docs
    assert _shard_path('snap-1', 2) not in h.docs
    assert _shard_path('snap-1', 1) in h.deleted

    loaded = await store.get_snapshot(snapshot_id='snap-1')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {}


@pytest.mark.asyncio
async def test_firestore_session_store_missing_session_id_raises_invalid_argument() -> None:
    """A snapshot without a sessionId cannot be pointer-indexed: INVALID_ARGUMENT."""
    h = FakeStoreHarness()
    store = h.store()

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'snap-1',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-1',
                session_id='',
                created_at='2026-07-03T00:00:00Z',
                status=SnapshotStatus.COMPLETED,
                state=SessionState(session_id=''),
            ),
        )
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'session_id' in str(exc_info.value)
    assert _pointer_path('') not in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_listener_failure_rolls_back_subscription(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the realtime watch cannot start, the queue registration is rolled back."""
    h = FakeStoreHarness()
    store = h.store()

    def boom() -> None:
        raise GenkitError(status='FAILED_PRECONDITION', message='no sync client')

    monkeypatch.setattr(store, '_ensure_sync_client', boom)

    with pytest.raises(GenkitError):
        await store.on_snapshot_status_change('snap-missing')

    assert len(store._subscriptions) == 0


def test_firestore_session_store_close_is_idempotent_without_watches() -> None:
    """close() with nothing to tear down returns cleanly, twice."""
    h = FakeStoreHarness()
    store = h.store()
    store.close()
    store.close()


@pytest.mark.parametrize(
    ('kwargs', 'fragment'),
    [
        ({'checkpoint_interval': 0}, 'checkpoint_interval'),
        ({'checkpoint_interval': -3}, 'checkpoint_interval'),
        ({'shard_size': 0}, 'shard_size'),
        ({'shard_size': 2_000_000}, 'shard_size'),
    ],
)
def test_firestore_session_store_rejects_invalid_tuning(kwargs: dict, fragment: str) -> None:
    """Out-of-range tuning parameters fail at construction, not at first write."""
    h = FakeStoreHarness()
    with pytest.raises(GenkitError) as exc_info:
        h.store(**kwargs)
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert fragment in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_multibyte_state_survives_shard_boundaries() -> None:
    """Raw UTF-8 shards may split mid-codepoint; stitch must concatenate before decoding."""
    h = FakeStoreHarness()
    store = h.store(shard_size=7)  # tiny shards force splits inside CJK/emoji sequences

    text = '中文😀' * 20
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'text': text}))
    assert _shard_path('snap-1', 3) in h.docs  # genuinely multi-shard

    loaded = await store.get_snapshot(snapshot_id='snap-1')
    assert loaded is not None
    assert loaded.state is not None
    assert loaded.state.custom == {'text': text}


@pytest.mark.asyncio
async def test_firestore_session_store_tampered_diff_id_raises_data_loss() -> None:
    """A diff doc whose snapshotId disagrees with its address fails the end-of-chain check."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    h.docs[_snap_path('snap-a')]['snapshotId'] = 'snap-other'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-a')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_non_object_shard_state_raises_data_loss() -> None:
    """Shards decoding to valid JSON that is not an object are corruption, not empty state."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'x': 1}))
    h.docs[_shard_path('snap-1', 0)]['chunk'] = b'42'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_malformed_state_patch_raises_data_loss() -> None:
    """A segment whose statePatch fails to parse names the corrupt doc via DATA_LOSS."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    h.docs[_snap_path('snap-a')]['statePatch'] = json.dumps([{'op': 'bogus-op', 'path': '/n'}])

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-a')
    assert exc_info.value.status == 'DATA_LOSS'
    assert 'statePatch' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_array_state_patch_names_field() -> None:
    """Pre-fix array statePatch is DATA_LOSS naming the field and JSON string expectation."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    h.docs[_snap_path('snap-a')]['statePatch'] = [{'op': 'replace', 'path': '/n', 'value': 1}]

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-a')
    assert exc_info.value.status == 'DATA_LOSS'
    msg = str(exc_info.value)
    assert 'statePatch' in msg and 'JSON string' in msg and 'found list' in msg


@pytest.mark.asyncio
async def test_firestore_session_store_null_state_patch_is_data_loss() -> None:
    """Null/empty/missing statePatch on a diff segment is DATA_LOSS, not a silent []."""
    from genkit_google_cloud.session_store.firestore import _decode_state_patch

    with pytest.raises(GenkitError) as e_none:
        _decode_state_patch(None, snapshot_id='snap-a')
    assert e_none.value.status == 'DATA_LOSS'
    assert 'statePatch' in str(e_none.value) and 'missing or empty' in str(e_none.value)
    assert 're-create' in str(e_none.value)

    with pytest.raises(GenkitError) as e_empty:
        _decode_state_patch('', snapshot_id='snap-a')
    assert e_empty.value.status == 'DATA_LOSS' and 'missing or empty' in str(e_empty.value)

    assert _decode_state_patch('[]', snapshot_id='snap-a') == []

    with pytest.raises(GenkitError) as e_bad:
        _decode_state_patch('{', snapshot_id='snap-a')
    assert 'unparseable' in str(e_bad.value)

    with pytest.raises(GenkitError) as e_obj:
        _decode_state_patch('{"a":1}', snapshot_id='snap-a')
    assert 'not a patch array' in str(e_obj.value)

    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    h.docs[_snap_path('snap-a')]['statePatch'] = None
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-a')
    assert exc_info.value.status == 'DATA_LOSS'
    assert 'missing or empty' in str(exc_info.value)

    # Legitimate zero-change turn: encoder writes "[]", reconstruct equals parent.
    h.docs[_snap_path('snap-a')]['statePatch'] = '[]'
    got = await store.get_snapshot(snapshot_id='snap-a')
    assert got is not None and got.state is not None
    assert got.state.custom == {'n': 0}


@pytest.mark.asyncio
async def test_firestore_session_store_unparseable_pointer_read_raises_data_loss() -> None:
    """On reads, a pointer doc that fails validation is a corrupt index, not a missing session."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    h.docs[_pointer_path('sess-1')] = {'currentSnapshotId': 'snap-1'}  # no segmentPath: invalid

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_unparseable_pointer_write_self_heals() -> None:
    """On writes, a corrupt pointer is rewritten wholesale so the session stays writable."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    h.docs[_pointer_path('sess-1')] = {'currentSnapshotId': 'snap-1'}  # invalid shape

    def beat(existing: SessionSnapshot | None) -> SessionSnapshot | None:
        assert existing is not None
        return existing.model_copy(update={'heartbeat_at': '2026-07-03T00:00:05Z'})

    await store.save_snapshot('snap-1', beat)
    pointer = h.docs[_pointer_path('sess-1')]
    assert pointer['currentSnapshotId'] == 'snap-1'
    assert pointer['segmentPath'] == []
    loaded = await store.get_snapshot(session_id='sess-1')
    assert loaded is not None
    assert loaded.snapshot_id == 'snap-1'


@pytest.mark.asyncio
async def test_firestore_session_store_deleted_diff_leaf_is_not_found_via_both_paths() -> None:
    """A wholly deleted diff leaf is None via session lookup AND by id — never DATA_LOSS.

    The pointer fast path discovers the deletion as a missing *final* segment;
    that segment is the target itself, so it must match by-id not-found
    semantics, not the broken-chain rule for interior segments.
    """
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)
    await store.save_snapshot('snap-root', _mk('snap-root', custom={'n': 0}))
    await store.save_snapshot('snap-a', _mk('snap-a', 'snap-root', custom={'n': 1}))
    del h.docs[_snap_path('snap-a')]  # TTL ate the leaf; pointer still names it

    assert await store.get_snapshot(session_id='sess-1') is None
    assert await store.get_snapshot(snapshot_id='snap-a') is None


@pytest.mark.asyncio
async def test_firestore_session_store_watch_uses_sync_client_despite_async_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Watches must always attach via the sync client, never the async ref.

    Real AsyncDocumentReference exposes an ``on_snapshot`` stub whose body is
    ``raise NotImplementedError``; a fake ref that mirrors it must not derail
    the watch onto the async path. Regression for the real-backend bug where
    every subscription died with NotImplementedError (originally caused by
    dispatching on ``hasattr(ref, 'on_snapshot')``).
    """
    h = FakeStoreHarness()
    store = h.store()
    watches, captured_cb = _wire_sync_watch(store)

    orig_snapshot_ref = store._snapshot_ref

    def trapped_snapshot_ref(snapshot_id: str, prefix: str) -> Any:  # noqa: ANN401
        ref = orig_snapshot_ref(snapshot_id, prefix)
        # Mirror the real library: the async ref HAS on_snapshot, but it raises.
        ref.on_snapshot = cast(Any, MagicMock(side_effect=NotImplementedError))
        return ref

    monkeypatch.setattr(store, '_snapshot_ref', trapped_snapshot_ref)
    assert hasattr(store._snapshot_ref('snap-w', 'global'), 'on_snapshot')  # the trap is armed

    _stream = await store.on_snapshot_status_change('snap-w')
    assert all(s.queue.empty() for s in store._subscriptions)
    assert len(captured_cb) == 1  # watch attached via the SYNC client
    assert len(store._subscriptions) == 1
    (sub,) = store._subscriptions
    assert sub.watch is watches[0]


@pytest.mark.asyncio
async def test_firestore_session_store_retry_exhaustion_raises_typed_aborted() -> None:
    """Contention beyond transaction_max_attempts surfaces as GenkitError ABORTED."""
    h = FakeStoreHarness()
    store = h.store()
    h.transaction._max_attempts = 2
    h.commit_aborts_remaining = 5

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', _mk('snap-1'))
    assert exc_info.value.status == 'ABORTED'
    assert 'injected abort' in str(exc_info.value)  # the final Aborted's text, verbatim


@pytest.mark.asyncio
async def test_firestore_session_store_payload_limit_raises_typed_invalid_argument() -> None:
    """A Firestore InvalidArgument (e.g. ~10 MiB txn payload) is wrapped with a hint."""
    h = FakeStoreHarness()
    store = h.store()
    h.commit_raises = google_exceptions.InvalidArgument('maximum entity size exceeded')

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', _mk('snap-1'))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'maximum entity size exceeded' in str(exc_info.value)  # server text verbatim


@pytest.mark.asyncio
async def test_firestore_session_store_transaction_max_attempts_plumbed_and_validated() -> None:
    """The retry knob reaches client.transaction() and rejects nonsense values."""
    h = FakeStoreHarness()
    store = h.store(transaction_max_attempts=7)
    await store.save_snapshot('snap-1', _mk('snap-1'))
    assert h.client.transaction.call_args.kwargs.get('max_attempts') == 7

    with pytest.raises(GenkitError) as exc_info:
        h.store(transaction_max_attempts=0)
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_firestore_session_store_maps_other_google_errors_mechanically() -> None:
    """Any GoogleAPICallError maps 1:1 by class with the server message preserved."""
    h = FakeStoreHarness()
    store = h.store()
    h.commit_raises = google_exceptions.DeadlineExceeded('deadline of 60.0s exceeded')

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', _mk('snap-1'))
    assert exc_info.value.status == 'DEADLINE_EXCEEDED'
    assert 'deadline of 60.0s exceeded' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_non_google_errors_pass_through_untouched() -> None:
    """Fail-open: a plain ValueError from the mutation fn is not mistranslated."""
    h = FakeStoreHarness()
    store = h.store()

    def bad_fn(_existing: SessionSnapshot | None) -> SessionSnapshot | None:
        raise ValueError('user code exploded')

    with pytest.raises(ValueError, match='user code exploded'):
        await store.save_snapshot('snap-1', bad_fn)


@pytest.mark.asyncio
async def test_firestore_session_store_masked_rollback_valueerror_raises_typed_aborted() -> None:
    """The library's no-transaction-ID ValueError (unchained; original error masked) maps to ABORTED.

    Observed on real Firestore under hot-pointer contention: a failure before
    the transaction obtains an ID triggers rollback, which raises this bare
    ValueError and destroys the root cause. Regression for the C-01 raw leak.
    """
    h = FakeStoreHarness()
    store = h.store()
    h.commit_raises = ValueError('The transaction has no transaction ID, so it cannot be rolled back.')

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', _mk('snap-1'))
    assert exc_info.value.status == 'ABORTED'
    assert 'no transaction ID' in str(exc_info.value)


@pytest.mark.parametrize(
    'bad_id',
    ['a/b', 'a/b/c', '.', '..', '', '__id__'],
)
@pytest.mark.asyncio
async def test_firestore_session_store_rejects_path_structured_ids(bad_id: str) -> None:
    """Ids that Firestore would parse as path structure fail typed, at every entrypoint.

    Regression for W-08: odd-segment ids leaked a raw client ValueError; even
    segment ids silently addressed a deeper document and left a zombie watch.
    """
    h = FakeStoreHarness()
    store = h.store()

    with pytest.raises(GenkitError) as e1:
        await store.save_snapshot(bad_id, _mk('snap-x'))
    assert e1.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as e2:
        await store.get_snapshot(snapshot_id=bad_id)
    assert e2.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as e3:
        await store.get_snapshot(session_id=bad_id)
    assert e3.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as e4:
        await store._read_snapshot(bad_id, prefix='global')
    assert e4.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_firestore_session_store_subscribe_rejects_bad_id_with_no_partial_state() -> None:
    """Subscribe validation fires before ANY registration: no queue, no watch, no zombie."""
    h = FakeStoreHarness()
    store = h.store()

    with pytest.raises(GenkitError) as exc_info:
        await store.on_snapshot_status_change('a/b/c')
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'a/b/c' not in store._subscriptions
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_terminal_status_change_raises_failed_precondition() -> None:
    """Terminal statuses are absorbing: ABORTED can never become COMPLETED.

    Deliberately stricter than JS (which orders writes with a lock but permits
    the overwrite); an aborted run silently becoming completed misrepresents
    what happened, and no subscriber can ever observe the second transition.
    """
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.ABORTED,
            state=SessionState(session_id='sess-1'),
        ),
    )

    def complete(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'status': SnapshotStatus.COMPLETED})

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', complete)
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert h.docs[_snap_path('snap-1')]['status'] == 'aborted'  # unchanged


@pytest.mark.asyncio
async def test_firestore_session_store_terminal_resurrection_raises_failed_precondition() -> None:
    """COMPLETED cannot be resurrected to PENDING either — same absorbing rule."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))  # completed

    def resurrect(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'status': SnapshotStatus.PENDING})

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', resurrect)
    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_firestore_session_store_same_terminal_status_rewrite_allowed() -> None:
    """Rewriting a finished snapshot without changing its status stays legal."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'n': 1}))  # completed

    def annotate(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(
            update={'state': SessionState(session_id='sess-1', custom={'n': 1, 'note': 'post-hoc'})}
        )

    saved = await store.save_snapshot('snap-1', annotate)
    assert saved is not None
    loaded = await store.get_snapshot(snapshot_id='snap-1')
    assert loaded is not None and loaded.state is not None
    assert loaded.state.custom == {'n': 1, 'note': 'post-hoc'}
    assert h.docs[_snap_path('snap-1')]['status'] == 'completed'


@pytest.mark.asyncio
async def test_firestore_session_store_invalid_prefix_raises_before_any_path() -> None:
    """B1: snapshot_path_prefix is the tenant boundary; a path-structured prefix is rejected everywhere."""
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=lambda _ctx: 'ten/ant')

    for call in (
        store.save_snapshot('snap-1', _mk('snap-1')),
        store.get_snapshot(snapshot_id='snap-1'),
        store.get_snapshot(session_id='sess-1'),
        store.on_snapshot_status_change('snap-1'),
    ):
        with pytest.raises(GenkitError) as exc_info:
            await call
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'snapshot_path_prefix' in str(exc_info.value)
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_mutator_supplied_path_ids_rejected() -> None:
    """B1: session_id/parent_id coming back FROM the mutator are held to the same rules."""
    h = FakeStoreHarness()
    store = h.store()

    def bad_session(_e: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-1',
            session_id='a/b',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='a/b'),
        )

    with pytest.raises(GenkitError) as e1:
        await store.save_snapshot('snap-1', bad_session)
    assert e1.value.status == 'INVALID_ARGUMENT' and 'session_id' in str(e1.value)

    def bad_parent(_e: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            parent_id='x/y',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1'),
        )

    with pytest.raises(GenkitError) as e2:
        await store.save_snapshot('snap-1', bad_parent)
    assert e2.value.status == 'INVALID_ARGUMENT' and 'parent_id' in str(e2.value)


@pytest.mark.asyncio
async def test_firestore_session_store_empty_current_snapshot_id_is_data_loss() -> None:
    """A pointer that exists but names nothing is corrupt, not not-found."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    pointer = h.docs[_pointer_path('sess-1')]
    pointer['currentSnapshotId'] = ''

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-1')
    assert exc_info.value.status == 'DATA_LOSS'


@pytest.mark.asyncio
async def test_firestore_session_store_subscribe_context_kwarg_selects_tenant() -> None:
    """N11: explicit context= reaches the watch path (ambient context stays the default)."""
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=lambda ctx: (ctx or {}).get('tenant', 'global'))
    _watches, captured_cb = _wire_sync_watch(store)

    _stream = await store.on_snapshot_status_change('snap-w', context={'tenant': 't9'})
    assert all(s.queue.empty() for s in store._subscriptions)
    assert len(captured_cb) == 1
    sync_mock = cast(Any, store.client)._to_sync_copy.return_value
    sync_mock.collection.return_value.document.assert_any_call('t9')


@pytest.mark.parametrize('terminal', [SnapshotStatus.EXPIRED, SnapshotStatus.FAILED])
@pytest.mark.asyncio
async def test_firestore_session_store_all_terminal_statuses_are_absorbing(terminal: SnapshotStatus) -> None:
    """R4-High: EXPIRED and FAILED are absorbing too, not just ABORTED/COMPLETED."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=terminal,
            state=SessionState(session_id='sess-1'),
        ),
    )

    def complete(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'status': SnapshotStatus.COMPLETED})

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', complete)
    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_firestore_session_store_read_paths_plumb_transaction_max_attempts() -> None:
    """R4-High: the retry knob reaches read-only transactions too."""
    h = FakeStoreHarness()
    store = h.store(transaction_max_attempts=9)
    await store.save_snapshot('snap-1', _mk('snap-1'))
    await store.get_snapshot(snapshot_id='snap-1')
    ro_calls = [c for c in h.client.transaction.call_args_list if c.kwargs.get('read_only')]
    assert ro_calls, 'expected a read-only transaction'
    assert all(c.kwargs.get('max_attempts') == 9 for c in ro_calls)


@pytest.mark.asyncio
async def test_firestore_session_store_promotion_then_prune_cleans_orphan_shards() -> None:
    """R4-High: a diff promoted to checkpoint, then shrunk, deletes its stale shards."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25, shard_size=120)
    await store.save_snapshot('snap-root', _mk('snap-root', None, custom={'n': 0}))
    # Child whose patch exceeds shard_size -> promoted to a multi-shard checkpoint.
    await store.save_snapshot('snap-big', _mk('snap-big', 'snap-root', custom={'blob': 'X' * 500}))
    shards_before = [k for k in h.docs if '/shards/' in k and 'snap-big' in k]
    assert len(shards_before) > 1  # promotion happened, multi-shard

    def shrink(existing: SessionSnapshot | None) -> SessionSnapshot:
        assert existing is not None
        return existing.model_copy(update={'state': SessionState(session_id='sess-1', custom={'n': 1})})

    await store.save_snapshot('snap-big', shrink)
    shards_after = [k for k in h.docs if '/shards/' in k and 'snap-big' in k]
    assert len(shards_after) < len(shards_before)  # orphans pruned
    loaded = await store.get_snapshot(snapshot_id='snap-big')
    assert loaded is not None and loaded.state is not None
    assert loaded.state.custom == {'n': 1}


@pytest.mark.asyncio
async def test_firestore_session_store_watch_isolation_across_tenants_same_snapshot_id() -> None:
    """B1 regression: two tenants, one snapshot id — independent listeners, queues, teardown.

    Pre-fix, subscriptions were keyed by snapshot_id alone: the second tenant
    piggybacked on the first tenant's watch, statuses crossed tenants, and the
    first terminal tore down both.
    """
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=lambda ctx: (ctx or {}).get('tenant', 'global'))
    watches, captured_cb = _wire_sync_watch(store)

    q1 = await store.on_snapshot_status_change('snap-x', context={'tenant': 't1'})
    stream_t2 = await store.on_snapshot_status_change('snap-x', context={'tenant': 't2'})

    assert len(captured_cb) == 2  # one listener per tenant, not a shared one
    assert len(store._subscriptions) == 2
    assert len([s for s in store._subscriptions if s.watch is not None]) == 2

    await store.save_snapshot('snap-x', _mk('snap-x'), context={'tenant': 't1'})
    await _pump_watch(captured_cb, 0, _status_doc('snap-x', 'completed'))
    assert await _next_status(q1) == SnapshotStatus.COMPLETED
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(anext(stream_t2), timeout=0.05)


@pytest.mark.asyncio
async def test_firestore_session_store_stream_survives_polling_timeout() -> None:
    """A consumer polling with wait_for must not kill the subscription."""
    h = FakeStoreHarness()
    store = h.store()
    watches, captured_cb = _wire_sync_watch(store)
    stream = await store.on_snapshot_status_change('snap-poll')

    for _ in range(2):
        pull = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.05)
        assert not pull.done()
        pull.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pull

    await store.save_snapshot('snap-poll', _mk('snap-poll'))
    await _pump_watch(captured_cb, 0, _status_doc('snap-poll', 'completed'))
    final = asyncio.ensure_future(anext(stream))
    for _ in range(200):
        if final.done():
            break
        await asyncio.sleep(0.01)
    assert final.done() and final.result() == SnapshotStatus.COMPLETED

    store.close()
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()  # ended stays ended (latched)


@pytest.mark.asyncio
async def test_firestore_session_store_close_delivers_end_of_stream_locally() -> None:
    """Wake-on-close: a consumer awaiting a queue gets the bare None marker."""
    h = FakeStoreHarness()
    store = h.store()
    _wire_sync_watch(store)
    q = await store.on_snapshot_status_change('snap-pending')

    store.close()
    assert await _stream_ended(q)  # bare EOS: ended without resolution
    assert len(store._subscriptions) == 0


@pytest.mark.asyncio
async def test_firestore_session_store_close_from_thread_wakes_waiters() -> None:
    """close() from a foreign thread delivers the marker via the subscribers' loop."""
    h = FakeStoreHarness()
    store = h.store()
    _wire_sync_watch(store)
    q = await store.on_snapshot_status_change('snap-pending')

    getter = asyncio.ensure_future(_stream_ended(q))
    await asyncio.to_thread(store.close)
    assert await asyncio.wait_for(getter, timeout=2)


@pytest.mark.asyncio
async def test_firestore_session_store_close_after_terminal_adds_nothing() -> None:
    """A stream already ended by a terminal status gets no second marker from close."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))  # completed
    watches, captured_cb = _wire_sync_watch(store)
    q = await store.on_snapshot_status_change('snap-1')
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'completed'))
    assert await _next_status(q) == SnapshotStatus.COMPLETED
    assert await _stream_ended(q)

    store.close()
    assert await _stream_ended(q)  # exhausted stream stays ended; close adds nothing


@pytest.mark.asyncio
async def test_sibling_stores_accept_context_kwarg() -> None:
    """B2: the widened protocol signature is honored by non-tenant stores too."""
    from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore

    mem = InMemorySessionStore()
    statuses = await mem.on_snapshot_status_change('snap-x', context={'tenant': 't1'})
    assert hasattr(statuses, '__anext__')  # accepted and ignored, no error


@pytest.mark.asyncio
async def test_firestore_session_store_subscribe_after_close_raises() -> None:
    """Subscribing to an already-closed store is a typed error, not a zombie stream."""
    h = FakeStoreHarness()
    store = h.store()
    store.close()

    with pytest.raises(GenkitError) as exc_info:
        await store.on_snapshot_status_change('snap-x')
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'closed' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_prefix_failure_never_surfaces_after_commit() -> None:
    """A failing prefix function fails BEFORE the transaction, never after it.

    Pre-fix, save re-invoked the prefix function after commit for the
    subscriber notification; a function that failed on that call turned a
    durably committed write into a raised (and raw) exception.
    """
    # Commit-aware tripwire: the prefix function itself asserts it is never
    # invoked after the transaction has committed.
    h = FakeStoreHarness()

    def commit_aware_prefix(_ctx: Any) -> str:  # noqa: ANN401
        assert h.commit_attempts == 0, 'prefix function invoked AFTER commit'
        return 'global'

    store = h.store(snapshot_path_prefix=commit_aware_prefix)
    saved = await store.save_snapshot('snap-1', _mk('snap-1'))  # must NOT raise
    assert saved is not None
    assert any('snap-1' in k for k in h.docs)

    # And a prefix that fails immediately fails before anything is written.
    h2 = FakeStoreHarness()
    store2 = h2.store(snapshot_path_prefix=lambda _c: (_ for _ in ()).throw(KeyError('down')))
    with pytest.raises(KeyError):
        await store2.save_snapshot('snap-1', _mk('snap-1'))
    assert not any('snap-1' in k for k in h2.docs)  # nothing committed


def _owned_snap(sid: str, session: str, parent: str = '', custom: dict | None = None) -> Any:  # noqa: ANN401
    def fn(_e: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id=sid,
            session_id=session,
            parent_id=parent,
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id=session, custom=custom or {}),
        )

    return fn


@pytest.mark.asyncio
async def test_firestore_session_store_snapshot_id_collision_across_sessions_rejected() -> None:
    """Snapshot ids share one namespace per prefix; a foreign session cannot rewrite one.

    Pre-guard, session B saving 'turn-1' silently overwrote session A's
    'turn-1': A's descendants then reconstructed against B's state — silent
    cross-session data corruption and leakage with no error anywhere.
    """
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('turn-1', _owned_snap('turn-1', 'sess-A', custom={'secret': 'alpha'}))
    await store.save_snapshot('turn-2', _owned_snap('turn-2', 'sess-A', 'turn-1', custom={'secret': 'alpha', 'n': 2}))

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('turn-1', _owned_snap('turn-1', 'sess-B'))
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'sess-A' in str(exc_info.value) and 'sess-B' in str(exc_info.value)

    a2 = await store.get_snapshot(snapshot_id='turn-2')  # A's chain intact
    assert a2 is not None and a2.state is not None
    assert a2.state.custom == {'secret': 'alpha', 'n': 2}


@pytest.mark.asyncio
async def test_inmemory_store_snapshot_id_collision_rejected_too() -> None:
    """The ownership guard lives in shared apply_save: every Python store enforces it."""
    from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore

    mem = InMemorySessionStore()
    await mem.save_snapshot('turn-1', _owned_snap('turn-1', 'sess-A'))
    with pytest.raises(GenkitError) as exc_info:
        await mem.save_snapshot('turn-1', _owned_snap('turn-1', 'sess-B'))
    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_firestore_session_store_cross_session_parent_rejected() -> None:
    """A new snapshot cannot anchor its diff chain to another session's snapshot."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('a-root', _owned_snap('a-root', 'sess-A', custom={'n': 1}))

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('b-child', _owned_snap('b-child', 'sess-B', parent='a-root'))
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'owned by' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_pointer_to_foreign_session_is_data_loss() -> None:
    """A pointer resolving to another session's snapshot is a corrupt index, not an answer."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('a-1', _owned_snap('a-1', 'sess-A'))
    await store.save_snapshot('b-1', _owned_snap('b-1', 'sess-B'))
    # Corrupt sess-A's pointer to name sess-B's snapshot (simulates legacy damage).
    h.docs[_pointer_path('sess-A')] = dict(h.docs[_pointer_path('sess-B')])

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-A')
    assert exc_info.value.status == 'DATA_LOSS'
    assert 'sess-B' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_interior_state_rewrite_rejected() -> None:
    """Rewriting an interior snapshot's state would corrupt descendant diffs."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('parent', _mk('parent', custom={'secret': 'alpha'}))
    await store.save_snapshot('child', _mk('child', parent='parent', custom={'secret': 'alpha', 'n': 2}))

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'parent',
            lambda existing: (
                existing.model_copy(update={'state': SessionState(session_id='sess-1', custom={'secret': 'MUTATED'})})
                if existing
                else None
            ),
        )
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'not session' in str(exc_info.value) and 'current snapshot' in str(exc_info.value)
    assert 'branch' in str(exc_info.value)

    child = await store.get_snapshot(snapshot_id='child')
    assert child is not None and child.state is not None
    assert child.state.custom == {'secret': 'alpha', 'n': 2}


@pytest.mark.asyncio
async def test_firestore_session_store_interior_heartbeat_still_allowed() -> None:
    """Metadata-only rewrites on an interior snapshot must not break the child."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('parent', _mk('parent', custom={'secret': 'alpha'}))
    await store.save_snapshot('child', _mk('child', parent='parent', custom={'secret': 'alpha', 'n': 2}))
    before = await store.get_snapshot(snapshot_id='child')
    assert before is not None and before.state is not None
    before_custom = dict(before.state.custom)

    saved = await store.save_snapshot(
        'parent',
        lambda existing: existing.model_copy(update={'heartbeat_at': '2026-07-03T00:00:09Z'}) if existing else None,
    )
    assert saved is not None
    assert saved.heartbeat_at == '2026-07-03T00:00:09Z'

    after = await store.get_snapshot(snapshot_id='child')
    assert after is not None and after.state is not None
    assert after.state.custom == before_custom == {'secret': 'alpha', 'n': 2}


@pytest.mark.asyncio
async def test_firestore_session_store_parent_id_immutable_on_upsert() -> None:
    """An upsert cannot re-parent an existing snapshot."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('root-a', _mk('root-a', custom={'n': 1}))
    await store.save_snapshot('root-b', _mk('root-b', custom={'n': 2}))
    await store.save_snapshot('leaf', _mk('leaf', parent='root-a', custom={'n': 3}))

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'leaf',
            lambda existing: existing.model_copy(update={'parent_id': 'root-b'}) if existing else None,
        )
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert "parent_id is immutable ('root-a' -> 'root-b')" in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_tip_state_rewrite_still_allowed() -> None:
    """The tip remains writable: heartbeats/finalize that change state still succeed."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('parent', _mk('parent', custom={'n': 1}))
    await store.save_snapshot(
        'tip',
        lambda _e: SessionSnapshot(
            snapshot_id='tip',
            parent_id='parent',
            session_id='sess-1',
            created_at='2026-07-03T00:00:01Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1', custom={'n': 2}),
        ),
    )

    saved = await store.save_snapshot(
        'tip',
        lambda existing: (
            existing.model_copy(
                update={
                    'status': SnapshotStatus.COMPLETED,
                    'state': SessionState(session_id='sess-1', custom={'n': 3}),
                }
            )
            if existing
            else None
        ),
    )
    assert saved is not None
    assert saved.status == SnapshotStatus.COMPLETED
    loaded = await store.get_snapshot(snapshot_id='tip')
    assert loaded is not None and loaded.state is not None
    assert loaded.state.custom == {'n': 3}


def test_firestore_session_store_rejects_zero_shard_size() -> None:
    """shard_size=0 is a typed INVALID_ARGUMENT, not ZeroDivisionError later."""
    h = FakeStoreHarness()
    with pytest.raises(GenkitError) as exc_info:
        h.store(shard_size=0)
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'shard_size' in str(exc_info.value)


def test_firestore_session_store_config_frozen_after_init() -> None:
    """collection/shard_size/checkpoint_interval/attempts cannot be reassigned."""
    h = FakeStoreHarness()
    store = h.store()
    for attr in ('collection', 'shard_size', 'checkpoint_interval', 'transaction_max_attempts'):
        with pytest.raises(GenkitError) as exc_info:
            setattr(store, attr, 'other' if attr == 'collection' else 1)
        assert exc_info.value.status == 'FAILED_PRECONDITION'
        assert 'new store instance' in str(exc_info.value)
    assert h.store(collection='custom-col', shard_size=2048).collection == 'custom-col'


def test_firestore_session_store_rejects_bad_collection_name() -> None:
    """Collection is a path segment; empty or '/' is INVALID_ARGUMENT at construction."""
    h = FakeStoreHarness()
    with pytest.raises(GenkitError) as exc_info:
        h.store(collection='bad/name')
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'collection' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_deep_nested_diff_round_trips() -> None:
    """Deep nested custom state survives as a JSON-string diff (not a map nesting error)."""
    h = FakeStoreHarness()
    store = h.store()
    nested: Any = 'leaf'
    for _ in range(25):
        nested = {'n': nested}
    await store.save_snapshot('root', _mk('root', custom={'n': 0}))
    await store.save_snapshot('deep', _mk('deep', parent='root', custom={'tree': nested}))
    assert isinstance(h.docs[_snap_path('deep')]['statePatch'], str)
    loaded = await store.get_snapshot(snapshot_id='deep')
    assert loaded is not None and loaded.state is not None
    assert loaded.state.custom == {'tree': nested}


@pytest.mark.asyncio
async def test_firestore_session_store_zero_checkpoint_shard_count_is_data_loss() -> None:
    """checkpointShardCount < 1 is corruption, not an empty state."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'x': 1}))
    h.docs[_snap_path('snap-1')]['checkpointShardCount'] = 0

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'
    assert 'invalid snapshot document' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_subscribe_tolerates_missing_shards() -> None:
    """Status subscribe reads the snapshot doc only; missing shards do not DATA_LOSS the watch."""
    h = FakeStoreHarness()
    store = h.store()
    watches, captured_cb = _wire_sync_watch(store)
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1', custom={'x': 1}),
        ),
    )
    del h.docs[_shard_path('snap-1', 0)]

    stream = await store.on_snapshot_status_change('snap-1')
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'pending'))
    assert await _next_status(stream) == SnapshotStatus.PENDING
    assert len(store._subscriptions) == 1


@pytest.mark.asyncio
async def test_firestore_session_store_mutator_valueerror_propagates_verbatim() -> None:
    """App ValueError from a mutator must not be misclassified as retryable ABORTED."""
    h = FakeStoreHarness()
    store = h.store()

    def boom(_existing: SessionSnapshot | None) -> SessionSnapshot:
        raise ValueError('no transaction ID')

    with pytest.raises(ValueError, match='no transaction ID') as exc_info:
        await store.save_snapshot('snap-1', boom)
    assert not isinstance(exc_info.value, GenkitError)


@pytest.mark.asyncio
async def test_firestore_session_store_maps_retry_error_cause() -> None:
    """RetryError unwraps to its GoogleAPICallError cause when present."""
    h = FakeStoreHarness()
    store = h.store()
    h.commit_raises = google_exceptions.RetryError(
        'deadline',
        google_exceptions.DeadlineExceeded('deadline of 60.0s exceeded'),
    )

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', _mk('snap-1'))
    assert exc_info.value.status == 'DEADLINE_EXCEEDED'
    assert 'deadline of 60.0s exceeded' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_rejects_unstorable_state() -> None:
    """Circular / non-JSON state fails as INVALID_ARGUMENT, not a raw encoder error."""
    h = FakeStoreHarness()
    store = h.store()
    cyclic: dict[str, Any] = {}
    cyclic['self'] = cyclic

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'snap-1',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-1',
                session_id='sess-1',
                created_at='2026-07-03T00:00:00Z',
                status=SnapshotStatus.COMPLETED,
                state=SessionState(session_id='sess-1', custom=cyclic),
            ),
        )
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'not storable' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_rejects_lone_surrogate_state() -> None:
    """Lone surrogates in session state are rejected as not storable."""
    h = FakeStoreHarness()
    store = h.store()

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'snap-1',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-1',
                session_id='sess-1',
                created_at='2026-07-03T00:00:00Z',
                status=SnapshotStatus.COMPLETED,
                state=SessionState(session_id='sess-1', custom={'bad': '\ud800'}),
            ),
        )
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'not storable' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_corrupt_shard_json_is_data_loss() -> None:
    """Unparseable checkpoint shard bytes surface as DATA_LOSS."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1', custom={'x': 1}))
    h.docs[_shard_path('snap-1', 0)] = {'chunk': b'not-json{'}

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-1')
    assert exc_info.value.status == 'DATA_LOSS'
    assert 'corrupt' in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_firestore_session_store_async_mutator_rejected() -> None:
    """Async save mutators are a typed INVALID_ARGUMENT, not a hung coroutine."""
    h = FakeStoreHarness()
    store = h.store()

    async def bad(_existing: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-1'),
        )

    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot('snap-1', bad)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'synchronous' in str(exc_info.value)


@pytest.mark.asyncio
async def test_firestore_session_store_rejects_whitespace_ids() -> None:
    """Leading/trailing whitespace in ids is INVALID_ARGUMENT on save and get."""
    h = FakeStoreHarness()
    store = h.store()

    with pytest.raises(GenkitError) as e1:
        await store.save_snapshot(' snap-1 ', _mk('snap-1'))
    assert e1.value.status == 'INVALID_ARGUMENT'
    assert 'whitespace' in str(e1.value)

    with pytest.raises(GenkitError) as e2:
        await store.get_snapshot(snapshot_id=' snap-1 ')
    assert e2.value.status == 'INVALID_ARGUMENT'
    assert 'whitespace' in str(e2.value)


@pytest.mark.asyncio
async def test_firestore_session_store_aclose_tears_down_subscription() -> None:
    """Closing a stream drops its queue and stops the watch when it was the last consumer."""
    h = FakeStoreHarness()
    store = h.store()
    watches, _captured = _wire_sync_watch(store)
    stream = await store.on_snapshot_status_change('snap-1')
    assert len(store._subscriptions) == 1
    assert all(s.watch is not None for s in store._subscriptions)

    await stream.aclose()  # type: ignore[attr-defined]
    assert len(store._subscriptions) == 0
    watches[0].unsubscribe.assert_called_once()


@pytest.mark.asyncio
async def test_firestore_session_store_aclose_keeps_watch_while_others_remain() -> None:
    """Closing one stream unsubscribes only its watch; siblings keep theirs."""
    h = FakeStoreHarness()
    store = h.store()
    watches, _captured = _wire_sync_watch(store)
    first = await store.on_snapshot_status_change('snap-1')
    second = await store.on_snapshot_status_change('snap-1')
    assert len(store._subscriptions) == 2

    await first.aclose()  # type: ignore[attr-defined]
    assert len(store._subscriptions) == 1
    watches[0].unsubscribe.assert_called_once()
    watches[1].unsubscribe.assert_not_called()

    await second.aclose()  # type: ignore[attr-defined]
    assert len(store._subscriptions) == 0
    watches[1].unsubscribe.assert_called_once()


@pytest.mark.asyncio
async def test_firestore_session_store_aclose_after_terminal_is_idempotent() -> None:
    """aclose after a terminal teardown is a no-op, not an error."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    watches, captured_cb = _wire_sync_watch(store)
    stream = await store.on_snapshot_status_change('snap-1')
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'completed'))
    assert await _next_status(stream) == SnapshotStatus.COMPLETED
    assert await _stream_ended(stream)
    await stream.aclose()  # type: ignore[attr-defined]
    await stream.aclose()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_firestore_session_store_async_with_break_tears_down() -> None:
    """async with + break is the blessed early-exit shape and releases the watch."""
    h = FakeStoreHarness()
    store = h.store()
    watches, captured_cb = _wire_sync_watch(store)
    await store.save_snapshot(
        'snap-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-1',
            session_id='sess-1',
            created_at='2026-07-03T00:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-1'),
        ),
    )
    stream = await store.on_snapshot_status_change('snap-1')
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'pending'))
    async with stream:  # type: ignore[attr-defined]
        async for _status in stream:
            break
    assert len(store._subscriptions) == 0
    watches[0].unsubscribe.assert_called_once()


@pytest.mark.asyncio
async def test_firestore_session_store_lock_is_private() -> None:
    """The subscriber lock is private; SessionStore no longer advertises lock."""
    h = FakeStoreHarness()
    store = h.store()
    assert isinstance(store._lock, asyncio.Lock)
    assert not hasattr(store, 'lock')
    # Structural SessionStore surface still present (Protocol is not runtime_checkable).
    assert callable(store.get_snapshot) and callable(store.save_snapshot)


def _pending_mk(sid: str) -> Any:  # noqa: ANN401
    return lambda _e: SessionSnapshot(
        snapshot_id=sid,
        session_id='sess-1',
        created_at='2026-07-03T00:00:00Z',
        status=SnapshotStatus.PENDING,
        state=SessionState(session_id='sess-1'),
    )


@pytest.mark.asyncio
async def test_firestore_session_store_close_unsubscribes_inline_normally() -> None:
    """Normal close() unsubscribes the watch once on the calling thread."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _pending_mk('snap-1'))
    watches, captured_cb = _wire_sync_watch(store)
    calls: list[threading.Thread] = []

    def unsubscribe() -> None:
        calls.append(threading.current_thread())

    stream = await store.on_snapshot_status_change('snap-1')
    watches[0].unsubscribe.side_effect = unsubscribe
    assert len(store._subscriptions) == 1
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'pending'))
    assert await stream.__anext__() == SnapshotStatus.PENDING
    store.close()
    assert len(calls) == 1
    assert calls[0] is threading.current_thread()
    assert len(store._subscriptions) == 0
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_firestore_session_store_close_from_listener_thread_retries_off_thread() -> None:
    """close() on the watch listener thread finishes unsubscribe on a helper thread."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _pending_mk('snap-1'))

    listener = threading.current_thread()
    unsub_threads: list[threading.Thread] = []
    done = threading.Event()

    class FakeConsumer:
        _thread = listener

    watches, captured_cb = _wire_sync_watch(store)

    stream = await store.on_snapshot_status_change('snap-1')
    watches[0]._consumer = FakeConsumer()

    def unsubscribe() -> None:
        unsub_threads.append(threading.current_thread())
        done.set()

    watches[0].unsubscribe.side_effect = unsubscribe
    assert len(store._subscriptions) == 1
    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'pending'))
    assert await stream.__anext__() == SnapshotStatus.PENDING
    store.close()
    assert done.wait(timeout=2.0)
    assert unsub_threads and all(t is not listener for t in unsub_threads)
    assert len(store._subscriptions) == 0
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_firestore_session_store_watch_dedupes_repeated_status() -> None:
    """A watch replay of the same status yields one event, not duplicates."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _pending_mk('snap-1'))
    watches, captured = _wire_sync_watch(store)
    stream = await store.on_snapshot_status_change('snap-1')
    pending = _status_doc('snap-1', 'pending')
    await _pump_watch(captured, 0, pending)
    assert await stream.__anext__() == SnapshotStatus.PENDING
    await _pump_watch(captured, 0, pending)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.__anext__(), timeout=0.05)
    await _pump_watch(captured, 0, _status_doc('snap-1', 'completed'))
    assert await stream.__anext__() == SnapshotStatus.COMPLETED
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    watches[0].unsubscribe.assert_called_once()


def test_firestore_session_store_subscribe_works_after_prior_loop_dies() -> None:
    """After a notebook-style asyncio.run cell, a new loop can subscribe cleanly."""
    h = FakeStoreHarness()
    store = h.store()
    errors: list[BaseException] = []

    def run_first() -> None:
        async def body() -> None:
            await store.save_snapshot('snap-1', _pending_mk('snap-1'))
            _wire_sync_watch(store)
            await store.on_snapshot_status_change('snap-1')

        try:
            asyncio.run(body())
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=run_first)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()
    assert errors == []

    async def second() -> None:
        store.sync_client = None
        _wire_sync_watch(store)
        stream = await store.on_snapshot_status_change('snap-1')
        assert len(store._subscriptions) == 1
        await stream.aclose()

    asyncio.run(second())


def test_firestore_session_store_new_loop_prunes_dead_watch_and_starts_fresh() -> None:
    """A later loop prunes the dead loop's watch and starts its own listener."""
    h = FakeStoreHarness()
    store = h.store()
    unsub_calls: list[int] = []

    def run_first() -> None:
        async def body() -> None:
            await store.save_snapshot('snap-1', _pending_mk('snap-1'))
            watches, _ = _wire_sync_watch(store)
            await store.on_snapshot_status_change('snap-1')

            def track_unsub() -> None:
                unsub_calls.append(1)

            watches[0].unsubscribe.side_effect = track_unsub
            # Notebook pattern: cell ends without aclose.

        asyncio.run(body())

    t = threading.Thread(target=run_first)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive()

    got_completed = False

    async def second() -> None:
        nonlocal got_completed
        # Drop the loop-A-derived sync client so the harness mock rebinds.
        store.sync_client = None
        watches2, captured2 = _wire_sync_watch(store)
        stream = await store.on_snapshot_status_change('snap-1')
        assert len(store._subscriptions) == 1
        assert len(captured2) == 1, 'new loop must start a fresh watch'
        assert unsub_calls, 'dead loop watch must be unsubscribed on prune'
        await _pump_watch(captured2, 0, _status_doc('snap-1', 'pending'))
        assert await stream.__anext__() == SnapshotStatus.PENDING

        await _pump_watch(captured2, 0, _status_doc('snap-1', 'completed'))
        assert await stream.__anext__() == SnapshotStatus.COMPLETED
        got_completed = True
        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    asyncio.run(second())
    assert got_completed


def test_firestore_session_store_two_live_loops_have_independent_subscriptions() -> None:
    """App loop + Dev UI loop can each subscribe on the same store instance."""
    h = FakeStoreHarness()
    store = h.store()
    ready = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []
    loop_a_got: list[SnapshotStatus] = []

    def hold_loop() -> None:
        async def body() -> None:
            await store.save_snapshot('snap-1', _pending_mk('snap-1'))
            _watches, captured = _wire_sync_watch(store)
            stream = await store.on_snapshot_status_change('snap-1')
            await _pump_watch(captured, 0, _status_doc('snap-1', 'pending'))
            loop_a_got.append(await stream.__anext__())
            ready.set()
            await asyncio.get_running_loop().run_in_executor(None, release.wait)
            await stream.aclose()

        try:
            asyncio.run(body())
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=hold_loop)
    t.start()
    assert ready.wait(timeout=5.0)

    async def other() -> None:
        store.sync_client = None
        _watches, captured = _wire_sync_watch(store)
        stream = await store.on_snapshot_status_change('snap-1')
        await _pump_watch(captured, 0, _status_doc('snap-1', 'pending'))
        assert await stream.__anext__() == SnapshotStatus.PENDING
        assert len(store._subscriptions) == 1
        await stream.aclose()

    try:
        asyncio.run(other())
    finally:
        release.set()
        t.join(timeout=5.0)
    assert errors == []
    assert loop_a_got == [SnapshotStatus.PENDING]


def test_firestore_session_store_clients_are_loop_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provided async client binds to the first loop; later loops get a clone + own sync."""
    clones: list[MagicMock] = []

    def fake_async_client(**kwargs: Any) -> MagicMock:  # noqa: ANN401
        m = MagicMock(name=f'async-clone-{len(clones)}')
        m.project = kwargs['project']
        m._database = kwargs.get('database')
        m._to_sync_copy = MagicMock(return_value=MagicMock(name=f'sync-clone-{len(clones)}'))
        clones.append(m)
        return m

    monkeypatch.setattr(
        'genkit_google_cloud.session_store.firestore.firestore.AsyncClient',
        fake_async_client,
    )

    provided = MagicMock(name='async-provided')
    provided.project = 'proj-a'
    provided._database = 'db-a'
    provided._credentials = object()
    provided._to_sync_copy = MagicMock(return_value=MagicMock(name='sync-provided'))

    store = FirestoreSessionStore(client=provided)
    ready = threading.Event()
    release = threading.Event()
    clients_a: list[Any] = []
    syncs_a: list[Any] = []
    clients_b: list[Any] = []
    syncs_b: list[Any] = []

    def hold_loop() -> None:
        async def body() -> None:
            clients_a.append(store.client)
            syncs_a.append(store._ensure_sync_client())
            ready.set()
            await asyncio.get_running_loop().run_in_executor(None, release.wait)

        asyncio.run(body())

    t = threading.Thread(target=hold_loop)
    t.start()
    assert ready.wait(timeout=5.0)

    async def other() -> None:
        clients_b.append(store.client)
        syncs_b.append(store._ensure_sync_client())

    try:
        asyncio.run(other())
    finally:
        release.set()
        t.join(timeout=5.0)

    assert clients_a == [provided]
    assert clients_b and clients_b[0] is not provided
    assert clients_b[0] in clones
    assert syncs_a and syncs_b and syncs_a[0] is not syncs_b[0]


@pytest.mark.asyncio
async def test_firestore_session_store_prefix_captured_before_awaits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutating context mid-save cannot split snapshot/shards/pointer across tenants."""
    h = FakeStoreHarness()
    ctx: dict[str, Any] = {'tenant': 't-orig'}
    store = h.store(snapshot_path_prefix=lambda c: (c or {}).get('tenant', 'global'))

    gate = asyncio.Event()
    inside = asyncio.Event()
    real_reconstruct = store._reconstruct
    seen: list[str] = []

    async def gated_reconstruct(transaction: Any, snapshot_id: str, *, prefix: str) -> Any:  # noqa: ANN401
        seen.append(prefix)
        if len(seen) == 1:
            inside.set()
            await gate.wait()
            ctx['tenant'] = 't-mutated'
        return await real_reconstruct(transaction, snapshot_id, prefix=prefix)

    monkeypatch.setattr(store, '_reconstruct', gated_reconstruct)

    task = asyncio.create_task(store.save_snapshot('snap-1', _mk('snap-1', custom={'n': 1}), context=ctx))
    await inside.wait()
    assert ctx['tenant'] == 't-orig'  # not yet mutated
    gate.set()
    saved = await task
    assert saved is not None
    assert seen[0] == 't-orig'
    assert _snap_path('snap-1', prefix='t-orig') in h.docs
    assert _pointer_path('sess-1', prefix='t-orig') in h.docs
    assert _shard_path('snap-1', 0, prefix='t-orig') in h.docs
    assert not any('/t-mutated/' in path for path in h.docs)


@pytest.mark.asyncio
async def test_firestore_session_store_no_context_defaults_to_global_prefix() -> None:
    """Context-free apps keep writing under the default ``global`` prefix."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _mk('snap-1'))
    assert _snap_path('snap-1', prefix='global') in h.docs
    assert _pointer_path('sess-1', prefix='global') in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_ambient_context_shared_across_ops() -> None:
    """get/save/subscribe with no context= all honor the ambient action context."""
    from genkit._core._action import _action_context

    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=lambda c: (c or {}).get('tenant', 'global'))
    _watches, captured_cb = _wire_sync_watch(store)
    token = _action_context.set({'tenant': 'ambient-t'})
    try:
        await store.save_snapshot(
            'snap-1',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-1',
                session_id='sess-1',
                created_at='2026-07-03T00:00:00Z',
                status=SnapshotStatus.PENDING,
                state=SessionState(session_id='sess-1'),
            ),
        )
        loaded = await store.get_snapshot(snapshot_id='snap-1')
        stream = await store.on_snapshot_status_change('snap-1')
        assert len(store._subscriptions) == 1
        await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'pending'))
        assert await _next_status(stream) == SnapshotStatus.PENDING
    finally:
        _action_context.reset(token)

    assert loaded is not None and loaded.snapshot_id == 'snap-1'
    assert _snap_path('snap-1', prefix='ambient-t') in h.docs
    assert _snap_path('snap-1', prefix='global') not in h.docs


@pytest.mark.asyncio
async def test_firestore_session_store_status_observed_via_watch_not_save() -> None:
    """Status streams only advance when the Firestore watch fires, not on save."""
    h = FakeStoreHarness()
    store = h.store()
    await store.save_snapshot('snap-1', _pending_mk('snap-1'))
    watches, captured_cb = _wire_sync_watch(store)
    stream = await store.on_snapshot_status_change('snap-1')

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.__anext__(), timeout=0.05)

    def to_completed(_e: SessionSnapshot | None) -> SessionSnapshot:
        base = _pending_mk('snap-1')(_e)
        return base.model_copy(update={'status': SnapshotStatus.COMPLETED})

    await store.save_snapshot('snap-1', to_completed)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.__anext__(), timeout=0.05)

    await _pump_watch(captured_cb, 0, _status_doc('snap-1', 'completed'))
    assert await stream.__anext__() == SnapshotStatus.COMPLETED
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()
    watches[0].unsubscribe.assert_called_once()
