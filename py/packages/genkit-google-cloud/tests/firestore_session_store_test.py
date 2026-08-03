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
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_cloud import FirestoreSessionStore
from google.cloud import firestore

from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SessionState, SnapshotStatus


def _mock_txn_client() -> tuple[MagicMock, MagicMock]:
    """Return (client, transaction) with async transactional plumbing mocked."""
    mock_client = MagicMock()
    mock_transaction = MagicMock()
    mock_transaction._max_attempts = 1
    mock_transaction._read_only = False
    mock_transaction._begin = AsyncMock()
    mock_transaction._commit = AsyncMock()
    mock_transaction._rollback = AsyncMock()
    mock_client.transaction.return_value = mock_transaction
    return mock_client, mock_transaction


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


class FakeStoreHarness:
    """In-memory Firestore stand-in wired for AsyncClient-style access."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.client, self.transaction = _mock_txn_client()
        self.deleted: list[str] = []

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

        async def txn_get_all(refs: list[Any]) -> Any:  # noqa: ANN401
            for ref in refs:
                path = ref.path
                if path in self.docs:
                    yield _doc(path=path, exists=True, data=self.docs[path], doc_id=ref.id)
                else:
                    yield _doc(path=path, exists=False, data=None, doc_id=ref.id)

        self.transaction.get_all = txn_get_all

        def txn_set(ref: Any, data: dict[str, Any]) -> None:  # noqa: ANN401
            self.docs[ref.path] = dict(data)

        def txn_update(ref: Any, data: dict[str, Any]) -> None:  # noqa: ANN401
            current = dict(self.docs.get(ref.path, {}))
            for key, value in data.items():
                if value is firestore.DELETE_FIELD:
                    current.pop(key, None)
                elif value is firestore.SERVER_TIMESTAMP:
                    current[key] = 'SERVER_TIMESTAMP'
                else:
                    current[key] = value
            self.docs[ref.path] = current

        def txn_delete(ref: Any) -> None:  # noqa: ANN401
            self.docs.pop(ref.path, None)
            self.deleted.append(ref.path)

        self.transaction.set = txn_set
        self.transaction.update = txn_update
        self.transaction.delete = txn_delete

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
    assert pointer['leaves'] == {'snap-1': '2026-07-03T00:00:00Z'}


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
    assert child['statePatch']

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
    assert pointer['isAmbiguous'] is False
    assert pointer['leaves'] == {'snap-1': '2026-07-03T00:00:00Z'}


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
    assert pointer_after['leaves'] == pointer_before['leaves']
    assert pointer_after['isAmbiguous'] is False
    assert pointer_after['currentSnapshotId'] == 'snap-B'


@pytest.mark.asyncio
async def test_firestore_session_store_branching_ambiguity() -> None:
    """Two tips from different parents mark the session ambiguous."""
    h = FakeStoreHarness()
    store = h.store(reject_ambiguous_session=True)

    await store.save_snapshot(
        'snap-root',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-root',
            session_id='sess-branch-1',
            created_at='2026-07-03T00:00:00Z',
            state=SessionState(session_id='sess-branch-1'),
        ),
    )
    await store.save_snapshot(
        'snap-existing',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-existing',
            parent_id='snap-root',
            session_id='sess-branch-1',
            created_at='2026-07-03T00:00:01Z',
            state=SessionState(session_id='sess-branch-1', custom={'branch': 'a'}),
        ),
    )
    await store.save_snapshot(
        'snap-branch',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-branch',
            parent_id='snap-root',
            session_id='sess-branch-1',
            created_at='2026-07-03T00:00:02Z',
            state=SessionState(session_id='sess-branch-1', custom={'branch': 'b'}),
        ),
    )

    pointer = h.docs[_pointer_path('sess-branch-1')]
    assert pointer['isAmbiguous'] is True
    assert set(pointer['leaves']) == {'snap-existing', 'snap-branch'}

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-branch-1')
    assert exc_info.value.status == 'FAILED_PRECONDITION'

    store_permissive = h.store(reject_ambiguous_session=False)
    resolved = await store_permissive.get_snapshot(session_id='sess-branch-1')
    assert resolved is not None
    assert resolved.snapshot_id == 'snap-branch'


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
    h.docs[_pointer_path('sess-corrupt')] = {'currentSnapshotId': 'snap-deleted'}

    store = h.store()
    assert await store.get_snapshot(session_id='sess-corrupt') is None
    assert h.docs[_pointer_path('sess-corrupt')]['currentSnapshotId'] == 'snap-deleted'


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

    watch_mock = MagicMock()
    captured_cb: list[Any] = []

    def on_snapshot_side_effect(cb: Any) -> Any:  # noqa: ANN401
        captured_cb.append(cb)
        return watch_mock

    mock_sync_doc_ref = MagicMock()
    mock_sync_doc_ref.on_snapshot.side_effect = on_snapshot_side_effect
    mock_sync_col = MagicMock()
    mock_sync_col.document.return_value = mock_sync_doc_ref
    mock_sync_doc_ref.collection.return_value = mock_sync_col
    mock_sync_client = MagicMock()
    mock_sync_client.collection.return_value = mock_sync_col

    with patch('genkit_google_cloud.session_store.firestore.firestore.Client', return_value=mock_sync_client):
        queue = await store.on_snapshot_status_change('snap-sub')
        assert await queue.get() == SnapshotStatus.PENDING
        assert len(captured_cb) == 1

        terminal_doc = MagicMock()
        terminal_doc.exists = True
        terminal_doc.to_dict.return_value = {
            'snapshotId': 'snap-sub',
            'sessionId': 'sess-1',
            'createdAt': '2026-07-03T00:00:00Z',
            'status': 'aborted',
        }

        captured_cb[0]([terminal_doc], None, None)
        await asyncio.sleep(0.05)

        assert await queue.get() == SnapshotStatus.ABORTED
        assert await queue.get() is None
        watch_mock.unsubscribe.assert_called_once()
        assert 'snap-sub' not in store.subs
        assert 'snap-sub' not in store._watches


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

    watch_mock = MagicMock()
    mock_sync_doc_ref = MagicMock()
    mock_sync_doc_ref.on_snapshot.return_value = watch_mock
    mock_sync_col = MagicMock()
    mock_sync_col.document.return_value = mock_sync_doc_ref
    mock_sync_doc_ref.collection.return_value = mock_sync_col
    mock_sync_client = MagicMock()
    mock_sync_client.collection.return_value = mock_sync_col

    with patch('genkit_google_cloud.session_store.firestore.firestore.Client', return_value=mock_sync_client):
        await store.on_snapshot_status_change('snap-close')
        assert store.sync_client is mock_sync_client
        assert 'snap-close' in store._watches

        store.close()
        watch_mock.unsubscribe.assert_called_once()
        mock_sync_client.close.assert_called_once()
        assert store.sync_client is None
        assert store._watches == {}
        assert store.subs == {}


@pytest.mark.asyncio
async def test_firestore_session_store_close_does_not_close_injected_sync_client() -> None:
    """Caller-owned sync_client is left open by close()."""
    h = FakeStoreHarness()
    injected = MagicMock()
    store = h.store(sync_client=injected)
    store.close()
    injected.close.assert_not_called()
    assert store.sync_client is injected


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
