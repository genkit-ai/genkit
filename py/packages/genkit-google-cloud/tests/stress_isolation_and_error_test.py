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

"""Stress, multi-tenant isolation, and corrupt document recovery tests for FirestoreSessionStore."""

from __future__ import annotations

import asyncio
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
                prefix_ref = MagicMock()

                def subcollection(sub: str) -> MagicMock:
                    sub_col = MagicMock()

                    def item_document(item_id: str) -> MagicMock:
                        path = f'{col_name}/{doc_id}/{sub}/{item_id}'
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


def _user_prefix(context: dict[str, Any] | None) -> str:
    if not isinstance(context, dict):
        return 'anonymous'
    return str(context.get('user_id', 'anonymous'))


@pytest.mark.asyncio
async def test_multi_tenant_context_isolation_stress() -> None:
    """Stress test tenant isolation with 50+ interleaved save/get calls using identical session_ids across tenants."""
    h = FakeStoreHarness()
    store = h.store(snapshot_path_prefix=_user_prefix)

    shared_session_id = 'common-session-id'
    tenants = [f'user_{i}' for i in range(10)]

    # Track expected latest state for each tenant
    tenant_states: dict[str, dict[str, Any]] = {t: {} for t in tenants}
    tenant_last_snap_id: dict[str, str | None] = {t: None for t in tenants}

    # Interleaved execution: 6 rounds, operating across all 10 tenants (60 operations total)
    for round_idx in range(6):
        tasks = []
        for t_idx, tenant in enumerate(tenants):
            ctx = {'user_id': tenant}
            snap_id = f'snap-{tenant}-r{round_idx}'
            parent_id = tenant_last_snap_id[tenant]
            tenant_last_snap_id[tenant] = snap_id
            state_data = {'tenant': tenant, 'round': round_idx, 'counter': round_idx * 100 + t_idx}
            tenant_states[tenant] = state_data

            def make_save_fn(s_id: str, p_id: str | None, data: dict[str, Any], r_idx: int):
                def save_fn(_e: SessionSnapshot | None) -> SessionSnapshot:
                    return SessionSnapshot(
                        snapshot_id=s_id,
                        parent_id=p_id,
                        session_id=shared_session_id,
                        created_at=f'2026-08-03T00:0{r_idx}:00Z',
                        status=SnapshotStatus.COMPLETED,
                        state=SessionState(session_id=shared_session_id, custom=data),
                    )

                return save_fn

            tasks.append(
                store.save_snapshot(
                    snap_id,
                    make_save_fn(snap_id, parent_id, state_data, round_idx),
                    context=ctx,
                )
            )

        # Execute round of saves concurrently across all tenants
        saved_snapshots = await asyncio.gather(*tasks)
        assert len(saved_snapshots) == len(tenants)

        # Verification pass for each tenant in this round
        get_tasks = [
            store.get_snapshot(session_id=shared_session_id, context={'user_id': tenant}) for tenant in tenants
        ]
        retrieved_snapshots = await asyncio.gather(*get_tasks)

        for tenant, loaded in zip(tenants, retrieved_snapshots, strict=True):
            assert loaded is not None, f'Tenant {tenant} snapshot should exist'
            assert loaded.session_id == shared_session_id
            assert loaded.snapshot_id == tenant_last_snap_id[tenant]
            assert loaded.state is not None
            assert loaded.state.custom == tenant_states[tenant], (
                f'Tenant {tenant} got state {loaded.state.custom}, expected {tenant_states[tenant]}'
            )

    # Verify document key paths in store: each tenant should have distinct snapshot paths
    for tenant in tenants:
        for r in range(6):
            expected_path = f'genkit-sessions/{tenant}/snapshots/snap-{tenant}-r{r}'
            assert expected_path in h.docs, f'Snapshot path {expected_path} missing'

        expected_pointer = f'genkit-sessions-pointers/{tenant}/pointers/{shared_session_id}'
        assert expected_pointer in h.docs, f'Pointer path {expected_pointer} missing'


@pytest.mark.asyncio
async def test_missing_checkpoint_shard_data_loss_exception() -> None:
    """Test exception and recovery when a checkpoint shard document is missing (DATA_LOSS exception)."""
    h = FakeStoreHarness()
    store = h.store()

    # Save a root checkpoint snapshot
    await store.save_snapshot(
        'snap-root',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-root',
            session_id='sess-loss-test',
            created_at='2026-08-03T00:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-loss-test', custom={'step': 1}),
        ),
    )

    shard_path = 'genkit-sessions-shards/global/shards/snap-root_0'
    assert shard_path in h.docs

    # Simulate shard data loss by removing shard document
    h.docs.pop(shard_path)

    # Reading snapshot by ID should raise GenkitError with DATA_LOSS
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-root')
    assert exc_info.value.status == 'DATA_LOSS'
    assert "missing checkpoint shard 'snap-root_0'" in exc_info.value.original_message

    # Reading snapshot by session_id should also raise GenkitError with DATA_LOSS
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-loss-test')
    assert exc_info.value.status == 'DATA_LOSS'

    # Attempting to save a child snapshot whose parent checkpoint shard is missing
    # will fail with DATA_LOSS when trying to reconstruct parent for diffing.
    with pytest.raises(GenkitError) as exc_info:
        await store.save_snapshot(
            'snap-child',
            lambda _e: SessionSnapshot(
                snapshot_id='snap-child',
                parent_id='snap-root',
                session_id='sess-loss-test',
                created_at='2026-08-03T00:01:00Z',
                state=SessionState(session_id='sess-loss-test', custom={'step': 2}),
            ),
        )
    assert exc_info.value.status == 'DATA_LOSS'

    # Recovery scenario: Write a new root checkpoint for the session without referencing missing parent
    recovered = await store.save_snapshot(
        'snap-recovered-root',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-recovered-root',
            session_id='sess-loss-test',
            created_at='2026-08-03T00:02:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-loss-test', custom={'recovered': True}),
        ),
    )

    assert recovered is not None
    assert recovered.snapshot_id == 'snap-recovered-root'

    # Now get_snapshot by session_id should succeed and return the recovered checkpoint
    loaded = await store.get_snapshot(session_id='sess-loss-test')
    assert loaded is not None
    assert loaded.snapshot_id == 'snap-recovered-root'
    assert loaded.state is not None
    assert loaded.state.custom == {'recovered': True}


@pytest.mark.asyncio
async def test_invalid_unapplyable_diff_patch_error_and_recovery() -> None:
    """Test error handling and recovery when a diff patch contains invalid/unapplyable ops."""
    h = FakeStoreHarness()
    store = h.store(checkpoint_interval=25)

    # Save a valid checkpoint
    await store.save_snapshot(
        'snap-checkpoint',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-checkpoint',
            session_id='sess-diff-corrupt',
            created_at='2026-08-03T00:00:00Z',
            state=SessionState(session_id='sess-diff-corrupt', custom={'a': 1}),
        ),
    )

    # Save a valid child diff
    await store.save_snapshot(
        'snap-diff-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-diff-1',
            parent_id='snap-checkpoint',
            session_id='sess-diff-corrupt',
            created_at='2026-08-03T00:01:00Z',
            state=SessionState(session_id='sess-diff-corrupt', custom={'a': 2}),
        ),
    )

    child_path = 'genkit-sessions/global/snapshots/snap-diff-1'
    assert child_path in h.docs

    # Corrupt statePatch with invalid patch op (failed TEST operation)
    h.docs[child_path]['statePatch'] = [
        {'op': 'test', 'path': '/custom/a', 'value': 999}  # Mismatch: actual is 1
    ]

    # Reconstructing snap-diff-1 should raise ValueError due to failed JSON Patch test op
    with pytest.raises(ValueError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-diff-1')
    assert 'JSON Patch test failed' in str(exc_info.value)

    # Corrupt statePatch with invalid JSON Pointer (missing leading slash)
    h.docs[child_path]['statePatch'] = [{'op': 'add', 'path': 'invalid_pointer', 'value': 123}]
    with pytest.raises(ValueError) as exc_info:
        await store.get_snapshot(snapshot_id='snap-diff-1')
    assert 'must start with "/"' in str(exc_info.value)

    # Corrupt statePatch with invalid JSON Patch op name
    h.docs[child_path]['statePatch'] = [{'op': 'non_existent_op', 'path': '/custom/a', 'value': 123}]
    with pytest.raises((ValueError, GenkitError, Exception)):  # noqa: B017
        await store.get_snapshot(snapshot_id='snap-diff-1')

    # Recovery scenario: Create a new full checkpoint to reset state for the session
    recovered_snap = await store.save_snapshot(
        'snap-checkpoint-recovery',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-checkpoint-recovery',
            session_id='sess-diff-corrupt',
            created_at='2026-08-03T00:02:00Z',
            status=SnapshotStatus.COMPLETED,
            state=SessionState(session_id='sess-diff-corrupt', custom={'recovered': 'ok'}),
        ),
    )

    assert recovered_snap is not None
    assert recovered_snap.snapshot_id == 'snap-checkpoint-recovery'

    # Session lookup now successfully resolves to the new checkpoint
    loaded = await store.get_snapshot(session_id='sess-diff-corrupt')
    assert loaded is not None
    assert loaded.snapshot_id == 'snap-checkpoint-recovery'
    assert loaded.state is not None
    assert loaded.state.custom == {'recovered': 'ok'}


@pytest.mark.asyncio
async def test_listener_cleanup_repeated_close_calls() -> None:
    """Test listener cleanup when calling close() repeatedly while watches are active."""
    h = FakeStoreHarness()
    store = h.store()

    # Pre-populate 10 snapshots in store
    num_watches = 10
    snapshot_ids = [f'snap-watch-{i}' for i in range(num_watches)]
    for sid in snapshot_ids:
        await store.save_snapshot(
            sid,
            lambda _e, s_id=sid: SessionSnapshot(
                snapshot_id=s_id,
                session_id='sess-watch',
                created_at='2026-08-03T00:00:00Z',
                status=SnapshotStatus.PENDING,
                state=SessionState(session_id='sess-watch'),
            ),
        )

    # Set up mock sync client and watches
    watch_mocks = [MagicMock() for _ in range(num_watches)]
    watch_map = {}
    for idx, sid in enumerate(snapshot_ids):
        doc_ref = MagicMock()
        doc_ref.on_snapshot.return_value = watch_mocks[idx]
        watch_map[sid] = doc_ref

    mock_snapshots_col = MagicMock()
    mock_snapshots_col.document.side_effect = lambda sid: watch_map[sid]

    mock_prefix_doc = MagicMock()
    mock_prefix_doc.collection.return_value = mock_snapshots_col

    mock_genkit_col = MagicMock()
    mock_genkit_col.document.return_value = mock_prefix_doc

    mock_sync_client = MagicMock()
    mock_sync_client.collection.return_value = mock_genkit_col

    with patch('genkit_google_cloud.session_store.firestore.firestore.Client', return_value=mock_sync_client):
        # Register watches on all 10 snapshots
        for sid in snapshot_ids:
            await store.on_snapshot_status_change(sid)

        assert len(store._watches) == num_watches
        assert len(store.subs) == num_watches
        assert store.sync_client is mock_sync_client

        # Call close() first time
        store.close()

        # Assert all watch unsubscribe methods were called exactly once
        for wm in watch_mocks:
            wm.unsubscribe.assert_called_once()

        # Assert internal dicts and client were cleaned up
        assert len(store._watches) == 0
        assert len(store.subs) == 0
        assert store.sync_client is None

        # Call close() repeatedly (4 more times)
        for _ in range(4):
            store.close()

        # Ensure no additional unsubscribe or close calls were made
        for wm in watch_mocks:
            assert wm.unsubscribe.call_count == 1
        mock_sync_client.close.assert_called_once()
