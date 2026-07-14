# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for FirestoreSessionStore."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_cloud.session_store import FirestoreSessionStore
from google.cloud import firestore

from genkit._ai._agents._session import SessionErrorType
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus


@pytest.mark.asyncio
async def test_firestore_session_store_save_and_get() -> None:
    """Test saving and getting a session snapshot using a mock Firestore client."""
    mock_client = MagicMock()
    mock_transaction = MagicMock()
    mock_transaction._max_attempts = 1
    mock_transaction._read_only = False
    mock_transaction._begin = AsyncMock()
    mock_transaction._commit = AsyncMock()
    mock_transaction._rollback = AsyncMock()
    mock_client.transaction.return_value = mock_transaction
    mock_doc_snapshot = MagicMock()

    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)
    mock_doc_ref.set = AsyncMock()

    mock_col = MagicMock()
    mock_col.document.return_value = mock_doc_ref

    mock_doc_ref.collection.return_value = mock_col
    mock_client.collection.return_value = mock_col

    store = FirestoreSessionStore(client=mock_client)

    snapshot_data = {
        'snapshotId': 'snap-123',
        'sessionId': 'sess-456',
        'createdAt': '2026-07-03T00:00:00Z',
    }
    mock_doc_snapshot.exists = True
    mock_doc_snapshot.to_dict.return_value = snapshot_data

    retrieved = await store.get_snapshot(snapshot_id='snap-123')
    assert retrieved is not None
    assert retrieved.snapshot_id == 'snap-123'
    assert retrieved.session_id == 'sess-456'

    mock_doc_snapshot.exists = False
    mock_doc_snapshot.to_dict.return_value = None

    def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-789',
            session_id='sess-456',
            created_at='2026-07-03T00:00:01Z',
        )

    saved = await store.save_snapshot(None, save_fn)
    assert saved is not None
    assert saved.session_id == 'sess-456'
    assert isinstance(saved.snapshot_id, str)
    mock_doc_ref.set.assert_called()
    mock_transaction.set.assert_called_once()


@pytest.mark.asyncio
async def test_firestore_session_store_status_change_and_cleanup() -> None:
    """Test snapshot status subscription and thread-safe cleanup on terminal status."""
    mock_client = MagicMock()
    mock_doc_snapshot = MagicMock()
    mock_doc_ref = MagicMock()
    mock_doc_ref.get = AsyncMock(return_value=mock_doc_snapshot)
    mock_col = MagicMock()
    mock_col.document.return_value = mock_doc_ref
    mock_doc_ref.collection.return_value = mock_col
    mock_client.collection.return_value = mock_col

    # Delete on_snapshot to force the fallback to sync_client
    del mock_doc_ref.on_snapshot

    store = FirestoreSessionStore(client=mock_client)

    mock_doc_snapshot.exists = True
    mock_doc_snapshot.to_dict.return_value = {
        'snapshotId': 'snap-sub',
        'sessionId': 'sess-1',
        'createdAt': '2026-07-03T00:00:00Z',
        'status': 'pending',
    }

    watch_mock = MagicMock()
    captured_cb = []

    def on_snapshot_side_effect(cb: Any) -> Any:  # noqa: ANN401
        captured_cb.append(cb)
        return watch_mock

    # Mock the sync Client to return a mock document ref that has on_snapshot
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


@pytest.mark.asyncio
async def test_firestore_session_store_update_existing_pointer() -> None:
    """Test that pointer is updated correctly when saving a snapshot on top of an existing one."""
    mock_client = MagicMock()
    mock_transaction = MagicMock()
    mock_transaction._max_attempts = 1
    mock_transaction._read_only = False
    mock_transaction._begin = AsyncMock()
    mock_transaction._commit = AsyncMock()
    mock_transaction._rollback = AsyncMock()
    mock_client.transaction.return_value = mock_transaction

    snap_col = MagicMock()
    snap_doc_ref = MagicMock()
    snap_col.document.return_value = snap_doc_ref
    snap_doc_ref.collection.return_value = snap_col

    pointer_col = MagicMock()
    pointer_doc_ref = MagicMock()
    pointer_col.document.return_value = pointer_doc_ref
    pointer_doc_ref.collection.return_value = pointer_col

    def collection_side_effect(col_name: str) -> Any:  # noqa: ANN401
        if 'pointer' in col_name:
            return pointer_col
        return snap_col

    mock_client.collection.side_effect = collection_side_effect

    snap_snapshot = MagicMock()
    snap_snapshot.exists = True
    snap_snapshot.to_dict.return_value = {
        'snapshotId': 'snap-123',
        'sessionId': 'sess-456',
        'createdAt': '2026-07-03T00:00:00Z',
    }
    snap_doc_ref.get = AsyncMock(return_value=snap_snapshot)
    snap_doc_ref.set = AsyncMock()

    pointer_snapshot = MagicMock()
    pointer_snapshot.exists = True
    pointer_snapshot.to_dict.return_value = {
        'currentSnapshotId': 'snap-123',
    }
    pointer_doc_ref.get = AsyncMock(return_value=pointer_snapshot)

    store = FirestoreSessionStore(client=mock_client)

    def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-789',
            parent_id='snap-123',
            session_id='sess-456',
            created_at='2026-07-03T00:00:01Z',
        )

    saved = await store.save_snapshot(None, save_fn)
    assert saved is not None
    assert isinstance(saved.snapshot_id, str)
    mock_transaction.set.assert_called_once()
    assert mock_transaction.set.call_args[0][1]['currentSnapshotId'] == saved.snapshot_id


@pytest.mark.asyncio
async def test_firestore_session_store_branching_ambiguity() -> None:
    """Test pointer update setting isAmbiguous on branch split, and get_snapshot raising when reject_ambiguous is True."""
    mock_client = MagicMock()
    mock_transaction = MagicMock()
    mock_transaction._max_attempts = 1
    mock_transaction._read_only = False
    mock_transaction._begin = AsyncMock()
    mock_transaction._commit = AsyncMock()
    mock_transaction._rollback = AsyncMock()
    mock_client.transaction.return_value = mock_transaction

    snap_doc_ref = MagicMock()
    snap_doc_ref.set = AsyncMock()
    snap_snapshot = MagicMock()
    snap_snapshot.exists = False
    snap_doc_ref.get = AsyncMock(return_value=snap_snapshot)

    pointer_doc_ref = MagicMock()
    pointer_doc_ref.set = AsyncMock()
    pointer_snapshot = MagicMock()
    pointer_snapshot.exists = True
    pointer_snapshot.to_dict.return_value = {
        'currentSnapshotId': 'snap-existing',
    }
    pointer_doc_ref.get = AsyncMock(return_value=pointer_snapshot)

    mock_doc_ref = MagicMock()
    mock_col = MagicMock()
    def get_doc(doc_id: str) -> Any:  # noqa: ANN401
        if doc_id == 'sess-branch-1':
            return pointer_doc_ref
        if doc_id == 'global':
            return mock_doc_ref
        return snap_doc_ref
    mock_col.document.side_effect = get_doc
    pointer_doc_ref.collection.return_value = mock_col
    snap_doc_ref.collection.return_value = mock_col
    mock_doc_ref.collection.return_value = mock_col
    mock_client.collection.return_value = mock_col

    store = FirestoreSessionStore(client=mock_client, reject_ambiguous_session=True)

    def save_fn(existing: SessionSnapshot | None) -> SessionSnapshot:
        return SessionSnapshot(
            snapshot_id='snap-branch',
            parent_id='snap-root',
            session_id='sess-branch-1',
            created_at='2026-07-03T00:00:02Z',
        )

    saved = await store.save_snapshot(None, save_fn)
    assert saved is not None
    mock_transaction.update.assert_called_once()
    update_payload = mock_transaction.update.call_args[0][1]
    assert update_payload['isAmbiguous'] is True
    assert update_payload['currentSnapshotId'] == firestore.DELETE_FIELD

    # Now verify get_snapshot raises early when isAmbiguous=True on pointer doc:
    pointer_snapshot.to_dict.return_value = {
        'isAmbiguous': True,
    }
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='sess-branch-1')
    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert exc_info.value.details is not None
    assert exc_info.value.details['type'] == SessionErrorType.AMBIGUOUS_BRANCH
