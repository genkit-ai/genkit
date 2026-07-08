# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for FirestoreSessionStore."""

from unittest.mock import MagicMock

import pytest
from genkit._core._typing import SessionSnapshot
from genkit.plugins.google_cloud.session_store.firestore import FirestoreSessionStore


@pytest.mark.asyncio
async def test_firestore_session_store_save_and_get() -> None:
    """Test saving and getting a session snapshot using a mock Firestore client."""
    mock_client = MagicMock()
    mock_doc_snapshot = MagicMock()

    # Configure mock chain so any ref.get() returns mock_doc_snapshot
    mock_doc_ref = MagicMock()
    mock_doc_ref.get.return_value = mock_doc_snapshot

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

    # Test get_snapshot
    retrieved = await store.get_snapshot(snapshot_id='snap-123')
    assert retrieved is not None
    assert retrieved.snapshot_id == 'snap-123'
    assert retrieved.session_id == 'sess-456'

    # Test save_snapshot
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
