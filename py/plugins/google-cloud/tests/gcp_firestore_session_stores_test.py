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

"""Unit tests for Firestore-backed session stores using a high-fidelity mock."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import google.cloud.firestore
import pytest

from genkit._ai._agents._session import StoreRecordKind
from genkit._ai._agents._session_stores._branching import BranchRecord
from genkit._ai._agents._session_stores._latest_state import LatestRecord
from genkit._ai._agents._session_stores._linear import TurnRecord
from genkit._core._typing import SessionSnapshot, SessionState, SnapshotStatus
from genkit.plugins.google_cloud.session_store import (
    FirestoreBranchingSessionStore,
    FirestoreLatestStateStore,
    FirestoreLinearSessionStore,
)

# ---------------------------------------------------------------------------
# High-Fidelity In-Memory Mock Firestore Client
# ---------------------------------------------------------------------------


class MockDocumentSnapshot:
    """Mock representing a Firestore DocumentSnapshot."""

    def __init__(self, exists: bool, data: dict[str, Any] | None = None) -> None:
        self.exists = exists
        self._data = data or {}

    def to_dict(self) -> dict[str, Any]:
        """Return document data as a dictionary."""
        return copy.deepcopy(self._data)


class MockDocumentReference:
    """Mock representing a Firestore DocumentReference."""

    def __init__(self, doc_id: str, collection: MockCollectionReference) -> None:
        self.id = doc_id
        self.collection = collection
        self._data: dict[str, Any] = {}
        self._exists = False

    async def get(self, transaction: MockTransaction | None = None) -> MockDocumentSnapshot:
        """Get the current snapshot of the document."""
        return MockDocumentSnapshot(self._exists, copy.deepcopy(self._data))

    async def set(self, data: dict[str, Any]) -> None:
        """Set the document data."""
        self._data = copy.deepcopy(data)
        self._exists = True

    async def delete(self) -> None:
        """Delete the document."""
        self._data = {}
        self._exists = False


@dataclass
class MockFieldFilter:
    """Mock representing a Firestore FieldFilter."""

    _field_path: str
    _op: str
    _value: Any


# Inject MockFieldFilter to mimic google.cloud.firestore.FieldFilter
google.cloud.firestore.FieldFilter = MockFieldFilter  # type: ignore


class MockCollectionReference:
    """Mock representing a Firestore CollectionReference."""

    def __init__(self, col_name: str, client: MockFirestoreClient) -> None:
        self.id = col_name
        self.client = client
        self._docs: dict[str, MockDocumentReference] = {}
        self._filters: list[tuple[str, str, Any]] = []

    def document(self, doc_id: str) -> MockDocumentReference:
        """Get or create a DocumentReference by ID."""
        if doc_id not in self._docs:
            self._docs[doc_id] = MockDocumentReference(doc_id, self)
        return self._docs[doc_id]

    def where(self, filter: MockFieldFilter) -> MockCollectionReference:  # noqa: A002
        """Add a query filter."""
        self._filters.append((filter._field_path, filter._op, filter._value))
        return self

    def limit(self, limit: int) -> MockCollectionReference:  # noqa: A002
        """Limit query results."""
        return self

    async def get(self) -> list[MockDocumentSnapshot]:
        """Execute query and return list of matching DocumentSnapshots."""
        results = []
        for doc in self._docs.values():
            if not doc._exists:
                continue

            match = True
            for field, op, val in self._filters:
                # Handle nested lookups (e.g. 'last_good.snapshot_id')
                parts = field.split('.')
                actual_val: Any = doc._data
                for part in parts:
                    if isinstance(actual_val, dict) and part in actual_val:
                        actual_val = actual_val[part]
                    else:
                        actual_val = None
                        break

                if op == '==':
                    if actual_val != val:
                        match = False
                        break

            if match:
                results.append(MockDocumentSnapshot(True, copy.deepcopy(doc._data)))

        self._filters = []  # Clear filters for subsequent queries
        return results


class MockTransaction:
    """Mock representing a Firestore AsyncTransaction with a commit engine."""

    def __init__(self, client: MockFirestoreClient) -> None:
        self.client = client
        self._ops: list[tuple[str, MockDocumentReference, dict[str, Any] | None]] = []

    def set(self, doc_ref: MockDocumentReference, data: dict[str, Any]) -> None:
        """Stage a document write."""
        self._ops.append(('set', doc_ref, copy.deepcopy(data)))

    def delete(self, doc_ref: MockDocumentReference) -> None:
        """Stage a document delete."""
        self._ops.append(('delete', doc_ref, None))

    async def get(self, doc_ref: MockDocumentReference) -> MockDocumentSnapshot:
        """Read a document snapshot within the transaction."""
        return await doc_ref.get(transaction=self)

    async def __aenter__(self) -> MockTransaction:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is None:
            # Commit transactional writes/deletes only upon successful exit!
            for op, doc_ref, data in self._ops:
                if op == 'set':
                    assert data is not None
                    doc_ref._data = data
                    doc_ref._exists = True
                elif op == 'delete':
                    doc_ref._data = {}
                    doc_ref._exists = False


class MockFirestoreClient:
    """Mock representing a top-level Firestore AsyncClient."""

    def __init__(self) -> None:
        self._collections: dict[str, MockCollectionReference] = {}
        self.FieldFilter = MockFieldFilter

    def collection(self, name: str) -> MockCollectionReference:
        """Get or create a CollectionReference."""
        if name not in self._collections:
            self._collections[name] = MockCollectionReference(name, self)
        return self._collections[name]

    def transaction(self) -> MockTransaction:
        """Create a new transactional context."""
        return MockTransaction(self)


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_firestore() -> MockFirestoreClient:
    """Yield a fresh MockFirestoreClient instance."""
    return MockFirestoreClient()


@pytest.fixture
def collection_prefix() -> str:
    """Generate a unique randomized collection prefix for isolation."""
    return f'sessions-{random.randint(1000, 9999)}'


# ---------------------------------------------------------------------------
# Test Helpers (Using 100% Valid Pydantic Schemas)
# ---------------------------------------------------------------------------


def make_latest_record(session_id: str, snapshot_id: str) -> LatestRecord:
    """Helper to construct a valid LatestRecord."""
    now = datetime.now(timezone.utc).isoformat()
    snap = SessionSnapshot(
        snapshot_id=snapshot_id,
        session_id=session_id,
        created_at=now,
        state=SessionState(session_id=session_id, custom={'counter': 42}),
        status=SnapshotStatus.COMPLETED,
    )
    return LatestRecord(
        session_id=session_id,
        last_good=snap,
    )


def make_turn_record(session_id: str, snapshot_id: str, seq: int, parent_id: str | None = None) -> TurnRecord:
    """Helper to construct a valid TurnRecord."""
    return TurnRecord(
        session_id=session_id,
        seq=seq,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        kind=StoreRecordKind.CHECKPOINT,
        state_or_patch={'notes': ['setup']},
        status=SnapshotStatus.COMPLETED,
    )


def make_branch_record(session_id: str, snapshot_id: str, depth: int, parent_id: str | None = None) -> BranchRecord:
    """Helper to construct a valid BranchRecord."""
    return BranchRecord(
        session_id=session_id,
        depth=depth,
        snapshot_id=snapshot_id,
        parent_id=parent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        kind=StoreRecordKind.CHECKPOINT,
        state_or_patch={'notes': ['setup']},
        status=SnapshotStatus.COMPLETED,
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_firestore_latest_state_store(mock_firestore: Any, collection_prefix: str) -> None:
    """Test FirestoreLatestStateStore operations."""
    store = FirestoreLatestStateStore(client=mock_firestore, collection_prefix=collection_prefix)

    session_id = str(uuid4())
    snapshot_id = str(uuid4())
    record = make_latest_record(session_id, snapshot_id)

    # 1. Verify read returns None for missing session
    assert await store.read_record_by_session(session_id) is None
    assert await store.read_record_by_snapshot(snapshot_id) is None

    # 2. Persist the record
    await store.put_record(record)

    # 3. Read it back by session_id and verify fields
    read_session = await store.read_record_by_session(session_id)
    assert read_session is not None
    assert read_session.session_id == session_id
    assert read_session.last_good is not None
    assert read_session.last_good.snapshot_id == snapshot_id
    assert read_session.last_good.state.custom == {'counter': 42}
    assert read_session.last_good.status == SnapshotStatus.COMPLETED

    # 4. Read it back by snapshot_id (last_good slot) and verify
    read_snap = await store.read_record_by_snapshot(snapshot_id)
    assert read_snap is not None
    assert read_snap.session_id == session_id

    # 5. Check pending slot lookup
    pending_snapshot_id = str(uuid4())
    pending_snap = SessionSnapshot(
        snapshot_id=pending_snapshot_id,
        session_id=session_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        state=SessionState(session_id=session_id),
        status=SnapshotStatus.PENDING,
    )
    record.pending = pending_snap
    await store.put_record(record)

    read_pending = await store.read_record_by_snapshot(pending_snapshot_id)
    assert read_pending is not None
    assert read_pending.session_id == session_id

    # 6. Delete the record and verify deletion
    await store.delete_record(session_id)
    assert await store.read_record_by_session(session_id) is None


@pytest.mark.asyncio
async def test_firestore_linear_session_store(mock_firestore: Any, collection_prefix: str) -> None:
    """Test FirestoreLinearSessionStore operations."""
    store = FirestoreLinearSessionStore(client=mock_firestore, collection_prefix=collection_prefix)

    session_id = str(uuid4())
    snap1 = str(uuid4())
    snap2 = str(uuid4())

    record1 = make_turn_record(session_id, snap1, seq=0)
    record2 = make_turn_record(session_id, snap2, seq=1, parent_id=snap1)

    # 1. Verify empty lookups
    assert await store.read_leaf_seq(session_id) is None
    assert await store.read_turn(session_id, 0) is None
    assert await store.read_turn_by_snapshot(snap1) is None

    # 2. Append first turn (seq=0)
    await store.append_turn(session_id, 0, record1)

    # Verify pointer and turn are written
    assert await store.read_leaf_seq(session_id) == 0
    read_turn1 = await store.read_turn(session_id, 0)
    assert read_turn1 is not None
    assert read_turn1.snapshot_id == snap1
    assert read_turn1.parent_id is None

    # Verify query lookup by snapshot
    read_snap1 = await store.read_turn_by_snapshot(snap1)
    assert read_snap1 is not None
    assert read_snap1.snapshot_id == snap1

    # 3. Append second turn (seq=1)
    await store.append_turn(session_id, 1, record2)

    # Verify pointer updated and turn is written
    assert await store.read_leaf_seq(session_id) == 1
    read_turn2 = await store.read_turn(session_id, 1)
    assert read_turn2 is not None
    assert read_turn2.snapshot_id == snap2
    assert read_turn2.parent_id == snap1

    # 4. Update an existing turn record (e.g., status to done)
    record2.status = SnapshotStatus.FAILED
    await store.update_turn(session_id, 1, record2)
    read_updated = await store.read_turn(session_id, 1)
    assert read_updated is not None
    assert read_updated.status == SnapshotStatus.FAILED

    # 5. Truncate (rollback) to seq=0
    await store.truncate_to(session_id, 0)

    # Verify pointer rolled back, and seq=1 is deleted
    assert await store.read_leaf_seq(session_id) == 0
    assert await store.read_turn(session_id, 1) is None
    assert await store.read_turn(session_id, 0) is not None


@pytest.mark.asyncio
async def test_firestore_branching_session_store(mock_firestore: Any, collection_prefix: str) -> None:
    """Test FirestoreBranchingSessionStore operations."""
    store = FirestoreBranchingSessionStore(client=mock_firestore, collection_prefix=collection_prefix)

    session_id = str(uuid4())
    root_snap = str(uuid4())
    branch_a = str(uuid4())
    branch_b = str(uuid4())

    record_root = make_branch_record(session_id, root_snap, depth=0)
    record_a = make_branch_record(session_id, branch_a, depth=1, parent_id=root_snap)
    record_b = make_branch_record(session_id, branch_b, depth=1, parent_id=root_snap)

    # 1. Verify empty lookups
    assert await store.read_leaves(session_id) == []
    assert await store.read_record(root_snap) is None

    # 2. Append root checkpoint
    await store.append_child(session_id, parent_id=None, record=record_root)

    # Verify leaves list and snapshot
    assert await store.read_leaves(session_id) == [root_snap]
    read_root = await store.read_record(root_snap)
    assert read_root is not None
    assert read_root.snapshot_id == root_snap

    # 3. Fork Path A
    await store.append_child(session_id, parent_id=root_snap, record=record_a)

    # Verify leaf replacement: root is replaced by branch_a
    assert await store.read_leaves(session_id) == [branch_a]
    assert (await store.read_record(branch_a)).parent_id == root_snap

    # 4. Fork Path B (from the same root_snap checkpoint, causing a branch)
    await store.append_child(session_id, parent_id=root_snap, record=record_b)

    # Verify leaves list: since root_snap is no longer a leaf, and both branch_a and branch_b are active leaves
    # the leaves list should contain both branch_a and branch_b!
    leaves = await store.read_leaves(session_id)
    assert len(leaves) == 2
    assert branch_a in leaves
    assert branch_b in leaves

    # 5. Update a record in place
    record_b.status = SnapshotStatus.FAILED
    await store.update_record(branch_b, record_b)
    assert (await store.read_record(branch_b)).status == SnapshotStatus.FAILED
