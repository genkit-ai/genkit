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

"""Firestore-backed session store implementations for Genkit."""

from __future__ import annotations

import logging

from google.cloud import firestore

from genkit._ai._agents._session_stores._branching import BranchingSessionStore, BranchRecord
from genkit._ai._agents._session_stores._latest_state import LatestRecord, LatestStateStore
from genkit._ai._agents._session_stores._linear import LinearSessionStore, TurnRecord

logger = logging.getLogger(__name__)


def _init_client(
    client: firestore.AsyncClient | None = None,
    project: str | None = None,
    database: str | None = None,
) -> firestore.AsyncClient:
    """Helper to initialize the AsyncClient with mutual exclusivity checks."""
    if client is not None and (project is not None or database is not None):
        raise ValueError("Cannot provide both an explicit 'client' and 'project' or 'database' configurations.")

    if client is not None:
        return client

    return firestore.AsyncClient(project=project, database=database)


class FirestoreLatestStateStore(LatestStateStore):
    """Firestore-backed LatestStateStore.

    Stores only the latest state (and a single pending slot) per session_id.
    Layout:
      - Collection: `{prefix}-latest`
      - Document ID: `session_id`
    """

    def __init__(
        self,
        client: firestore.AsyncClient | None = None,
        project: str | None = None,
        database: str | None = None,
        collection_prefix: str = 'genkit-sessions',
    ) -> None:
        """Initialize the FirestoreLatestStateStore.

        Args:
            client: Optional pre-configured AsyncClient. Mutually exclusive with project/database.
            project: Optional GCP project ID. Mutually exclusive with client.
            database: Optional Firestore database ID. Mutually exclusive with client.
            collection_prefix: The prefix for the Firestore collection name.
        """
        super().__init__()
        self.client = _init_client(client, project, database)
        self.collection_name = f'{collection_prefix}-latest'

    def _doc(self, session_id: str) -> firestore.AsyncDocumentReference:
        return self.client.collection(self.collection_name).document(session_id)

    async def put_record(self, record: LatestRecord) -> None:
        """Put or update a LatestRecord in Firestore."""
        # Pydantic model_dump uses snake_case, but to conform to Genkit wire/DB
        # protocols we can serialize to standard dict.
        data = record.model_dump()
        await self._doc(record.session_id).set(data)

    async def read_record_by_session(self, session_id: str) -> LatestRecord | None:
        """Read a LatestRecord from Firestore by session_id."""
        snap = await self._doc(session_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        return LatestRecord.model_validate(data) if data else None

    async def read_record_by_snapshot(self, snapshot_id: str) -> LatestRecord | None:
        """Read a LatestRecord from Firestore by snapshot_id."""
        # In a latest state store, lookup by snapshot_id is rare and requires a query
        # because the primary key is the session_id.
        col = self.client.collection(self.collection_name)
        # Check active_snapshot_id slot
        query_active = col.where(filter=firestore.FieldFilter('last_good.snapshot_id', '==', snapshot_id)).limit(1)
        snaps = await query_active.get()
        if snaps:
            return LatestRecord.model_validate(snaps[0].to_dict())

        # Check pending_snapshot_id slot
        query_pending = col.where(filter=firestore.FieldFilter('pending.snapshot_id', '==', snapshot_id)).limit(1)
        snaps = await query_pending.get()
        if snaps:
            return LatestRecord.model_validate(snaps[0].to_dict())

        return None

    async def delete_record(self, session_id: str) -> None:
        """Delete a LatestRecord from Firestore by session_id."""
        await self._doc(session_id).delete()


class FirestoreLinearSessionStore(LinearSessionStore):
    """Firestore-backed LinearSessionStore.

    Stores turn history as a chain of incremental diffs.
    Layout:
      - Turns Collection: `{prefix}-linear-turns`
        - Document ID: `{session_id}_{seq}` (for strongly-consistent direct lookups)
      - Pointers Collection: `{prefix}-linear-pointers`
        - Document ID: `session_id`
    """

    def __init__(
        self,
        client: firestore.AsyncClient | None = None,
        project: str | None = None,
        database: str | None = None,
        collection_prefix: str = 'genkit-sessions',
        checkpoint_interval: int = 10,
    ) -> None:
        """Initialize the FirestoreLinearSessionStore.

        Args:
            client: Optional pre-configured AsyncClient. Mutually exclusive with project/database.
            project: Optional GCP project ID. Mutually exclusive with client.
            database: Optional Firestore database ID. Mutually exclusive with client.
            collection_prefix: The prefix for the Firestore collection names.
            checkpoint_interval: The number of turns between full-state checkpoints.
        """
        super().__init__(checkpoint_interval=checkpoint_interval)
        self.client = _init_client(client, project, database)
        self.turns_col = f'{collection_prefix}-linear-turns'
        self.pointers_col = f'{collection_prefix}-linear-pointers'

    def _turn_doc(self, session_id: str, seq: int) -> firestore.AsyncDocumentReference:
        return self.client.collection(self.turns_col).document(f'{session_id}_{seq}')

    def _pointer_doc(self, session_id: str) -> firestore.AsyncDocumentReference:
        return self.client.collection(self.pointers_col).document(session_id)

    async def append_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        """Append a new turn record to Firestore and update the session leaf pointer."""
        # Run in a strongly-consistent async transaction block
        async with self.client.transaction() as transaction:
            # 1. Write the turn record
            turn_ref = self._turn_doc(session_id, seq)
            transaction.set(turn_ref, record.model_dump())

            # 2. Update the pointer to point to the new leaf sequence number
            pointer_ref = self._pointer_doc(session_id)
            transaction.set(pointer_ref, {'leafSeq': seq})

    async def truncate_to(self, session_id: str, seq: int) -> None:
        """Truncate the linear session history in Firestore up to the specified sequence number."""
        # Atomic truncation requires deleting all turns after `seq` and updating the pointer.
        async with self.client.transaction() as transaction:
            # A. Update the pointer
            pointer_ref = self._pointer_doc(session_id)
            transaction.set(pointer_ref, {'leafSeq': seq})

            # B. Delete subsequent turns
            # Since sequence numbers are contiguous, we read the current pointer to know the upper bound.
            pointer_snap = await pointer_ref.get(transaction=transaction)
            if pointer_snap.exists:
                current_leaf = pointer_snap.to_dict().get('leafSeq', seq)
                for s in range(seq + 1, current_leaf + 1):
                    transaction.delete(self._turn_doc(session_id, s))

    async def update_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        """Update an existing turn record in place in Firestore."""
        await self._turn_doc(session_id, seq).set(record.model_dump())

    async def read_leaf_seq(self, session_id: str) -> int | None:
        """Read the active leaf sequence number for a session from Firestore."""
        snap = await self._pointer_doc(session_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        return data.get('leafSeq') if data else None

    async def read_turn(self, session_id: str, seq: int) -> TurnRecord | None:
        """Read a turn record from Firestore by session_id and sequence number."""
        snap = await self._turn_doc(session_id, seq).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        return TurnRecord.model_validate(data) if data else None

    async def read_turn_by_snapshot(self, snapshot_id: str) -> TurnRecord | None:
        """Read a turn record from Firestore by its unique snapshot_id."""
        # Direct lookup by snapshot_id requires a query on the turns collection.
        query = (
            self.client.collection(self.turns_col)
            .where(filter=firestore.FieldFilter('snapshot_id', '==', snapshot_id))
            .limit(1)
        )
        snaps = await query.get()
        if not snaps:
            return None
        data = snaps[0].to_dict()
        return TurnRecord.model_validate(data) if data else None


class FirestoreBranchingSessionStore(BranchingSessionStore):
    """Firestore-backed BranchingSessionStore.

    Stores checkpoint-based branching conversation trees.
    Layout:
      - Snapshots Collection: `{prefix}-branching-snapshots`
        - Document ID: `snapshot_id` (for strongly-consistent direct lookups)
      - Pointers Collection: `{prefix}-branching-pointers`
        - Document ID: `session_id`
    """

    def __init__(
        self,
        client: firestore.AsyncClient | None = None,
        project: str | None = None,
        database: str | None = None,
        collection_prefix: str = 'genkit-sessions',
        checkpoint_interval: int = 10,
    ) -> None:
        """Initialize the FirestoreBranchingSessionStore.

        Args:
            client: Optional pre-configured AsyncClient. Mutually exclusive with project/database.
            project: Optional GCP project ID. Mutually exclusive with client.
            database: Optional Firestore database ID. Mutually exclusive with client.
            collection_prefix: The prefix for the Firestore collection names.
            checkpoint_interval: The number of turns between full-state checkpoints.
        """
        super().__init__(checkpoint_interval=checkpoint_interval)
        self.client = _init_client(client, project, database)
        self.snapshots_col = f'{collection_prefix}-branching-snapshots'
        self.pointers_col = f'{collection_prefix}-branching-pointers'

    def _snapshot_doc(self, snapshot_id: str) -> firestore.AsyncDocumentReference:
        return self.client.collection(self.snapshots_col).document(snapshot_id)

    def _pointer_doc(self, session_id: str) -> firestore.AsyncDocumentReference:
        return self.client.collection(self.pointers_col).document(session_id)

    async def append_child(self, session_id: str, parent_id: str | None, record: BranchRecord) -> None:
        """Append a new child branch snapshot to Firestore and update active session leaves."""
        # Atomic append-child runs inside a transaction to ensure that the new snapshot
        # write and the active leaves list update are executed atomically together.
        async with self.client.transaction() as transaction:
            # 1. Write the new snapshot record
            snap_ref = self._snapshot_doc(record.snapshot_id)
            transaction.set(snap_ref, record.model_dump())

            # 2. Read the current active leaves pointer
            pointer_ref = self._pointer_doc(session_id)
            pointer_snap = await pointer_ref.get(transaction=transaction)

            leaves = []
            if pointer_snap.exists:
                leaves = pointer_snap.to_dict().get('leaves', [])

            # 3. Update the leaves set:
            # - Remove the parent_id from the active leaves list (since it has now been extended)
            # - Append the new snapshot_id as a new active leaf
            if parent_id in leaves:
                leaves.remove(parent_id)
            if record.snapshot_id not in leaves:
                leaves.append(record.snapshot_id)

            transaction.set(pointer_ref, {'leaves': leaves})

    async def update_record(self, snapshot_id: str, record: BranchRecord) -> None:
        """Update an existing branch record in place in Firestore."""
        await self._snapshot_doc(snapshot_id).set(record.model_dump())

    async def read_record(self, snapshot_id: str) -> BranchRecord | None:
        """Read a branch snapshot record from Firestore by snapshot_id."""
        snap = await self._snapshot_doc(snapshot_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict()
        return BranchRecord.model_validate(data) if data else None

    async def read_leaves(self, session_id: str) -> list[str]:
        """Read the list of active leaf snapshot IDs for a session from Firestore."""
        snap = await self._pointer_doc(session_id).get()
        if not snap.exists:
            return []
        data = snap.to_dict()
        return data.get('leaves', []) if data else []
