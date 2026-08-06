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

"""Firestore-backed session store for agent snapshots.

Persists each turn as either a JSON Patch diff from its parent or a periodic
full-state checkpoint. Checkpoint state is split across shard documents so no
single write approaches Firestore's ~1 MiB document limit.

Reads and writes use only document-ID lookups (pointer + snapshot + shards), so
deployments need no secondary indexes and stay strongly consistent.

Paths (default collection ``genkit-sessions``, prefix ``global``):

  genkit-sessions/{prefix}/snapshots/{snapshotId}
  genkit-sessions-shards/{prefix}/shards/{checkpointId}_{index}
  genkit-sessions-pointers/{prefix}/pointers/{sessionId}
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import Any, Generic, Literal

from google.cloud import firestore
from google.cloud.firestore import (
    AsyncClient,
    AsyncCollectionReference,
    AsyncDocumentReference,
    AsyncTransaction,
    DocumentSnapshot,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from genkit._ai._agents._session import (
    SessionStore,
    SnapshotSubscriber,
    StateT,
)
from genkit._ai._agents._session_stores._util import (
    SaveFn,
    Subs,
    apply_save,
    notify,
    require_one_selector,
    session_id_of,
    subscribe,
)
from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._core._action import get_current_context
from genkit._core._error import GenkitError
from genkit._core._typing import (
    AgentFinishReason,
    GenkitRuntimeError,
    JsonPatchOperation,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)

DEFAULT_COLLECTION = 'genkit-sessions'
DEFAULT_PREFIX = 'global'
logger = logging.getLogger(__name__)
# Favor common chat workloads: enough diffs between checkpoints to keep write
# amplification down, small enough that reconstruct stays cheap.
DEFAULT_CHECKPOINT_INTERVAL = 25
# Kept well under Firestore's 1 MiB/doc limit so a single shard or diff write
# cannot be rejected for size.
DEFAULT_SHARD_SIZE = 512 * 1024

TERMINAL_STATUSES = frozenset({
    SnapshotStatus.COMPLETED,
    SnapshotStatus.FAILED,
    SnapshotStatus.ABORTED,
    SnapshotStatus.EXPIRED,
})


class ShardDoc(BaseModel):
    """Schema for a checkpoint state shard document stored in Firestore."""

    model_config = ConfigDict(extra='ignore')

    chunk: bytes

    @field_validator('chunk', mode='before')
    @classmethod
    def _coerce_chunk(cls, v: object) -> object:
        """Coerce raw Firestore binary chunk representations to bytes.

        google-cloud-firestore's gRPC deserializer returns zero-copy memoryview
        objects for binary blob fields in DocumentSnapshot.to_dict().
        """
        if isinstance(v, memoryview):
            return v.tobytes()
        if isinstance(v, (bytes, bytearray)):
            return bytes(v)
        if isinstance(v, str):
            return v.encode('utf-8')
        return v


class SnapshotWriteMeta(BaseModel):
    """Metadata written onto a snapshot doc for later reconstruction."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    kind: Literal['diff', 'checkpoint']
    checkpoint_id: str
    checkpoint_shard_count: int
    segment_path: list[str] = Field(default_factory=list)
    state_patch: list[dict[str, Any]] | None = None


class ParentChainMeta(BaseModel):
    """Parent snapshot metadata required for diff calculation."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    checkpoint_id: str
    checkpoint_shard_count: int
    segment_path: list[str] = Field(default_factory=list)


class SnapshotDoc(BaseModel):
    """Schema for turn snapshot document stored in Firestore."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    snapshot_id: str
    session_id: str
    parent_id: str | None = None
    created_at: str
    updated_at: str | None = None
    status: SnapshotStatus | None = None
    heartbeat_at: str | None = None
    finish_reason: AgentFinishReason | None = None
    error: GenkitRuntimeError | None = None
    kind: Literal['diff', 'checkpoint']
    checkpoint_id: str
    checkpoint_shard_count: int
    segment_path: list[str] = Field(default_factory=list)
    state_patch: list[dict[str, Any]] | None = None

    def to_session_snapshot(self, state_raw: dict[str, Any] | SessionState | None = None) -> SessionSnapshot:
        """Convert Firestore snapshot document and reconstructed state to a SessionSnapshot."""
        state = state_from_dict(state_raw)
        return SessionSnapshot(
            snapshot_id=self.snapshot_id,
            session_id=self.session_id,
            parent_id=self.parent_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            heartbeat_at=self.heartbeat_at,
            status=self.status,
            finish_reason=self.finish_reason,
            error=self.error,
            state=state,
        )


class PointerDoc(BaseModel):
    """Schema for session pointer document stored in Firestore."""

    model_config = ConfigDict(
        extra='ignore',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    current_snapshot_id: str | None = None
    checkpoint_id: str | None = None
    checkpoint_shard_count: int | None = None
    segment_path: list[str] = Field(default_factory=list)
    is_ambiguous: bool = False
    leaves: dict[str, str] = Field(default_factory=dict)


def status_from_doc(doc_snapshot: DocumentSnapshot) -> SnapshotStatus | None:
    """Extract and validate the snapshot status from a Firestore document."""
    if not doc_snapshot.exists:
        return None
    status_val = (doc_snapshot.to_dict() or {}).get('status')
    if status_val is None:
        return None
    try:
        out: Any = SnapshotStatus(status_val)
    except ValueError:
        logger.warning("Unknown SnapshotStatus '%s' in Firestore document '%s'", status_val, doc_snapshot.id)
        return None
    return out


def sanitize(value: Any) -> Any:  # noqa: ANN401
    """Drop values Firestore rejects while keeping JSON round-trip semantics."""
    return json.loads(json.dumps(value, default=str))


def state_to_dict(state: SessionState | None) -> dict[str, Any]:
    """Convert a SessionState object to a dictionary representation."""
    if state is None:
        return {}
    dumped = state.model_dump(by_alias=True, exclude_none=True, mode='json')
    return dumped if isinstance(dumped, dict) else {}


def state_from_dict(data: dict[str, Any] | SessionState | None) -> SessionState | None:
    """Parse a dictionary or SessionState object into a SessionState instance."""
    if data is None:
        return None
    if isinstance(data, SessionState):
        return data
    return SessionState.model_validate(data)


def patch_to_json(patch: list[JsonPatchOperation]) -> list[dict[str, Any]]:
    """Convert a list of JsonPatchOperation objects to a list of dicts."""
    return [op.model_dump(by_alias=True, exclude_none=True, mode='json') for op in patch]


def patch_from_json(raw: list[dict[str, Any]] | None) -> list[JsonPatchOperation]:
    """Parse a list of raw dict patch operations into JsonPatchOperation objects."""
    if not raw:
        return []
    return [JsonPatchOperation.model_validate(op) for op in raw]


def byte_length(value: Any) -> int:  # noqa: ANN401
    """Calculate the UTF-8 byte length of a JSON-serialized value."""
    return len(json.dumps(value, separators=(',', ':'), default=str).encode('utf-8'))


class FirestoreSessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """Persist agent snapshots in Cloud Firestore as diffs + sharded checkpoints.

    Uses Application Default Credentials (or ``FIRESTORE_EMULATOR_HOST`` for
    the emulator). Session lookup is pointer + document-ID reads only — no
    secondary indexes. Pass ``snapshot_path_prefix`` to isolate tenants from the
    call ``context`` (e.g. authenticated user id) when session ids may collide
    across users.
    """

    def __init__(
        self,
        *,
        client: AsyncClient | None = None,
        sync_client: firestore.Client | None = None,
        collection: str = DEFAULT_COLLECTION,
        snapshot_path_prefix: Callable[[dict[str, Any] | None], str] | None = None,
        reject_ambiguous_session: bool = False,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        shard_size: int = DEFAULT_SHARD_SIZE,
    ) -> None:
        """Initialize the Firestore session store.

        Realtime status watches need a sync ``Client`` (``AsyncClient`` has no
        ``on_snapshot``). Pass ``sync_client`` to own that client yourself; otherwise
        one is created lazily to match ``client``'s project/database and closed by
        :meth:`close`.

        ``snapshot_path_prefix`` receives the per-call ``context`` dict (e.g. auth)
        that ``get_snapshot`` / ``save_snapshot`` thread as a kwarg — not a wrapper
        options object — so a tenant key can come straight from request context.

        When ``reject_ambiguous_session`` is set, a ``session_id`` lookup on a
        forked history raises instead of returning the newest leaf.
        """
        self.client = client or firestore.AsyncClient()
        self.collection = collection
        self.prefix_fn = snapshot_path_prefix or (lambda _context: DEFAULT_PREFIX)
        self.reject_ambiguous = reject_ambiguous_session
        self.checkpoint_interval = checkpoint_interval
        self.shard_size = shard_size
        self.subs: Subs = {}
        self.sync_client = sync_client
        self._owns_sync_client = sync_client is None
        self._watches: dict[str, Any] = {}

    def snapshots_col(self, context: dict[str, Any] | None = None) -> AsyncCollectionReference:
        """Return the Firestore collection reference for snapshots."""
        prefix = self.prefix_fn(context)
        return self.client.collection(self.collection).document(prefix).collection('snapshots')

    def pointers_col(self, context: dict[str, Any] | None = None) -> AsyncCollectionReference:
        """Return the Firestore collection reference for session pointers."""
        prefix = self.prefix_fn(context)
        return self.client.collection(f'{self.collection}-pointers').document(prefix).collection('pointers')

    def shards_col(self, context: dict[str, Any] | None = None) -> AsyncCollectionReference:
        """Return the Firestore collection reference for checkpoint shards."""
        prefix = self.prefix_fn(context)
        return self.client.collection(f'{self.collection}-shards').document(prefix).collection('shards')

    def snapshot_ref(self, snapshot_id: str, context: dict[str, Any] | None = None) -> AsyncDocumentReference:
        """Return the Firestore document reference for a snapshot ID."""
        return self.snapshots_col(context).document(snapshot_id)

    def pointer_ref(self, session_id: str, context: dict[str, Any] | None = None) -> AsyncDocumentReference:
        """Return the Firestore document reference for a session pointer ID."""
        return self.pointers_col(context).document(session_id)

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> SessionSnapshot | None:
        """Retrieve a session snapshot by snapshot ID or session ID."""
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        transaction = self.client.transaction(read_only=True)
        result: list[SessionSnapshot | None] = [None]

        @firestore.async_transactional
        async def read_in_transaction(transaction: AsyncTransaction) -> None:
            if snapshot_id is not None:
                reconstructed = await self._reconstruct(transaction, snapshot_id, context=context)
                result[0] = self._to_snapshot(reconstructed) if reconstructed else None
                return

            assert session_id is not None
            pointer_doc = await self.pointer_ref(session_id, context).get(transaction=transaction)
            if not pointer_doc.exists:
                result[0] = None
                return

            pointer = PointerDoc.model_validate(pointer_doc.to_dict() or {})
            if pointer.is_ambiguous:
                if self.reject_ambiguous:
                    raise GenkitError(
                        status='FAILED_PRECONDITION',
                        message=(
                            f"Session '{session_id}' has branching snapshots, so there is no single latest snapshot. "
                            'This happens when a conversation is branched (e.g. regenerate). '
                            'Resume by snapshot_id instead.'
                        ),
                    )
                if pointer.leaves:
                    newest_id = max(
                        pointer.leaves.items(),
                        key=lambda kv: (str(kv[1]), str(kv[0])),
                    )[0]
                    reconstructed = await self._reconstruct(transaction, newest_id, context=context)
                    result[0] = self._to_snapshot(reconstructed) if reconstructed else None
                    return

            if pointer.current_snapshot_id and pointer.checkpoint_id and pointer.checkpoint_shard_count is not None:
                reconstructed = await self._reconstruct_from(
                    transaction,
                    checkpoint_id=pointer.checkpoint_id,
                    shard_count=pointer.checkpoint_shard_count,
                    segment_path=pointer.segment_path,
                    target_id=pointer.current_snapshot_id,
                    context=context,
                )
                if reconstructed is not None:
                    result[0] = self._to_snapshot(reconstructed)
                    return

            if pointer.current_snapshot_id:
                reconstructed = await self._reconstruct(transaction, pointer.current_snapshot_id, context=context)
                result[0] = self._to_snapshot(reconstructed) if reconstructed else None

        await read_in_transaction(transaction)
        return result[0]

    async def save_snapshot(
        self,
        snapshot_id: str,
        fn: SaveFn,
        *,
        context: dict[str, Any] | None = None,
    ) -> SessionSnapshot | None:
        """Atomically read-modify-write a snapshot and its session pointer.

        Abort, heartbeat, and finalize all share this path. A process-local lock
        can't coordinate across instances, so the snapshot write and pointer
        update commit together in one Firestore transaction; status subscribers
        are notified only after that commit.
        """
        snap_ref = self.snapshot_ref(snapshot_id, context)
        transaction = self.client.transaction()
        committed: list[SessionSnapshot | None] = [None]

        @firestore.async_transactional
        async def rmw(transaction: AsyncTransaction) -> None:
            existing_recon = await self._reconstruct(transaction, snapshot_id, context=context)
            existing = self._to_snapshot(existing_recon) if existing_recon else None
            next_snapshot = apply_save(existing=existing, snapshot_id=snapshot_id, fn=fn)
            if next_snapshot is None:
                return

            sid = next_snapshot.snapshot_id
            session_id = session_id_of(next_snapshot)
            if not session_id:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message="FirestoreSessionStore requires 'sessionId' on the snapshot.",
                )
            assert sid is not None

            pointer_ref = self.pointer_ref(session_id, context)
            pointer_snap = await pointer_ref.get(transaction=transaction)
            pointer = PointerDoc.model_validate(pointer_snap.to_dict()) if pointer_snap.exists else None
            new_state = state_to_dict(next_snapshot.state)

            meta: SnapshotWriteMeta
            if existing_recon is not None:
                existing_doc, _ = existing_recon
                if existing_doc.kind == 'checkpoint':
                    meta = self._write_checkpoint(
                        transaction,
                        sid,
                        new_state,
                        old_shard_count=existing_doc.checkpoint_shard_count,
                        context=context,
                    )
                else:
                    parent_id = existing_doc.parent_id
                    parent_state = None
                    if parent_id:
                        parent_recon = await self._reconstruct(transaction, parent_id, context=context)
                        parent_state = parent_recon[1] if parent_recon else None
                    candidate_patch = patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    if byte_length(candidate_patch) > self.shard_size:
                        meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                    else:
                        meta = SnapshotWriteMeta(
                            kind='diff',
                            checkpoint_id=existing_doc.checkpoint_id,
                            checkpoint_shard_count=existing_doc.checkpoint_shard_count,
                            segment_path=existing_doc.segment_path,
                            state_patch=candidate_patch,
                        )
            else:
                parent_meta = None
                if next_snapshot.parent_id:
                    parent_meta = await self._load_parent_chain_meta(
                        transaction,
                        next_snapshot.parent_id,
                        pointer,
                        context=context,
                    )
                if (
                    not next_snapshot.parent_id
                    or parent_meta is None
                    or len(parent_meta.segment_path) + 1 >= self.checkpoint_interval
                ):
                    meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                else:
                    parent_recon = await self._reconstruct_from(
                        transaction,
                        checkpoint_id=parent_meta.checkpoint_id,
                        shard_count=parent_meta.checkpoint_shard_count,
                        segment_path=parent_meta.segment_path,
                        target_id=next_snapshot.parent_id,
                        context=context,
                    )
                    parent_state = parent_recon[1] if parent_recon else None
                    candidate_patch = patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    if byte_length(candidate_patch) > self.shard_size:
                        meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                    else:
                        meta = SnapshotWriteMeta(
                            kind='diff',
                            checkpoint_id=parent_meta.checkpoint_id,
                            checkpoint_shard_count=parent_meta.checkpoint_shard_count,
                            segment_path=[*parent_meta.segment_path, sid],
                            state_patch=candidate_patch,
                        )

            kind = meta.kind
            checkpoint_id = meta.checkpoint_id
            checkpoint_shard_count = meta.checkpoint_shard_count
            segment_path = meta.segment_path
            state_patch = meta.state_patch

            doc_model = SnapshotDoc(
                snapshot_id=sid,
                session_id=session_id,
                parent_id=next_snapshot.parent_id,
                created_at=next_snapshot.created_at,
                updated_at=next_snapshot.updated_at or next_snapshot.created_at,
                status=next_snapshot.status,
                heartbeat_at=next_snapshot.heartbeat_at,
                finish_reason=next_snapshot.finish_reason,
                error=next_snapshot.error,
                kind=kind,
                checkpoint_id=checkpoint_id,
                checkpoint_shard_count=checkpoint_shard_count,
                segment_path=segment_path,
                state_patch=state_patch,
            )
            transaction.set(snap_ref, sanitize(doc_model.model_dump(by_alias=True, exclude_none=True, mode='json')))
            await self._update_pointer_in_transaction(
                transaction,
                session_id,
                sid,
                pointer=pointer,
                pointer_exists=pointer_snap.exists,
                parent_snapshot_id=next_snapshot.parent_id,
                created_at=next_snapshot.created_at,
                is_new=existing_recon is None,
                checkpoint_id=checkpoint_id,
                checkpoint_shard_count=checkpoint_shard_count,
                segment_path=segment_path,
                context=context,
            )
            committed[0] = next_snapshot

        await rmw(transaction)
        saved = committed[0]
        if saved is not None and saved.snapshot_id is not None:
            notify(subs=self.subs, snapshot_id=saved.snapshot_id, status=saved.status)
        return saved

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Subscribe to status changes for a session snapshot."""
        context = get_current_context()
        async with self.lock:
            current = await self.read_snapshot(snapshot_id, context=context)
            is_first = snapshot_id not in self.subs
            q = await subscribe(subs=self.subs, snapshot_id=snapshot_id, current=current)
            if current is not None and current.status in TERMINAL_STATUSES:
                await q.put(None)
                self.subs.pop(snapshot_id, None)
                return q
            if is_first and (current is None or current.status not in TERMINAL_STATUSES):
                try:
                    self.start_listener(snapshot_id, context=context)
                except Exception:
                    self.subs.pop(snapshot_id, None)
                    raise
        return q

    async def read_snapshot(
        self,
        snapshot_id: str,
        context: dict[str, Any] | None = None,
    ) -> SessionSnapshot | None:
        """Read and reconstruct a session snapshot from Firestore."""
        transaction = self.client.transaction(read_only=True)
        result: list[SessionSnapshot | None] = [None]

        @firestore.async_transactional
        async def read_in_transaction(transaction: AsyncTransaction) -> None:
            reconstructed = await self._reconstruct(transaction, snapshot_id, context=context)
            result[0] = self._to_snapshot(reconstructed) if reconstructed else None

        await read_in_transaction(transaction)
        return result[0]

    async def _update_pointer_in_transaction(
        self,
        transaction: AsyncTransaction,
        session_id: str,
        snapshot_id: str,
        *,
        pointer: PointerDoc | None,
        pointer_exists: bool,
        parent_snapshot_id: str | None,
        created_at: str,
        is_new: bool,
        checkpoint_id: str,
        checkpoint_shard_count: int,
        segment_path: list[str],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update the session pointer inside an already-open transaction.

        Callers must pass the pointer already loaded earlier in the same
        transaction — Firestore rejects reads after any buffered writes.
        """
        ref = self.pointer_ref(session_id, context)
        leaves: dict[str, str] = dict(pointer.leaves) if pointer else {}

        if is_new:
            if parent_snapshot_id and parent_snapshot_id in leaves:
                leaves.pop(parent_snapshot_id, None)
            leaves[snapshot_id] = created_at
        elif snapshot_id in leaves:
            leaves[snapshot_id] = created_at
        else:
            return

        is_ambiguous = len(leaves) > 1
        new_pointer = PointerDoc(
            current_snapshot_id=next(iter(leaves.keys())) if not is_ambiguous else None,
            checkpoint_id=checkpoint_id,
            checkpoint_shard_count=checkpoint_shard_count,
            segment_path=segment_path,
            is_ambiguous=is_ambiguous,
            leaves=leaves,
        )
        payload = new_pointer.model_dump(by_alias=True, exclude_none=False, mode='python')
        payload['updatedAt'] = firestore.SERVER_TIMESTAMP
        if is_ambiguous and pointer and pointer.current_snapshot_id:
            payload['currentSnapshotId'] = firestore.DELETE_FIELD

        if pointer_exists:
            transaction.update(ref, payload)
        else:
            transaction.set(ref, payload)

    async def _load_parent_chain_meta(
        self,
        transaction: AsyncTransaction,
        parent_id: str,
        pointer: PointerDoc | None,
        *,
        context: dict[str, Any] | None = None,
    ) -> ParentChainMeta | None:
        """Resolve parent checkpoint/segment metadata without materializing state."""
        if pointer and pointer.current_snapshot_id == parent_id:
            if pointer.checkpoint_id and pointer.checkpoint_shard_count is not None:
                return ParentChainMeta(
                    checkpoint_id=pointer.checkpoint_id,
                    checkpoint_shard_count=pointer.checkpoint_shard_count,
                    segment_path=pointer.segment_path,
                )
        snap = await self.snapshot_ref(parent_id, context).get(transaction=transaction)
        if not snap.exists:
            logger.warning("Parent snapshot document '%s' does not exist", parent_id)
            return None
        try:
            doc = SnapshotDoc.model_validate(snap.to_dict())
        except Exception:
            logger.warning("Parent snapshot document '%s' contains invalid metadata", parent_id)
            return None
        return ParentChainMeta(
            checkpoint_id=doc.checkpoint_id,
            checkpoint_shard_count=doc.checkpoint_shard_count,
            segment_path=doc.segment_path,
        )

    async def _reconstruct(
        self,
        transaction: AsyncTransaction,
        snapshot_id: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> tuple[SnapshotDoc, dict[str, Any]] | None:
        snap = await self.snapshot_ref(snapshot_id, context).get(transaction=transaction)
        if not snap.exists:
            return None
        try:
            doc = SnapshotDoc.model_validate(snap.to_dict())
        except Exception:
            # Don't treat a corrupt doc as missing — save_snapshot would otherwise
            # mint a fresh checkpoint and advance the pointer over the bad leaf.
            logger.warning(
                "Snapshot document '%s' failed validation; treating as unreadable",
                snap.reference.path,
            )
            return None
        return await self._reconstruct_from(
            transaction,
            checkpoint_id=doc.checkpoint_id,
            shard_count=doc.checkpoint_shard_count,
            segment_path=doc.segment_path,
            target_id=snapshot_id,
            context=context,
        )

    async def _reconstruct_from(
        self,
        transaction: AsyncTransaction,
        *,
        checkpoint_id: str,
        shard_count: int,
        segment_path: list[str],
        target_id: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[SnapshotDoc, dict[str, Any]] | None:
        target_is_checkpoint = len(segment_path) == 0
        snapshots_col = self.snapshots_col(context)
        shards_col = self.shards_col(context)
        checkpoint_ref = snapshots_col.document(checkpoint_id)
        shard_refs = [shards_col.document(f'{checkpoint_id}_{i}') for i in range(max(shard_count, 0))]
        seg_refs = [snapshots_col.document(sid) for sid in segment_path]

        refs: list[AsyncDocumentReference] = []
        if target_is_checkpoint:
            refs.append(checkpoint_ref)
        refs.extend(shard_refs)
        refs.extend(seg_refs)

        if not refs:
            return None

        by_path: dict[str, DocumentSnapshot] = {}
        # AsyncTransaction.get_all awaits an async generator (library bug), so
        # batch-read through the client with the open transaction instead.
        snaps = [snap async for snap in self.client.get_all(refs, transaction=transaction)]
        for snap in snaps:
            by_path[snap.reference.path] = snap

        shard_snaps = [by_path[ref.path] for ref in shard_refs]
        state = self._stitch(shard_snaps)

        if target_is_checkpoint:
            checkpoint_snap = by_path.get(checkpoint_ref.path)
            if checkpoint_snap is None or not checkpoint_snap.exists:
                logger.warning(
                    "Checkpoint snapshot document '%s' does not exist for target '%s'",
                    checkpoint_ref.path,
                    target_id,
                )
                return None
            try:
                checkpoint_doc = SnapshotDoc.model_validate(checkpoint_snap.to_dict())
            except Exception:
                return None
            if checkpoint_doc.snapshot_id != target_id:
                logger.warning(
                    "Checkpoint document '%s' snapshotId mismatch (got '%s', expected '%s')",
                    checkpoint_ref.path,
                    checkpoint_doc.snapshot_id,
                    target_id,
                )
                return None
            return checkpoint_doc, state if isinstance(state, dict) else {}

        target_doc: SnapshotDoc | None = None
        for ref in seg_refs:
            seg_snap = by_path.get(ref.path)
            if seg_snap is None or not seg_snap.exists:
                logger.warning(
                    "Segment snapshot document '%s' does not exist for target '%s'",
                    ref.path,
                    target_id,
                )
                return None
            try:
                seg_doc = SnapshotDoc.model_validate(seg_snap.to_dict())
            except Exception:
                return None
            state = apply_json_patch(doc=state, patch=patch_from_json(seg_doc.state_patch))
            target_doc = seg_doc

        if target_doc is None or target_doc.snapshot_id != target_id:
            logger.warning(
                "Target segment snapshot document mismatch or missing for '%s'",
                target_id,
            )
            return None
        return target_doc, state if isinstance(state, dict) else {}

    def _write_shards(
        self,
        transaction: AsyncTransaction,
        checkpoint_id: str,
        state: dict[str, Any],
        *,
        old_shard_count: int = 0,
        context: dict[str, Any] | None = None,
    ) -> int:
        shards_col = self.shards_col(context)
        buf = json.dumps(state if state is not None else None, separators=(',', ':'), default=str).encode('utf-8')
        count = max(1, (len(buf) + self.shard_size - 1) // self.shard_size)
        for i in range(count):
            chunk = buf[i * self.shard_size : (i + 1) * self.shard_size]
            shard_doc = ShardDoc(chunk=chunk)
            transaction.set(shards_col.document(f'{checkpoint_id}_{i}'), shard_doc.model_dump(mode='python'))
        for i in range(count, old_shard_count):
            transaction.delete(shards_col.document(f'{checkpoint_id}_{i}'))
        return count

    def _write_checkpoint(
        self,
        transaction: AsyncTransaction,
        snapshot_id: str,
        state: dict[str, Any],
        *,
        old_shard_count: int = 0,
        context: dict[str, Any] | None = None,
    ) -> SnapshotWriteMeta:
        shard_count = self._write_shards(
            transaction,
            snapshot_id,
            state,
            old_shard_count=old_shard_count,
            context=context,
        )
        return SnapshotWriteMeta(
            kind='checkpoint',
            checkpoint_id=snapshot_id,
            checkpoint_shard_count=shard_count,
            segment_path=[],
            state_patch=None,
        )

    def _stitch(self, shard_snaps: list[DocumentSnapshot]) -> dict[str, Any] | None:
        if not shard_snaps:
            return {}
        buffers: list[bytes] = []
        for snap in shard_snaps:
            if not snap.exists:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: missing checkpoint shard '{snap.id}'.",
                )
            try:
                shard = ShardDoc.model_validate(snap.to_dict())
            except Exception as e:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid checkpoint shard '{snap.id}'.",
                ) from e
            buffers.append(shard.chunk)
        return json.loads(b''.join(buffers).decode('utf-8'))

    def _to_snapshot(self, reconstructed: tuple[SnapshotDoc, dict[str, Any]] | None) -> SessionSnapshot | None:
        if reconstructed is None:
            return None
        doc, state_raw = reconstructed
        return doc.to_session_snapshot(state_raw)

    def _ensure_sync_client(self) -> firestore.Client:
        """Return the sync client used for realtime watches.

        Uses ``_to_sync_copy()`` provided by ``google-cloud-firestore``'s
        ``AsyncClient`` to instantiate a sync client sharing the same project,
        credentials, and database.
        """
        if self.sync_client is None:
            if hasattr(self.client, '_to_sync_copy'):
                self.sync_client = self.client._to_sync_copy()
                self._owns_sync_client = True
            else:
                raise GenkitError(
                    status='FAILED_PRECONDITION',
                    message=(
                        'Realtime status watches require a synchronous Firestore client. '
                        'Unable to derive sync client from client. '
                        "Please pass 'sync_client' to FirestoreSessionStore."
                    ),
                )
        return self.sync_client

    def start_listener(
        self,
        snapshot_id: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Start a Firestore real-time listener for status changes on a snapshot."""
        ref = self.snapshot_ref(snapshot_id, context)
        if isinstance(self.client, firestore.AsyncClient) or not hasattr(ref, 'on_snapshot'):
            sync_client = self._ensure_sync_client()
            prefix = self.prefix_fn(context)
            ref = sync_client.collection(self.collection).document(prefix).collection('snapshots').document(snapshot_id)
        loop = asyncio.get_running_loop()

        def on_snapshot(doc_snapshots: list[DocumentSnapshot], changes: Any, read_time: Any) -> None:  # noqa: ANN401
            if not doc_snapshots:
                return
            doc_snapshot = doc_snapshots[0]
            status = status_from_doc(doc_snapshot)
            if status is None:
                return
            loop.call_soon_threadsafe(lambda: notify(subs=self.subs, snapshot_id=snapshot_id, status=status))
            if status not in TERMINAL_STATUSES:
                return

            loop.call_soon_threadsafe(lambda: notify(subs=self.subs, snapshot_id=snapshot_id, status=None))

            async def cleanup() -> None:
                async with self.lock:
                    self.subs.pop(snapshot_id, None)
                    watch = self._watches.pop(snapshot_id, None)
                if watch is not None:
                    await asyncio.to_thread(watch.unsubscribe)

            asyncio.run_coroutine_threadsafe(cleanup(), loop)

        self._watches[snapshot_id] = ref.on_snapshot(on_snapshot)

    def close(self) -> None:
        """Stop active watches and close a lazily-created sync client.

        Safe to call more than once. Does not close a ``sync_client`` the caller
        passed into the constructor, and never closes ``client``.
        """
        watches = list(self._watches.values())
        self._watches.clear()
        self.subs.clear()
        for watch in watches:
            with contextlib.suppress(Exception):
                watch.unsubscribe()
        if self._owns_sync_client and self.sync_client is not None:
            with contextlib.suppress(Exception):
                self.sync_client.close()
            self.sync_client = None
