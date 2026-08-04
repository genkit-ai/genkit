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
import inspect
import json
import logging
from collections.abc import AsyncIterable, Callable
from typing import Any, Generic, TypedDict, cast

from google.cloud import firestore
from google.cloud.firestore import (
    AsyncClient,
    AsyncCollectionReference,
    AsyncDocumentReference,
    AsyncTransaction,
    DocumentSnapshot,
)

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
from genkit._core._typing import JsonPatchOperation, SessionSnapshot, SessionState, SnapshotStatus

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


class SnapshotWriteMeta(TypedDict):
    """Metadata written onto a snapshot doc for later reconstruction."""

    kind: str
    checkpointId: str
    checkpointShardCount: int
    segmentPath: list[str]
    statePatch: list[dict[str, Any]] | None


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
        doc_id = getattr(doc_snapshot, 'id', 'unknown')
        logger.warning("Unknown SnapshotStatus '%s' in Firestore document '%s'", status_val, doc_id)
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

            pointer = pointer_doc.to_dict() or {}
            if pointer.get('isAmbiguous'):
                leaves = pointer.get('leaves')
                leaves_dict = leaves if isinstance(leaves, dict) else {}
                if self.reject_ambiguous:
                    raise GenkitError(
                        status='FAILED_PRECONDITION',
                        message=(
                            f"Session '{session_id}' has branching snapshots, so there is no single latest snapshot. "
                            'This happens when a conversation is branched (e.g. regenerate). '
                            'Resume by snapshot_id instead.'
                        ),
                    )
                if leaves_dict:
                    newest_id = max(
                        leaves_dict.items(),
                        key=lambda kv: (str(kv[1]), str(kv[0])),
                    )[0]
                    reconstructed = await self._reconstruct(transaction, newest_id, context=context)
                    result[0] = self._to_snapshot(reconstructed) if reconstructed else None
                    return

            current_id = pointer.get('currentSnapshotId')
            checkpoint_id = pointer.get('checkpointId')
            shard_count = pointer.get('checkpointShardCount')
            segment_path = pointer.get('segmentPath')
            if (
                isinstance(current_id, str)
                and isinstance(checkpoint_id, str)
                and isinstance(shard_count, int)
                and isinstance(segment_path, list)
            ):
                reconstructed = await self._reconstruct_from(
                    transaction,
                    checkpoint_id=checkpoint_id,
                    shard_count=shard_count,
                    segment_path=[str(x) for x in segment_path],
                    target_id=current_id,
                    context=context,
                )
                if reconstructed is not None:
                    result[0] = self._to_snapshot(reconstructed)
                    return

            if isinstance(current_id, str):
                reconstructed = await self._reconstruct(transaction, current_id, context=context)
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
            pointer = pointer_snap.to_dict() if pointer_snap.exists else None
            new_state = state_to_dict(next_snapshot.state)

            meta: SnapshotWriteMeta
            if existing_recon is not None:
                existing_doc = existing_recon['doc']
                if existing_doc.get('kind') == 'checkpoint':
                    meta = self._write_checkpoint(
                        transaction,
                        sid,
                        new_state,
                        old_shard_count=int(existing_doc.get('checkpointShardCount') or 0),
                        context=context,
                    )
                else:
                    parent_id = existing_doc.get('parentId')
                    parent_state = None
                    if isinstance(parent_id, str):
                        parent_recon = await self._reconstruct(transaction, parent_id, context=context)
                        parent_state = parent_recon['state'] if parent_recon else None
                    candidate_patch = patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    if byte_length(candidate_patch) > self.shard_size:
                        meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                    else:
                        meta = {
                            'kind': 'diff',
                            'checkpointId': str(existing_doc['checkpointId']),
                            'checkpointShardCount': int(existing_doc['checkpointShardCount']),
                            'segmentPath': [str(x) for x in (existing_doc.get('segmentPath') or [])],
                            'statePatch': candidate_patch,
                        }
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
                    or len(parent_meta['segmentPath']) + 1 >= self.checkpoint_interval
                ):
                    meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                else:
                    parent_recon = await self._reconstruct_from(
                        transaction,
                        checkpoint_id=parent_meta['checkpointId'],
                        shard_count=parent_meta['checkpointShardCount'],
                        segment_path=parent_meta['segmentPath'],
                        target_id=next_snapshot.parent_id,
                        context=context,
                    )
                    parent_state = parent_recon['state'] if parent_recon else None
                    candidate_patch = patch_to_json(diff_json(from_value=parent_state, to_value=new_state))
                    if byte_length(candidate_patch) > self.shard_size:
                        meta = self._write_checkpoint(transaction, sid, new_state, context=context)
                    else:
                        meta = {
                            'kind': 'diff',
                            'checkpointId': parent_meta['checkpointId'],
                            'checkpointShardCount': parent_meta['checkpointShardCount'],
                            'segmentPath': [*parent_meta['segmentPath'], sid],
                            'statePatch': candidate_patch,
                        }

            kind = meta['kind']
            checkpoint_id = meta['checkpointId']
            checkpoint_shard_count = meta['checkpointShardCount']
            segment_path = meta['segmentPath']
            state_patch = meta.get('statePatch')

            doc_payload: dict[str, Any] = {
                'snapshotId': sid,
                'sessionId': session_id,
                'createdAt': next_snapshot.created_at,
                'kind': kind,
                'checkpointId': checkpoint_id,
                'checkpointShardCount': checkpoint_shard_count,
                'segmentPath': segment_path,
            }
            if next_snapshot.parent_id is not None:
                doc_payload['parentId'] = next_snapshot.parent_id
            if next_snapshot.updated_at is not None:
                doc_payload['updatedAt'] = next_snapshot.updated_at
            else:
                doc_payload['updatedAt'] = next_snapshot.created_at
            if next_snapshot.status is not None:
                doc_payload['status'] = (
                    next_snapshot.status.value
                    if isinstance(next_snapshot.status, SnapshotStatus)
                    else next_snapshot.status
                )
            if next_snapshot.heartbeat_at is not None:
                doc_payload['heartbeatAt'] = next_snapshot.heartbeat_at
            if next_snapshot.finish_reason is not None:
                doc_payload['finishReason'] = next_snapshot.finish_reason
            if next_snapshot.error is not None:
                doc_payload['error'] = next_snapshot.error.model_dump(by_alias=True, exclude_none=True, mode='json')
            if state_patch is not None:
                doc_payload['statePatch'] = state_patch

            transaction.set(snap_ref, sanitize(doc_payload))
            await self._update_pointer_in_transaction(
                transaction,
                session_id,
                sid,
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
        parent_snapshot_id: str | None,
        created_at: str,
        is_new: bool,
        checkpoint_id: str,
        checkpoint_shard_count: int,
        segment_path: list[str],
        context: dict[str, Any] | None = None,
    ) -> None:
        """Update the session pointer inside an already-open transaction."""
        ref = self.pointer_ref(session_id, context)
        snapshot = await ref.get(transaction=transaction)
        pointer = snapshot.to_dict() if snapshot.exists else None

        leaves: dict[str, str] = {}
        if pointer and isinstance(pointer.get('leaves'), dict):
            leaves = {str(k): str(v) for k, v in pointer['leaves'].items() if isinstance(k, str) and isinstance(v, str)}

        if is_new:
            if parent_snapshot_id and parent_snapshot_id in leaves:
                leaves.pop(parent_snapshot_id, None)
            leaves[snapshot_id] = created_at
        elif snapshot_id in leaves:
            leaves[snapshot_id] = created_at
        else:
            return

        is_ambiguous = len(leaves) > 1
        payload: dict[str, Any] = {
            'isAmbiguous': is_ambiguous,
            'leaves': leaves,
            'updatedAt': firestore.SERVER_TIMESTAMP,
            'checkpointId': checkpoint_id,
            'checkpointShardCount': checkpoint_shard_count,
            'segmentPath': segment_path,
        }
        if not is_ambiguous:
            payload['currentSnapshotId'] = next(iter(leaves.keys()))
        elif pointer and 'currentSnapshotId' in pointer:
            payload['currentSnapshotId'] = firestore.DELETE_FIELD

        if pointer:
            transaction.update(ref, payload)
        else:
            transaction.set(ref, payload)

    async def _load_parent_chain_meta(
        self,
        transaction: AsyncTransaction,
        parent_id: str,
        pointer: dict[str, Any] | None,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Resolve parent checkpoint/segment metadata without materializing state."""
        if pointer and pointer.get('currentSnapshotId') == parent_id:
            checkpoint_id = pointer.get('checkpointId')
            shard_count = pointer.get('checkpointShardCount')
            segment_path = pointer.get('segmentPath')
            if isinstance(checkpoint_id, str) and isinstance(shard_count, int) and isinstance(segment_path, list):
                return {
                    'checkpointId': checkpoint_id,
                    'checkpointShardCount': shard_count,
                    'segmentPath': [str(x) for x in segment_path],
                }
        snap = await self.snapshot_ref(parent_id, context).get(transaction=transaction)
        if not snap.exists:
            logger.warning("Parent snapshot document '%s' does not exist", parent_id)
            return None
        data = snap.to_dict() or {}
        checkpoint_id = data.get('checkpointId')
        shard_count = data.get('checkpointShardCount')
        segment_path = data.get('segmentPath')
        if not isinstance(checkpoint_id, str) or not isinstance(shard_count, int) or not isinstance(segment_path, list):
            logger.warning("Parent snapshot document '%s' contains invalid metadata", parent_id)
            return None
        return {
            'checkpointId': checkpoint_id,
            'checkpointShardCount': shard_count,
            'segmentPath': [str(x) for x in segment_path],
        }

    async def _reconstruct(
        self,
        transaction: AsyncTransaction,
        snapshot_id: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        snap = await self.snapshot_ref(snapshot_id, context).get(transaction=transaction)
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        checkpoint_id = data.get('checkpointId')
        shard_count = data.get('checkpointShardCount')
        segment_path = data.get('segmentPath')
        if not isinstance(checkpoint_id, str) or not isinstance(shard_count, int) or not isinstance(segment_path, list):
            return None
        return await self._reconstruct_from(
            transaction,
            checkpoint_id=checkpoint_id,
            shard_count=shard_count,
            segment_path=[str(x) for x in segment_path],
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
    ) -> dict[str, Any] | None:
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
        res = transaction.get_all(refs)
        gen = await res if inspect.iscoroutine(res) else res
        if isinstance(gen, list):
            snaps = cast(list[DocumentSnapshot], gen)
        else:
            snaps = [s async for s in cast(AsyncIterable[DocumentSnapshot], gen)]
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
            checkpoint_doc = checkpoint_snap.to_dict() or {}
            if checkpoint_doc.get('snapshotId') != target_id:
                logger.warning(
                    "Checkpoint document '%s' snapshotId mismatch (got '%s', expected '%s')",
                    checkpoint_ref.path,
                    checkpoint_doc.get('snapshotId'),
                    target_id,
                )
                return None
            return {'doc': checkpoint_doc, 'state': state if isinstance(state, dict) else {}}

        target_doc: dict[str, Any] | None = None
        for ref in seg_refs:
            seg_snap = by_path.get(ref.path)
            if seg_snap is None or not seg_snap.exists:
                logger.warning(
                    "Segment snapshot document '%s' does not exist for target '%s'",
                    ref.path,
                    target_id,
                )
                return None
            seg_doc = seg_snap.to_dict() or {}
            state = apply_json_patch(doc=state, patch=patch_from_json(seg_doc.get('statePatch')))
            target_doc = seg_doc

        if target_doc is None or target_doc.get('snapshotId') != target_id:
            logger.warning(
                "Target segment snapshot document mismatch or missing for '%s'",
                target_id,
            )
            return None
        return {'doc': target_doc, 'state': state if isinstance(state, dict) else {}}

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
            transaction.set(shards_col.document(f'{checkpoint_id}_{i}'), {'chunk': chunk})
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
        return {
            'kind': 'checkpoint',
            'checkpointId': snapshot_id,
            'checkpointShardCount': shard_count,
            'segmentPath': [],
            'statePatch': None,
        }

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
            chunk = (snap.to_dict() or {}).get('chunk')
            if isinstance(chunk, memoryview):
                buffers.append(chunk.tobytes())
            elif isinstance(chunk, (bytes, bytearray)):
                buffers.append(bytes(chunk))
            elif isinstance(chunk, str):
                buffers.append(chunk.encode('utf-8'))
            else:
                raise GenkitError(
                    status='DATA_LOSS',
                    message=f"FirestoreSessionStore: invalid checkpoint shard '{snap.id}'.",
                )
        return json.loads(b''.join(buffers).decode('utf-8'))

    def _to_snapshot(self, reconstructed: dict[str, Any] | None) -> SessionSnapshot | None:
        if reconstructed is None:
            return None
        doc = reconstructed['doc']
        state = state_from_dict(reconstructed.get('state'))
        status_raw = doc.get('status')
        status = None
        if status_raw is not None:
            try:
                status = SnapshotStatus(status_raw)
            except ValueError:
                logger.warning(
                    "Unknown SnapshotStatus '%s' for snapshot '%s'",
                    status_raw,
                    doc.get('snapshotId'),
                )
                status = None
        return SessionSnapshot(
            snapshot_id=doc['snapshotId'],
            session_id=doc.get('sessionId'),
            parent_id=doc.get('parentId'),
            created_at=doc['createdAt'],
            updated_at=doc.get('updatedAt'),
            heartbeat_at=doc.get('heartbeatAt'),
            status=status,
            finish_reason=doc.get('finishReason'),
            error=doc.get('error'),
            state=state,
        )

    def _ensure_sync_client(self) -> firestore.Client:
        """Return the sync client used for realtime watches.

        ``google-cloud-firestore`` stores credentials / database / client options
        only on private attrs (``_credentials``, ``_database``,
        ``_client_options``) — there is no public getter. We copy those so the
        watch client hits the same project and database as ``self.client``. If a
        library bump renames them, pass an explicit ``sync_client=`` instead.
        """
        if self.sync_client is not None:
            return self.sync_client
        self.sync_client = firestore.Client(
            project=self.client.project,
            credentials=getattr(self.client, '_credentials', None),
            database=getattr(self.client, '_database', None),
            client_options=getattr(self.client, '_client_options', None),
        )
        self._owns_sync_client = True
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
