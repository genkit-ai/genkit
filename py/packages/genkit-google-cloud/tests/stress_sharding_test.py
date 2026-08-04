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

"""Stress test suite for FirestoreSessionStore sharding, payload limits, and deep history."""

from __future__ import annotations

import json
import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_google_cloud import FirestoreSessionStore
from google.cloud import firestore

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


def _make_state_with_size(session_id: str, target_bytes: int, tag: str = '') -> SessionState:
    """Generate a SessionState whose JSON payload is approximately target_bytes."""
    overhead = 60 + len(session_id) + len(tag)
    payload_len = max(0, int(target_bytes) - overhead)
    blob = 'x' * payload_len
    return SessionState(session_id=session_id, custom={'tag': tag, 'blob': blob})


@pytest.mark.asyncio
async def test_large_payload_sharding_1kb_to_5mb() -> None:
    """Test snapshots with state sizes ranging from 1 KB up to 5 MB requiring 10+ shards."""
    h = FakeStoreHarness()
    shard_size = 512 * 1024  # 512 KB default
    store = h.store(shard_size=shard_size)

    test_sizes = [
        1 * 1024,  # 1 KB -> 1 shard
        100 * 1024,  # 100 KB -> 1 shard
        512 * 1024,  # 512 KB -> 1 or 2 shards
        1 * 1024 * 1024,  # 1 MB -> 2-3 shards
        int(2.5 * 1024 * 1024),  # 2.5 MB -> 5-6 shards
        5 * 1024 * 1024,  # 5 MB -> 10+ shards
    ]

    for idx, target_size in enumerate(test_sizes):
        snap_id = f'snap-large-{idx}'
        sess_id = f'sess-large-{idx}'
        original_state = _make_state_with_size(sess_id, target_size, tag=f'size-{target_size}')

        saved = await store.save_snapshot(
            snap_id,
            lambda _e, s=original_state, sid=snap_id, sess=sess_id: SessionSnapshot(
                snapshot_id=sid,
                session_id=sess,
                created_at='2026-08-03T00:00:00Z',
                status=SnapshotStatus.COMPLETED,
                state=s,
            ),
        )
        assert saved is not None
        assert saved.snapshot_id == snap_id

        snap_doc = h.docs[_snap_path(snap_id)]
        assert snap_doc['kind'] == 'checkpoint'
        actual_shard_count = snap_doc['checkpointShardCount']

        # Calculate serialized byte length
        serialized = json.dumps(
            original_state.model_dump(by_alias=True, exclude_none=True, mode='json'),
            separators=(',', ':'),
            default=str,
        ).encode('utf-8')
        expected_shards = math.ceil(len(serialized) / shard_size)
        assert actual_shard_count == expected_shards

        if target_size >= 5 * 1024 * 1024:
            assert actual_shard_count >= 10

        # Verify all shard docs exist in storage
        for s_idx in range(actual_shard_count):
            assert _shard_path(snap_id, s_idx) in h.docs

        # Reconstruct by snapshot_id
        loaded_by_id = await store.get_snapshot(snapshot_id=snap_id)
        assert loaded_by_id is not None
        assert loaded_by_id.state is not None
        assert loaded_by_id.state.custom == original_state.custom

        # Reconstruct by session_id
        loaded_by_sess = await store.get_snapshot(session_id=sess_id)
        assert loaded_by_sess is not None
        assert loaded_by_sess.state is not None
        assert loaded_by_sess.state.custom == original_state.custom


@pytest.mark.asyncio
async def test_100_turn_continuous_session_history() -> None:
    """Test a 100-turn continuous session history crossing multiple checkpoint_interval boundaries."""
    h = FakeStoreHarness()
    checkpoint_interval = 25
    store = h.store(checkpoint_interval=checkpoint_interval)

    sess_id = 'sess-100-turns'
    states: dict[int, SessionState] = {}
    checkpoint_ids: set[str] = set()
    diff_ids: set[str] = set()

    for turn in range(100):
        snap_id = f'snap-turn-{turn}'
        parent_id = f'snap-turn-{turn - 1}' if turn > 0 else None
        state = SessionState(
            session_id=sess_id,
            custom={
                'turn': turn,
                'history': [f'turn_{t}_msg' for t in range(turn + 1)],
                'meta': {'turn_str': str(turn)},
            },
        )
        states[turn] = state

        saved = await store.save_snapshot(
            snap_id,
            lambda _e, sid=snap_id, pid=parent_id, st=state, t_idx=turn: SessionSnapshot(
                snapshot_id=sid,
                parent_id=pid,
                session_id=sess_id,
                created_at=f'2026-08-03T00:00:{t_idx:02d}Z',
                status=SnapshotStatus.COMPLETED,
                state=st,
            ),
        )
        assert saved is not None

        doc = h.docs[_snap_path(snap_id)]
        if doc['kind'] == 'checkpoint':
            checkpoint_ids.add(snap_id)
        else:
            diff_ids.add(snap_id)

    # Checkpoints must occur at turns 0, 25, 50, 75
    expected_checkpoints = {'snap-turn-0', 'snap-turn-25', 'snap-turn-50', 'snap-turn-75'}
    assert checkpoint_ids == expected_checkpoints
    assert len(diff_ids) == 100 - len(expected_checkpoints)

    # Verify reconstruction accuracy for ALL 100 turns individually
    for turn in range(100):
        snap_id = f'snap-turn-{turn}'
        loaded = await store.get_snapshot(snapshot_id=snap_id)
        assert loaded is not None
        assert loaded.state is not None
        assert loaded.state.custom == states[turn].custom

    # Verify session_id lookup returns latest turn (turn 99)
    latest = await store.get_snapshot(session_id=sess_id)
    assert latest is not None
    assert latest.snapshot_id == 'snap-turn-99'
    assert latest.state is not None
    assert latest.state.custom == states[99].custom


@pytest.mark.asyncio
async def test_state_reconstruction_accuracy_across_diffs_and_shards() -> None:
    """Test state reconstruction accuracy comparing original vs reconstructed
    across diffs and multi-shard checkpoints.
    """
    h = FakeStoreHarness()
    shard_size = 64 * 1024  # 64 KB shard size for faster multi-shard testing
    store = h.store(shard_size=shard_size, checkpoint_interval=5)

    sess_id = 'sess-reconstruct-accuracy'
    ground_truth: list[tuple[str, SessionState]] = []

    # Turn 0: Large initial state requiring ~3 shards
    state_0 = _make_state_with_size(sess_id, 180 * 1024, tag='initial-large')
    await store.save_snapshot(
        'turn-0',
        lambda _e: SessionSnapshot(
            snapshot_id='turn-0',
            session_id=sess_id,
            created_at='2026-08-03T00:00:00Z',
            state=state_0,
        ),
    )
    ground_truth.append(('turn-0', state_0))

    # Turns 1-4: Diffs modifying nested structure
    curr_custom: dict[str, Any] = dict(state_0.custom) if state_0.custom is not None else {}
    for i in range(1, 5):
        snap_id = f'turn-{i}'
        parent_id = f'turn-{i - 1}'
        curr_custom[f'key_{i}'] = f'val_{i}'
        curr_custom['step'] = i
        state_i = SessionState(session_id=sess_id, custom=dict(curr_custom))
        await store.save_snapshot(
            snap_id,
            lambda _e, sid=snap_id, pid=parent_id, st=state_i, idx=i: SessionSnapshot(
                snapshot_id=sid,
                parent_id=pid,
                session_id=sess_id,
                created_at=f'2026-08-03T00:00:0{idx}Z',
                state=st,
            ),
        )
        ground_truth.append((snap_id, state_i))

    # Turn 5: Triggers checkpoint interval promotion, and state is also multi-sharded (~3 shards)
    curr_custom['checkpoint_marker'] = True
    state_5 = SessionState(session_id=sess_id, custom=dict(curr_custom))
    await store.save_snapshot(
        'turn-5',
        lambda _e: SessionSnapshot(
            snapshot_id='turn-5',
            parent_id='turn-4',
            session_id=sess_id,
            created_at='2026-08-03T00:00:05Z',
            state=state_5,
        ),
    )
    ground_truth.append(('turn-5', state_5))
    assert h.docs[_snap_path('turn-5')]['kind'] == 'checkpoint'
    assert h.docs[_snap_path('turn-5')]['checkpointShardCount'] >= 2

    # Turns 6-8: Diffs after the new multi-sharded checkpoint
    for i in range(6, 9):
        snap_id = f'turn-{i}'
        parent_id = f'turn-{i - 1}'
        curr_custom[f'post_ckpt_key_{i}'] = [1, 2, 3, i]
        state_i = SessionState(session_id=sess_id, custom=dict(curr_custom))
        await store.save_snapshot(
            snap_id,
            lambda _e, sid=snap_id, pid=parent_id, st=state_i, idx=i: SessionSnapshot(
                snapshot_id=sid,
                parent_id=pid,
                session_id=sess_id,
                created_at=f'2026-08-03T00:00:0{idx}Z',
                state=st,
            ),
        )
        ground_truth.append((snap_id, state_i))

    # Validate exact state matching for every turn
    for snap_id, expected_state in ground_truth:
        reconstructed = await store.get_snapshot(snapshot_id=snap_id)
        assert reconstructed is not None
        assert reconstructed.state is not None
        assert reconstructed.state.custom == expected_state.custom


@pytest.mark.asyncio
async def test_checkpoint_update_stale_shard_cleanup_shrink() -> None:
    """Test updating a checkpoint with a smaller state (old_shard_count > new_shard_count)
    to verify stale shard cleanup.
    """
    h = FakeStoreHarness()
    shard_size = 128 * 1024  # 128 KB
    store = h.store(shard_size=shard_size)

    snap_id = 'snap-shrink'
    sess_id = 'sess-shrink'

    # Step 1: Save large state requiring ~10 shards (1.2 MB / 128 KB = ~10 shards)
    large_state = _make_state_with_size(sess_id, 1200 * 1024, tag='initial-large')
    saved_large = await store.save_snapshot(
        snap_id,
        lambda _e: SessionSnapshot(
            snapshot_id=snap_id,
            session_id=sess_id,
            created_at='2026-08-03T00:00:00Z',
            state=large_state,
        ),
    )
    assert saved_large is not None

    large_doc = h.docs[_snap_path(snap_id)]
    assert large_doc['kind'] == 'checkpoint'
    old_shard_count = large_doc['checkpointShardCount']
    assert old_shard_count >= 8

    # Verify initial shards exist
    for idx in range(old_shard_count):
        assert _shard_path(snap_id, idx) in h.docs

    # Step 2: Update the SAME snapshot with a tiny 1 KB state payload (1 shard)
    small_state = SessionState(session_id=sess_id, custom={'tag': 'updated-small', 'val': 42})
    saved_small = await store.save_snapshot(
        snap_id,
        lambda existing: existing.model_copy(update={'state': small_state}) if existing else None,
    )
    assert saved_small is not None

    small_doc = h.docs[_snap_path(snap_id)]
    assert small_doc['kind'] == 'checkpoint'
    new_shard_count = small_doc['checkpointShardCount']
    assert new_shard_count == 1
    assert old_shard_count > new_shard_count

    # Verify shard 0 exists and has updated content
    assert _shard_path(snap_id, 0) in h.docs
    shard_0_data = json.loads(h.docs[_shard_path(snap_id, 0)]['chunk'].decode('utf-8'))
    assert shard_0_data['custom']['tag'] == 'updated-small'

    # Verify all old stale shards (1 to old_shard_count - 1) have been deleted
    for idx in range(1, old_shard_count):
        assert _shard_path(snap_id, idx) not in h.docs

    # Verify reconstruction accuracy
    reconstructed_by_id = await store.get_snapshot(snapshot_id=snap_id)
    assert reconstructed_by_id is not None
    assert reconstructed_by_id.state is not None
    assert reconstructed_by_id.state.custom == {'tag': 'updated-small', 'val': 42}

    reconstructed_by_sess = await store.get_snapshot(session_id=sess_id)
    assert reconstructed_by_sess is not None
    assert reconstructed_by_sess.state is not None
    assert reconstructed_by_sess.state.custom == {'tag': 'updated-small', 'val': 42}


@pytest.mark.asyncio
async def test_checkpoint_update_grow_shard_count() -> None:
    """Test updating a checkpoint with a larger state (new_shard_count > old_shard_count)."""
    h = FakeStoreHarness()
    shard_size = 128 * 1024  # 128 KB
    store = h.store(shard_size=shard_size)

    snap_id = 'snap-grow'
    sess_id = 'sess-grow'

    # Step 1: Save small initial state (1 shard)
    small_state = SessionState(session_id=sess_id, custom={'tag': 'small-initial'})
    await store.save_snapshot(
        snap_id,
        lambda _e: SessionSnapshot(
            snapshot_id=snap_id,
            session_id=sess_id,
            created_at='2026-08-03T00:00:00Z',
            state=small_state,
        ),
    )

    small_doc = h.docs[_snap_path(snap_id)]
    assert small_doc['checkpointShardCount'] == 1
    assert _shard_path(snap_id, 0) in h.docs
    assert _shard_path(snap_id, 1) not in h.docs

    # Step 2: Update snapshot with large state requiring 5+ shards
    large_state = _make_state_with_size(sess_id, 800 * 1024, tag='updated-large')
    await store.save_snapshot(
        snap_id,
        lambda existing: existing.model_copy(update={'state': large_state}) if existing else None,
    )

    large_doc = h.docs[_snap_path(snap_id)]
    new_shard_count = large_doc['checkpointShardCount']
    assert new_shard_count >= 5

    # Verify all new shard documents exist
    for idx in range(new_shard_count):
        assert _shard_path(snap_id, idx) in h.docs

    # Verify reconstruction accuracy
    reconstructed = await store.get_snapshot(snapshot_id=snap_id)
    assert reconstructed is not None
    assert reconstructed.state is not None
    assert reconstructed.state.custom == large_state.custom
