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

from pathlib import Path
from uuid import uuid4

import pytest

from genkit._ai._agents._session_stores._branching import BranchingSessionStore
from genkit._ai._agents._session_stores._latest_state import LatestStateStore
from genkit._ai._agents._session_stores._linear import LinearSessionStore
from genkit._core._error import GenkitError
from genkit._core._typing import (
    MessageData,
    Part,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
)
from genkit.agent import (
    FileBranchingSessionStore,
    FileLatestStateStore,
    FileLinearSessionStore,
    InMemoryBranchingSessionStore,
    InMemoryLatestStateStore,
    InMemoryLinearSessionStore,
)


def make_snapshot(
    session_id: str, text: str, status: SnapshotStatus = SnapshotStatus.COMPLETED, parent_id: str | None = None
) -> SessionSnapshot:
    return SessionSnapshot(
        snapshot_id=str(uuid4()),
        parent_id=parent_id,
        created_at='2026-06-18T12:00:00Z',
        status=status,
        state=SessionState(
            session_id=session_id,
            messages=[
                MessageData(
                    role='user',
                    content=[Part(root=TextPart(text=text))],
                )
            ],
            custom={},
        ),
    )


@pytest.mark.asyncio
async def test_latest_state_store_in_memory() -> None:
    store = InMemoryLatestStateStore()
    await run_latest_state_store_test(store)


@pytest.mark.asyncio
async def test_latest_state_store_file(tmp_path: Path) -> None:
    store = FileLatestStateStore(str(tmp_path))
    await run_latest_state_store_test(store)


async def run_latest_state_store_test(store: LatestStateStore) -> None:
    session_id = 'sess-123'

    # 1. Save a new PENDING snapshot
    pending = make_snapshot(session_id, 'Hello', SnapshotStatus.PENDING)

    def _save_pending(_: object) -> SessionSnapshot:
        return pending

    saved_pending = await store.save_snapshot(None, _save_pending)
    assert saved_pending is not None
    assert saved_pending.snapshot_id != ''
    pending.snapshot_id = saved_pending.snapshot_id

    # 2. Get pending snapshot by snapshotId
    snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap is not None
    assert snap.state is not None
    assert snap.state.messages is not None
    assert snap.state.messages[0].content[0].root.text == 'Hello'

    # SessionId lookup should return lastGood (not pending yet, so returns None)
    snap_good = await store.get_snapshot(session_id=session_id)
    assert snap_good is None

    # 3. Finalize pending snapshot (promote to lastGood)
    good = make_snapshot(session_id, 'Hello Response', SnapshotStatus.COMPLETED)
    good.snapshot_id = pending.snapshot_id

    def _finalize(_: object) -> SessionSnapshot:
        return good

    await store.save_snapshot(pending.snapshot_id, _finalize)

    # 4. Get good snapshot by sessionId
    snap_good = await store.get_snapshot(session_id=session_id)
    assert snap_good is not None
    assert snap_good.status == SnapshotStatus.COMPLETED
    assert snap_good.state is not None
    assert snap_good.state.messages is not None
    assert snap_good.state.messages[0].content[0].root.text == 'Hello Response'

    # 5. Get pending by snapshotId now returns good (since promoted)
    snap_promoted = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap_promoted is not None
    assert snap_promoted.status == SnapshotStatus.COMPLETED

    # 6. Save a new turn
    new_good = make_snapshot(session_id, 'Hello again', SnapshotStatus.COMPLETED)

    def _save_new(_: object) -> SessionSnapshot:
        return new_good

    saved_new = await store.save_snapshot(None, _save_new)
    assert saved_new is not None

    # sessionId lookup returns the new latest state
    snap_latest = await store.get_snapshot(session_id=session_id)
    assert snap_latest is not None
    assert snap_latest.snapshot_id == saved_new.snapshot_id

    # Old snapshotId lookup now returns None (retains only latest slots)
    old_snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert old_snap is None


@pytest.mark.asyncio
async def test_linear_session_store_in_memory() -> None:
    store = InMemoryLinearSessionStore(checkpoint_interval=2)
    await run_linear_session_store_test(store)


@pytest.mark.asyncio
async def test_linear_session_store_file(tmp_path: Path) -> None:
    store = FileLinearSessionStore(str(tmp_path), checkpoint_interval=2)
    await run_linear_session_store_test(store)


async def run_linear_session_store_test(store: LinearSessionStore) -> None:
    session_id = 'sess-456'

    # Turn 0: Checkpoint (full state)
    t0 = make_snapshot(session_id, 'Turn 0')
    t0_saved = await store.save_snapshot(None, lambda _: t0)
    assert t0_saved is not None
    t0_id = t0_saved.snapshot_id

    # Turn 1: Diff
    t1 = make_snapshot(session_id, 'Turn 1', parent_id=t0_id)
    assert t0.state is not None
    assert t1.state is not None
    # Append message to Turn 0 messages
    t1.state.messages = (t0.state.messages or []) + (t1.state.messages or [])
    t1_saved = await store.save_snapshot(None, lambda _: t1)
    assert t1_saved is not None
    t1_id = t1_saved.snapshot_id

    # Turn 2: Checkpoint (since checkpoint_interval is 2)
    t2 = make_snapshot(session_id, 'Turn 2', parent_id=t1_id)
    assert t2.state is not None
    t2.state.messages = (t1.state.messages or []) + (t2.state.messages or [])
    t2_saved = await store.save_snapshot(None, lambda _: t2)
    assert t2_saved is not None
    t2_id = t2_saved.snapshot_id

    # Verify turn kinds
    rec0 = await store._read_turn_by_snapshot(t0_id)
    assert rec0 is not None
    assert rec0.kind == 'checkpoint'

    rec1 = await store._read_turn_by_snapshot(t1_id)
    assert rec1 is not None
    assert rec1.kind == 'diff'
    # Should store JSON Patch diff, not full state
    assert isinstance(rec1.state_or_patch, list)

    rec2 = await store._read_turn_by_snapshot(t2_id)
    assert rec2 is not None
    assert rec2.kind == 'checkpoint'

    # Reconstruct Turn 1
    snap1 = await store.get_snapshot(snapshot_id=t1_id)
    assert snap1 is not None
    assert snap1.state is not None
    assert snap1.state.messages is not None
    assert len(snap1.state.messages) == 2
    assert snap1.state.messages[0].content[0].root.text == 'Turn 0'
    assert snap1.state.messages[1].content[0].root.text == 'Turn 1'

    # Reconstruct Turn 2
    snap2 = await store.get_snapshot(snapshot_id=t2_id)
    assert snap2 is not None
    assert snap2.state is not None
    assert snap2.state.messages is not None
    assert len(snap2.state.messages) == 3

    # Reconstruct latest leaf
    leaf_snap = await store.get_snapshot(session_id=session_id)
    assert leaf_snap is not None
    assert leaf_snap.snapshot_id == t2_id

    # Truncate / Rollback to Turn 0
    t1_alt = make_snapshot(session_id, 'Turn 1 Alt', parent_id=t0_id)
    assert t1_alt.state is not None
    t1_alt.state.messages = (t0.state.messages or []) + (t1_alt.state.messages or [])
    t1_alt_saved = await store.save_snapshot(None, lambda _: t1_alt)
    assert t1_alt_saved is not None
    t1_alt_id = t1_alt_saved.snapshot_id

    # Check that Turn 1 and Turn 2 were deleted from index & storage
    assert await store._read_turn_by_snapshot(t1_id) is None
    assert await store._read_turn_by_snapshot(t2_id) is None

    # Latest leaf is now Turn 1 Alt
    latest = await store.get_snapshot(session_id=session_id)
    assert latest is not None
    assert latest.snapshot_id == t1_alt_id
    assert latest.state is not None
    assert latest.state.messages is not None
    assert len(latest.state.messages) == 2
    assert latest.state.messages[1].content[0].root.text == 'Turn 1 Alt'


@pytest.mark.asyncio
async def test_branching_session_store_in_memory() -> None:
    store = InMemoryBranchingSessionStore(checkpoint_interval=2)
    await run_branching_session_store_test(store)


@pytest.mark.asyncio
async def test_branching_session_store_file(tmp_path: Path) -> None:
    store = FileBranchingSessionStore(str(tmp_path), checkpoint_interval=2)
    await run_branching_session_store_test(store)


async def run_branching_session_store_test(store: BranchingSessionStore) -> None:
    session_id = 'sess-789'

    # Root Checkpoint
    root = make_snapshot(session_id, 'Root')
    root_saved = await store.save_snapshot(None, lambda _: root)
    assert root_saved is not None
    root_id = root_saved.snapshot_id

    # Branch 1 (Minimal)
    b1_t1 = make_snapshot(session_id, 'Minimal Direction', parent_id=root_id)
    assert root.state is not None
    assert b1_t1.state is not None
    b1_t1.state.messages = (root.state.messages or []) + (b1_t1.state.messages or [])
    b1_t1_saved = await store.save_snapshot(None, lambda _: b1_t1)
    assert b1_t1_saved is not None
    b1_t1_id = b1_t1_saved.snapshot_id

    # Branch 2 (Bold)
    b2_t1 = make_snapshot(session_id, 'Bold Direction', parent_id=root_id)
    assert b2_t1.state is not None
    b2_t1.state.messages = (root.state.messages or []) + (b2_t1.state.messages or [])
    b2_t1_saved = await store.save_snapshot(None, lambda _: b2_t1)
    assert b2_t1_saved is not None
    b2_t1_id = b2_t1_saved.snapshot_id

    # Verify both branches exist as sibling leaves
    snap_min = await store.get_snapshot(snapshot_id=b1_t1_id)
    assert snap_min is not None
    assert snap_min.state is not None
    assert snap_min.state.messages is not None
    assert snap_min.state.messages[1].content[0].root.text == 'Minimal Direction'

    snap_bold = await store.get_snapshot(snapshot_id=b2_t1_id)
    assert snap_bold is not None
    assert snap_bold.state is not None
    assert snap_bold.state.messages is not None
    assert snap_bold.state.messages[1].content[0].root.text == 'Bold Direction'

    # Since there are branching leaf snapshots, sessionId lookup should raise FAILED_PRECONDITION
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id=session_id)
    assert 'branching snapshots (2 leaves)' in str(exc_info.value)
