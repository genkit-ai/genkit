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

from __future__ import annotations

from uuid import uuid4

import pytest

from genkit._core._error import GenkitError
from genkit._core._typing import (
    MessageData,
    Part,
    SessionSnapshot,
    SessionState,
    SnapshotEvent,
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
    session_id: str, text: str, status: SnapshotStatus = SnapshotStatus.DONE, parent_id: str | None = None
) -> SessionSnapshot:
    return SessionSnapshot(
        snapshot_id=str(uuid4()),
        parent_id=parent_id,
        created_at='2026-06-18T12:00:00Z',
        event=SnapshotEvent.TURNEND,
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
async def test_latest_state_store_file(tmp_path) -> None:
    store = FileLatestStateStore(str(tmp_path))
    await run_latest_state_store_test(store)


async def run_latest_state_store_test(store) -> None:
    session_id = 'sess-123'

    # 1. Save a new PENDING snapshot
    pending = make_snapshot(session_id, 'Hello', SnapshotStatus.PENDING)

    def _save_pending(_):
        return pending

    saved_pending = await store.save_snapshot(None, _save_pending)
    assert saved_pending.snapshot_id != ''
    pending.snapshot_id = saved_pending.snapshot_id

    # 2. Get pending snapshot by snapshotId
    snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap is not None
    assert snap.status == SnapshotStatus.PENDING
    assert snap.state.messages[0].content[0].root.text == 'Hello'

    # SessionId lookup should return lastGood (not pending yet, so returns None)
    snap_good = await store.get_snapshot(session_id=session_id)
    assert snap_good is None

    # 3. Finalize pending snapshot (promote to lastGood)
    good = make_snapshot(session_id, 'Hello Response', SnapshotStatus.DONE)
    good.snapshot_id = pending.snapshot_id

    def _finalize(_):
        return good

    await store.save_snapshot(pending.snapshot_id, _finalize)

    # 4. Get good snapshot by sessionId
    snap_good = await store.get_snapshot(session_id=session_id)
    assert snap_good is not None
    assert snap_good.status == SnapshotStatus.DONE
    assert snap_good.state.messages[0].content[0].root.text == 'Hello Response'

    # 5. Get pending by snapshotId now returns good (since promoted)
    snap_promoted = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap_promoted is not None
    assert snap_promoted.status == SnapshotStatus.DONE

    # 6. Save a new turn
    new_good = make_snapshot(session_id, 'Hello again', SnapshotStatus.DONE)

    def _save_new(_):
        return new_good

    saved_new = await store.save_snapshot(None, _save_new)

    # sessionId lookup returns the new latest state
    snap_latest = await store.get_snapshot(session_id=session_id)
    assert snap_latest.snapshot_id == saved_new.snapshot_id

    # Old snapshotId lookup now returns None (retains only latest slots)
    old_snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert old_snap is None


@pytest.mark.asyncio
async def test_linear_session_store_in_memory() -> None:
    store = InMemoryLinearSessionStore(checkpoint_interval=2)
    await run_linear_session_store_test(store)


@pytest.mark.asyncio
async def test_linear_session_store_file(tmp_path) -> None:
    store = FileLinearSessionStore(str(tmp_path), checkpoint_interval=2)
    await run_linear_session_store_test(store)


async def run_linear_session_store_test(store) -> None:
    session_id = 'sess-456'

    # Turn 0: Checkpoint (full state)
    t0 = make_snapshot(session_id, 'Turn 0')
    t0_saved = await store.save_snapshot(None, lambda _: t0)
    t0_id = t0_saved.snapshot_id

    # Turn 1: Diff
    t1 = make_snapshot(session_id, 'Turn 1', parent_id=t0_id)
    # Append message to Turn 0 messages
    t1.state.messages = t0.state.messages + t1.state.messages
    t1_saved = await store.save_snapshot(None, lambda _: t1)
    t1_id = t1_saved.snapshot_id

    # Turn 2: Checkpoint (since checkpoint_interval is 2)
    t2 = make_snapshot(session_id, 'Turn 2', parent_id=t1_id)
    t2.state.messages = t1.state.messages + t2.state.messages
    t2_saved = await store.save_snapshot(None, lambda _: t2)
    t2_id = t2_saved.snapshot_id

    # Verify turn kinds
    rec0 = await store._read_turn_by_snapshot(t0_id)
    assert rec0.kind == 'checkpoint'

    rec1 = await store._read_turn_by_snapshot(t1_id)
    assert rec1.kind == 'diff'
    # Should store JSON Patch diff, not full state
    assert isinstance(rec1.state_or_patch, list)

    rec2 = await store._read_turn_by_snapshot(t2_id)
    assert rec2.kind == 'checkpoint'

    # Reconstruct Turn 1
    snap1 = await store.get_snapshot(snapshot_id=t1_id)
    assert snap1 is not None
    assert len(snap1.state.messages) == 2
    assert snap1.state.messages[0].content[0].root.text == 'Turn 0'
    assert snap1.state.messages[1].content[0].root.text == 'Turn 1'

    # Reconstruct Turn 2
    snap2 = await store.get_snapshot(snapshot_id=t2_id)
    assert len(snap2.state.messages) == 3

    # Reconstruct latest leaf
    leaf_snap = await store.get_snapshot(session_id=session_id)
    assert leaf_snap.snapshot_id == t2_id

    # Truncate / Rollback to Turn 0
    t1_alt = make_snapshot(session_id, 'Turn 1 Alt', parent_id=t0_id)
    t1_alt.state.messages = t0.state.messages + t1_alt.state.messages
    t1_alt_saved = await store.save_snapshot(None, lambda _: t1_alt)
    t1_alt_id = t1_alt_saved.snapshot_id

    # Check that Turn 1 and Turn 2 were deleted from index & storage
    assert await store._read_turn_by_snapshot(t1_id) is None
    assert await store._read_turn_by_snapshot(t2_id) is None

    # Latest leaf is now Turn 1 Alt
    latest = await store.get_snapshot(session_id=session_id)
    assert latest.snapshot_id == t1_alt_id
    assert len(latest.state.messages) == 2
    assert latest.state.messages[1].content[0].root.text == 'Turn 1 Alt'


@pytest.mark.asyncio
async def test_branching_session_store_in_memory() -> None:
    store = InMemoryBranchingSessionStore(checkpoint_interval=2)
    await run_branching_session_store_test(store)


@pytest.mark.asyncio
async def test_branching_session_store_file(tmp_path) -> None:
    store = FileBranchingSessionStore(str(tmp_path), checkpoint_interval=2)
    await run_branching_session_store_test(store)


async def run_branching_session_store_test(store) -> None:
    session_id = 'sess-789'

    # Root Checkpoint
    root = make_snapshot(session_id, 'Root')
    root_saved = await store.save_snapshot(None, lambda _: root)
    root_id = root_saved.snapshot_id

    # Branch 1 (Minimal)
    b1_t1 = make_snapshot(session_id, 'Minimal Direction', parent_id=root_id)
    b1_t1.state.messages = root.state.messages + b1_t1.state.messages
    b1_t1_saved = await store.save_snapshot(None, lambda _: b1_t1)
    b1_t1_id = b1_t1_saved.snapshot_id

    # Branch 2 (Bold)
    b2_t1 = make_snapshot(session_id, 'Bold Direction', parent_id=root_id)
    b2_t1.state.messages = root.state.messages + b2_t1.state.messages
    b2_t1_saved = await store.save_snapshot(None, lambda _: b2_t1)
    b2_t1_id = b2_t1_saved.snapshot_id

    # Verify both branches exist as sibling leaves
    snap_min = await store.get_snapshot(snapshot_id=b1_t1_id)
    assert snap_min.state.messages[1].content[0].root.text == 'Minimal Direction'

    snap_bold = await store.get_snapshot(snapshot_id=b2_t1_id)
    assert snap_bold.state.messages[1].content[0].root.text == 'Bold Direction'

    # Since there are branching leaf snapshots, sessionId lookup should raise FAILED_PRECONDITION
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id=session_id)
    assert 'branching snapshots (2 leaves)' in str(exc_info.value)
