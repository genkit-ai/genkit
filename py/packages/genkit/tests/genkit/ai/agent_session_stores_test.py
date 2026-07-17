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

from genkit._ai._agents._snapshot import abort_snapshot_in_store
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
    FileSessionStore,
    InMemorySessionStore,
    SessionStore,
)


def make_snapshot(
    session_id: str,
    text: str,
    status: SnapshotStatus = SnapshotStatus.COMPLETED,
    parent_id: str | None = None,
    created_at: str = '2026-06-18T12:00:00Z',
) -> SessionSnapshot:
    return SessionSnapshot(
        snapshot_id=str(uuid4()),
        parent_id=parent_id,
        created_at=created_at,
        status=status,
        state=SessionState(
            session_id=session_id,
            messages=[MessageData(role='user', content=[Part(root=TextPart(text=text))])],
            custom={},
        ),
    )


def first_text(snap: SessionSnapshot) -> str | None:
    """Pull the first message's leading text out of a snapshot for assertions."""
    assert snap.state is not None
    messages = snap.state.messages
    assert messages is not None
    content = messages[0].content
    assert content is not None
    return getattr(content[0].root, 'text', None)


# --- Core lifecycle: save, get-by-id, get-by-session leaf, retain history ---


@pytest.mark.asyncio
async def test_in_memory_store_lifecycle() -> None:
    await run_lifecycle_test(InMemorySessionStore())


@pytest.mark.asyncio
async def test_file_store_lifecycle(tmp_path: Path) -> None:
    await run_lifecycle_test(FileSessionStore(str(tmp_path)))


async def run_lifecycle_test(store: SessionStore) -> None:
    session_id = 'sess-123'

    # A new pending snapshot. The store mints the id and the session leaf
    # resolves to it regardless of status (the flat store keeps every turn).
    pending = make_snapshot(session_id, 'Hello', SnapshotStatus.PENDING)
    saved = await store.save_snapshot(None, lambda _: pending)
    assert saved is not None and saved.snapshot_id
    first_id = saved.snapshot_id

    snap = await store.get_snapshot(snapshot_id=first_id)
    assert snap is not None
    assert snap.session_id == session_id  # top-level id mirrors the state's
    assert first_text(snap) == 'Hello'

    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.snapshot_id == first_id and leaf.status == SnapshotStatus.PENDING

    # Finalize that snapshot in place (pending -> completed).
    done = make_snapshot(session_id, 'Hello Response', SnapshotStatus.COMPLETED)
    await store.save_snapshot(first_id, lambda _: done)
    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.status == SnapshotStatus.COMPLETED
    assert first_text(leaf) == 'Hello Response'

    # A second turn chained off the first. The session leaf advances, but the
    # earlier snapshot is still addressable by id (full history is retained).
    second = make_snapshot(session_id, 'Hello again', parent_id=first_id, created_at='2026-06-18T12:00:01Z')
    saved2 = await store.save_snapshot(None, lambda _: second)
    assert saved2 is not None
    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.snapshot_id == saved2.snapshot_id
    assert await store.get_snapshot(snapshot_id=first_id) is not None


@pytest.mark.asyncio
async def test_get_snapshot_requires_exactly_one_selector() -> None:
    store = InMemorySessionStore()
    with pytest.raises(GenkitError):
        await store.get_snapshot()
    with pytest.raises(GenkitError):
        await store.get_snapshot(snapshot_id='a', session_id='b')


# --- Abort lifecycle ---


@pytest.mark.asyncio
async def test_abort_flips_pending_only() -> None:
    store = InMemorySessionStore()
    session_id = 'sess-abort'

    pending = await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'work', SnapshotStatus.PENDING))
    assert pending is not None

    assert await abort_snapshot_in_store(store=store, snapshot_id=pending.snapshot_id) == SnapshotStatus.ABORTED
    snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap is not None and snap.status == SnapshotStatus.ABORTED

    # A terminal snapshot is never rewritten by a late abort.
    done = await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'done', SnapshotStatus.COMPLETED))
    assert done is not None
    assert await abort_snapshot_in_store(store=store, snapshot_id=done.snapshot_id) == SnapshotStatus.COMPLETED

    assert await abort_snapshot_in_store(store=store, snapshot_id='does-not-exist') is None


@pytest.mark.asyncio
async def test_status_subscription_observes_abort() -> None:
    store = InMemorySessionStore()
    pending = await store.save_snapshot(None, lambda _: make_snapshot('sess-sub', 'work', SnapshotStatus.PENDING))
    assert pending is not None

    queue = await store.on_snapshot_status_change(pending.snapshot_id)
    assert await queue.get() == SnapshotStatus.PENDING  # current status on subscribe

    await abort_snapshot_in_store(store=store, snapshot_id=pending.snapshot_id)
    assert await queue.get() == SnapshotStatus.ABORTED


# --- Branching leaf resolution ---


@pytest.mark.asyncio
async def test_branched_session_newest_leaf_wins_by_default() -> None:
    store = InMemorySessionStore()
    session_id = 'sess-fork'

    root = await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'root'))
    assert root is not None

    older = make_snapshot(session_id, 'branch A', parent_id=root.snapshot_id, created_at='2026-06-18T12:00:01Z')
    newer = make_snapshot(session_id, 'branch B', parent_id=root.snapshot_id, created_at='2026-06-18T12:00:02Z')
    await store.save_snapshot(None, lambda _: older)
    saved_newer = await store.save_snapshot(None, lambda _: newer)
    assert saved_newer is not None

    # Two sibling leaves: the most recently created one wins, so a stale branch
    # (e.g. one left behind by an aborted turn) never shadows the live timeline.
    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.snapshot_id == saved_newer.snapshot_id


@pytest.mark.asyncio
async def test_branched_session_rejected_when_opted_in() -> None:
    store = InMemorySessionStore(reject_ambiguous_session=True)
    session_id = 'sess-fork-strict'

    root = await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'root'))
    assert root is not None
    await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'A', parent_id=root.snapshot_id))
    await store.save_snapshot(None, lambda _: make_snapshot(session_id, 'B', parent_id=root.snapshot_id))

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id=session_id)
    assert 'branching snapshots (2 leaves)' in str(exc_info.value)


# --- File store chain pruning ---


async def save_chained(store: SessionStore, session_id: str, text: str, parent_id: str | None, when: str) -> str:
    """Save one turn chained onto ``parent_id`` and return the minted snapshot id."""
    snap = make_snapshot(session_id, text, parent_id=parent_id, created_at=when)
    saved = await store.save_snapshot(None, lambda _: snap)
    assert saved is not None
    return saved.snapshot_id


@pytest.mark.asyncio
async def test_file_store_prunes_oldest_past_cap(tmp_path: Path) -> None:
    store = FileSessionStore(str(tmp_path), max_persisted_chain_length=3)
    session_id = 'sess-prune'

    ids: list[str] = []
    parent: str | None = None
    for i in range(4):
        parent = await save_chained(store, session_id, f'turn {i}', parent, f'2026-06-18T12:00:0{i}Z')
        ids.append(parent)

    # Cap is 3, so writing the 4th turn drops the oldest snapshot from disk...
    assert await store.get_snapshot(snapshot_id=ids[0]) is None
    for kept in ids[1:]:
        assert await store.get_snapshot(snapshot_id=kept) is not None

    # ...while the chat still resolves and continues from the newest leaf.
    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.snapshot_id == ids[3]

    # A 5th turn rolls the window forward: the walk stops at the already-deleted
    # parent, and the next-oldest turn is trimmed while the newest three remain.
    ids.append(await save_chained(store, session_id, 'turn 4', ids[3], '2026-06-18T12:00:05Z'))
    assert await store.get_snapshot(snapshot_id=ids[1]) is None
    for kept in ids[2:]:
        assert await store.get_snapshot(snapshot_id=kept) is not None


@pytest.mark.asyncio
async def test_file_store_without_cap_retains_full_chain(tmp_path: Path) -> None:
    store = FileSessionStore(str(tmp_path))
    session_id = 'sess-keep'

    ids: list[str] = []
    parent: str | None = None
    for i in range(5):
        parent = await save_chained(store, session_id, f'turn {i}', parent, f'2026-06-18T12:00:0{i}Z')
        ids.append(parent)

    for kept in ids:
        assert await store.get_snapshot(snapshot_id=kept) is not None
