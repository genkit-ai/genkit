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

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from genkit._ai._agents._session_stores._util import apply_save
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
    SnapshotStatusStream,
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

    # A new pending snapshot under a reserved id; the session leaf resolves
    # to it regardless of status (the flat store keeps every turn).
    pending = make_snapshot(session_id, 'Hello', SnapshotStatus.PENDING)
    saved = await store.save_snapshot(pending.snapshot_id, lambda _: pending)
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
    saved2 = await store.save_snapshot(second.snapshot_id, lambda _: second)
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


@pytest.mark.asyncio
async def test_get_snapshot_rejects_whitespace_only_selector() -> None:
    """Whitespace-only ids are rejected (unusable as document keys)."""
    store = InMemorySessionStore()
    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(snapshot_id='   ')
    assert exc_info.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id='\t')
    assert exc_info.value.status == 'INVALID_ARGUMENT'


# --- Abort lifecycle ---


@pytest.mark.asyncio
async def test_abort_flips_pending_only() -> None:
    store = InMemorySessionStore()
    session_id = 'sess-abort'

    pending_snap = make_snapshot(session_id, 'work', SnapshotStatus.PENDING)
    pending = await store.save_snapshot(pending_snap.snapshot_id, lambda _: pending_snap)
    assert pending is not None

    assert await abort_snapshot_in_store(store=store, snapshot_id=pending.snapshot_id) == SnapshotStatus.ABORTED
    snap = await store.get_snapshot(snapshot_id=pending.snapshot_id)
    assert snap is not None and snap.status == SnapshotStatus.ABORTED

    # A terminal snapshot is never rewritten by a late abort.
    done_snap = make_snapshot(session_id, 'done', SnapshotStatus.COMPLETED)
    done = await store.save_snapshot(done_snap.snapshot_id, lambda _: done_snap)
    assert done is not None
    assert await abort_snapshot_in_store(store=store, snapshot_id=done.snapshot_id) == SnapshotStatus.COMPLETED

    assert await abort_snapshot_in_store(store=store, snapshot_id='does-not-exist') is None


def test_apply_save_rejects_terminal_status_flip() -> None:
    """Direct apply_save: a terminal snapshot cannot change status."""
    existing = make_snapshot('sess-term', 'done', SnapshotStatus.ABORTED)

    def flip(prev: SessionSnapshot | None) -> SessionSnapshot | None:
        assert prev is not None
        return prev.model_copy(update={'status': SnapshotStatus.COMPLETED})

    with pytest.raises(GenkitError) as exc_info:
        apply_save(existing=existing, snapshot_id=existing.snapshot_id, fn=flip)
    assert exc_info.value.status == 'FAILED_PRECONDITION'


def test_apply_save_rejects_session_id_rewrite() -> None:
    """Direct apply_save: a snapshot cannot be rewritten onto another session."""
    existing = make_snapshot('sess-a', 'work', SnapshotStatus.PENDING)
    existing.session_id = 'sess-a'

    def rewrite(prev: SessionSnapshot | None) -> SessionSnapshot | None:
        assert prev is not None
        return prev.model_copy(update={'session_id': 'sess-b'})

    with pytest.raises(GenkitError) as exc_info:
        apply_save(existing=existing, snapshot_id=existing.snapshot_id, fn=rewrite)
    assert exc_info.value.status == 'FAILED_PRECONDITION'


def test_apply_save_rejects_async_mutator() -> None:
    """Direct apply_save: an async mutator is INVALID_ARGUMENT, not awaited."""
    existing = make_snapshot('sess-async', 'work', SnapshotStatus.PENDING)

    async def bad(_prev: SessionSnapshot | None) -> SessionSnapshot | None:
        return existing

    with pytest.raises(GenkitError) as exc_info:
        apply_save(existing=existing, snapshot_id=existing.snapshot_id, fn=bad)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_status_subscription_observes_abort() -> None:
    store = InMemorySessionStore()
    pending_snap = make_snapshot('sess-sub', 'work', SnapshotStatus.PENDING)
    pending = await store.save_snapshot(pending_snap.snapshot_id, lambda _: pending_snap)
    assert pending is not None

    seen: list[SnapshotStatus] = []

    async def consume() -> None:
        statuses = await store.on_snapshot_status_change(pending.snapshot_id)
        async for status in statuses:
            seen.append(status)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the consumer take the seeded PENDING
    await abort_snapshot_in_store(store=store, snapshot_id=pending.snapshot_id)
    await asyncio.wait_for(task, 2)
    assert seen == [SnapshotStatus.PENDING, SnapshotStatus.ABORTED]
    assert pending.snapshot_id not in store.subs


@pytest.mark.asyncio
async def test_status_subscription_waits_for_missing_snapshot() -> None:
    """Subscribing before the snapshot exists stays open until the first save."""
    store = InMemorySessionStore()
    snapshot_id = 'not-yet'
    statuses = await store.on_snapshot_status_change(snapshot_id)
    assert store.subs[snapshot_id][0].empty()  # nothing yet; stream stays open

    pending_snap = make_snapshot('sess-wait', 'hello', SnapshotStatus.PENDING).model_copy(
        update={'snapshot_id': snapshot_id}
    )
    await store.save_snapshot(snapshot_id, lambda _: pending_snap)
    assert await asyncio.wait_for(anext(statuses), 2) == SnapshotStatus.PENDING
    await statuses.aclose()


@pytest.mark.asyncio
async def test_status_subscription_already_terminal_ends() -> None:
    """Subscribe on a finished snapshot yields the status then ends (no hang)."""
    store = InMemorySessionStore()
    done = make_snapshot('sess-done', 'ok', SnapshotStatus.COMPLETED)
    await store.save_snapshot(done.snapshot_id, lambda _: done)

    seen: list[SnapshotStatus] = []
    statuses = await store.on_snapshot_status_change(done.snapshot_id)
    async for status in statuses:
        seen.append(status)
    assert seen == [SnapshotStatus.COMPLETED]
    assert done.snapshot_id not in store.subs


@pytest.mark.asyncio
async def test_status_subscription_aclose_unsubscribes() -> None:
    """Explicit ``aclose`` stops the subscription (JS-style unsubscribe)."""
    store = InMemorySessionStore()
    pending = make_snapshot('sess-aclose', 'work', SnapshotStatus.PENDING)
    await store.save_snapshot(pending.snapshot_id, lambda _: pending)

    statuses = await store.on_snapshot_status_change(pending.snapshot_id)
    assert await anext(statuses) == SnapshotStatus.PENDING
    await statuses.aclose()
    assert pending.snapshot_id not in store.subs


# --- Branching leaf resolution ---


@pytest.mark.asyncio
async def test_branched_session_newest_leaf_wins_by_default() -> None:
    store = InMemorySessionStore()
    session_id = 'sess-fork'

    root_snap = make_snapshot(session_id, 'root')
    root = await store.save_snapshot(root_snap.snapshot_id, lambda _: root_snap)
    assert root is not None

    older = make_snapshot(session_id, 'branch A', parent_id=root.snapshot_id, created_at='2026-06-18T12:00:01Z')
    newer = make_snapshot(session_id, 'branch B', parent_id=root.snapshot_id, created_at='2026-06-18T12:00:02Z')
    await store.save_snapshot(older.snapshot_id, lambda _: older)
    saved_newer = await store.save_snapshot(newer.snapshot_id, lambda _: newer)
    assert saved_newer is not None

    # Two sibling leaves: the most recently created one wins, so a stale branch
    # (e.g. one left behind by an aborted turn) never shadows the live timeline.
    leaf = await store.get_snapshot(session_id=session_id)
    assert leaf is not None and leaf.snapshot_id == saved_newer.snapshot_id


@pytest.mark.asyncio
async def test_branched_session_rejected_when_opted_in() -> None:
    store = InMemorySessionStore(reject_ambiguous_session=True)
    session_id = 'sess-fork-strict'

    root_snap = make_snapshot(session_id, 'root')
    root = await store.save_snapshot(root_snap.snapshot_id, lambda _: root_snap)
    assert root is not None
    branch_a = make_snapshot(session_id, 'A', parent_id=root.snapshot_id)
    branch_b = make_snapshot(session_id, 'B', parent_id=root.snapshot_id)
    await store.save_snapshot(branch_a.snapshot_id, lambda _: branch_a)
    await store.save_snapshot(branch_b.snapshot_id, lambda _: branch_b)

    with pytest.raises(GenkitError) as exc_info:
        await store.get_snapshot(session_id=session_id)
    assert 'branching snapshots (2 leaves)' in str(exc_info.value)


# --- File store chain pruning ---


async def save_chained(store: SessionStore, session_id: str, text: str, parent_id: str | None, when: str) -> str:
    """Save one turn chained onto ``parent_id`` and return the reserved snapshot id."""
    snap = make_snapshot(session_id, text, parent_id=parent_id, created_at=when)
    saved = await store.save_snapshot(snap.snapshot_id, lambda _: snap)
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


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_id', ['../../escape', '../x', 'a/b', r'a\b', '.', '..'])
async def test_file_store_rejects_unsafe_snapshot_ids(tmp_path: Path, bad_id: str) -> None:
    store = FileSessionStore(str(tmp_path))

    with pytest.raises(GenkitError) as exc:
        await store.get_snapshot(snapshot_id=bad_id)
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert 'Invalid snapshotId' in str(exc.value)

    with pytest.raises(GenkitError) as exc:
        await store.save_snapshot(
            bad_id,
            lambda current: make_snapshot('sess', 'x') if current is None else current,
        )
    assert exc.value.status == 'INVALID_ARGUMENT'

    # Path traversal must not create files outside the store directory.
    assert not (tmp_path.parent / 'escape.json').exists()


@pytest.mark.asyncio
async def test_file_store_accepts_plain_basename_snapshot_id(tmp_path: Path) -> None:
    store = FileSessionStore(str(tmp_path))
    snap_id = str(uuid4())
    saved = await store.save_snapshot(snap_id, lambda _: make_snapshot('sess', 'ok'))
    assert saved is not None
    assert saved.snapshot_id == snap_id
    got = await store.get_snapshot(snapshot_id=snap_id)
    assert got is not None
    missing_id = str(uuid4())
    assert await store.get_snapshot(snapshot_id=missing_id) is None  # missing but safe id


@pytest.mark.asyncio
async def test_snapshot_status_stream_is_public_and_runtime_checkable() -> None:
    """Wrapper authors can import and isinstance-check SnapshotStatusStream."""
    store = InMemorySessionStore()
    snap = make_snapshot('sess', 'hi', status=SnapshotStatus.PENDING)
    assert snap.snapshot_id is not None
    await store.save_snapshot(snap.snapshot_id, lambda _: snap)
    stream = await store.on_snapshot_status_change(snap.snapshot_id)
    assert isinstance(stream, SnapshotStatusStream)
