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

"""Advanced Agent Session Lifecycle Stress Tests for FirestoreSessionStore.

Verifies product correctness for tricky agent lifecycle states:
- Abort / Cancelled turns
- Heartbeat in-place updates (no re-branching)
- Detach / skipped saves (mutator returning None)
- Complex state payload reconstruction fidelity (Message parts, Tool calls, GenkitErrors)
"""

from __future__ import annotations

from typing import Any

import pytest
from genkit_google_cloud.session_store.firestore import FirestoreSessionStore

from genkit._core._typing import (
    GenkitRuntimeError,
    MessageData,
    Part,
    Role,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
)


class HarnessState:
    """In-memory Firestore emulator fake for unit tests."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}


class FakeDocRef:
    def __init__(self, path: str, harness: HarnessState) -> None:
        self.path = path
        self.harness = harness

    @property
    def id(self) -> str:
        return self.path.split('/')[-1]

    def collection(self, col_path: str) -> FakeColRef:
        return FakeColRef(f'{self.path}/{col_path}', self.harness)

    async def get(self, transaction: Any = None) -> FakeDocSnap:
        return FakeDocSnap(self.path, self.harness.docs.get(self.path), self)


class FakeDocSnap:
    def __init__(self, path: str, data: dict[str, Any] | None, reference: FakeDocRef | None = None) -> None:
        self.path = path
        self._data = data
        self.reference = reference or FakeDocRef(path, HarnessState())

    @property
    def exists(self) -> bool:
        return self._data is not None

    @property
    def id(self) -> str:
        return self.path.split('/')[-1]

    def to_dict(self) -> dict[str, Any] | None:
        return self._data.copy() if self._data is not None else None


class FakeColRef:
    def __init__(self, path: str, harness: HarnessState) -> None:
        self.path = path
        self.harness = harness

    def document(self, doc_id: str) -> FakeDocRef:
        return FakeDocRef(f'{self.path}/{doc_id}', self.harness)


class FakeClient:
    def __init__(self, harness: HarnessState) -> None:
        self.harness = harness
        self.project = 'fake-project'

    def collection(self, col_path: str) -> FakeColRef:
        return FakeColRef(col_path, self.harness)

    def transaction(self, read_only: bool = False) -> FakeTransaction:
        return FakeTransaction(self.harness)


class FakeTransaction:
    def __init__(self, harness: HarnessState) -> None:
        self.harness = harness

    async def get(self, ref: FakeDocRef) -> FakeDocSnap:
        return await ref.get()

    def get_all(self, refs: list[FakeDocRef]) -> list[FakeDocSnap]:
        out = []
        for ref in refs:
            out.append(FakeDocSnap(ref.path, self.harness.docs.get(ref.path), ref))
        return out

    def set(self, ref: FakeDocRef, data: dict[str, Any]) -> None:
        self.harness.docs[ref.path] = data

    def update(self, ref: FakeDocRef, data: dict[str, Any]) -> None:
        existing = self.harness.docs.get(ref.path, {})
        existing.update(data)
        self.harness.docs[ref.path] = existing

    def delete(self, ref: FakeDocRef) -> None:
        self.harness.docs.pop(ref.path, None)


def fake_transactional(fn: Any) -> Any:
    async def wrapper(transaction: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(transaction, *args, **kwargs)

    return wrapper


@pytest.fixture(autouse=True)
def patch_firestore_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('google.cloud.firestore.async_transactional', fake_transactional)


class AdvancedHarness:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}

    def store(self, **kwargs: Any) -> FirestoreSessionStore[Any]:
        hs = HarnessState()
        hs.docs = self.docs
        store = FirestoreSessionStore[Any](client=FakeClient(hs), **kwargs)  # type: ignore[arg-type]
        return store


def _snap_path(snapshot_id: str, prefix: str = 'global') -> str:
    return f'genkit-sessions/{prefix}/snapshots/{snapshot_id}'


def _pointer_path(session_id: str, prefix: str = 'global') -> str:
    return f'genkit-sessions-pointers/{prefix}/pointers/{session_id}'


@pytest.mark.asyncio
async def test_abort_turn_lifecycle_correctness() -> None:
    """Verify aborting an active turn updates status to ABORTED, records error/reason, and preserves pointer consistency."""
    h = AdvancedHarness()
    store = h.store()

    # Step 1: Initial turn saved as IN_PROGRESS
    await store.save_snapshot(
        'snap-running-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-running-1',
            session_id='sess-abort-1',
            created_at='2026-08-03T10:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-abort-1', messages=[MessageData(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])]),
        ),
    )

    # Step 2: Abort mutator updates turn status to ABORTED with finish_reason and GenkitRuntimeError
    abort_err = GenkitRuntimeError(status='CANCELLED', message='User explicitly aborted the agent turn.')
    aborted_snap = await store.save_snapshot(
        'snap-running-1',
        lambda existing: SessionSnapshot(
            snapshot_id='snap-running-1',
            session_id='sess-abort-1',
            created_at=existing.created_at if existing else '2026-08-03T10:00:00Z',
            status=SnapshotStatus.ABORTED,
            finish_reason='aborted',
            error=abort_err,
            state=existing.state if existing else None,
        ),
    )

    assert aborted_snap is not None
    assert aborted_snap.status == SnapshotStatus.ABORTED
    assert aborted_snap.finish_reason == 'aborted'
    assert aborted_snap.error is not None
    assert aborted_snap.error.message == 'User explicitly aborted the agent turn.'

    # Step 3: Verify read_snapshot retrieves exact aborted status and error payload
    reloaded = await store.get_snapshot(snapshot_id='snap-running-1')
    assert reloaded is not None
    assert reloaded.status == SnapshotStatus.ABORTED
    assert reloaded.finish_reason == 'aborted'
    assert reloaded.error is not None
    assert reloaded.error.status == 'CANCELLED'

    # Pointer verification: point to snap-running-1 with isAmbiguous=False
    pointer = h.docs[_pointer_path('sess-abort-1')]
    assert pointer['currentSnapshotId'] == 'snap-running-1'
    assert pointer['isAmbiguous'] is False


@pytest.mark.asyncio
async def test_heartbeat_in_place_update_no_rebranching() -> None:
    """Verify heartbeat mutators update timestamp on existing snapshot in-place without creating a new leaf branch."""
    h = AdvancedHarness()
    store = h.store()

    await store.save_snapshot(
        'snap-hb-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-hb-1',
            session_id='sess-hb-1',
            created_at='2026-08-03T10:00:00Z',
            heartbeat_at='2026-08-03T10:00:00Z',
            status=SnapshotStatus.PENDING,
            state=SessionState(session_id='sess-hb-1', custom={'step': 1}),
        ),
    )

    # Issue 3 consecutive heartbeat updates on snap-hb-1
    for i in range(1, 4):
        hb_time = f'2026-08-03T10:00:0{i}Z'
        updated = await store.save_snapshot(
            'snap-hb-1',
            lambda existing: SessionSnapshot(
                snapshot_id='snap-hb-1',
                session_id='sess-hb-1',
                created_at='2026-08-03T10:00:00Z',
                heartbeat_at=hb_time,
                status=SnapshotStatus.PENDING,
                state=existing.state if existing else None,
            ),
        )
        assert updated is not None
        assert updated.heartbeat_at == hb_time

    pointer = h.docs[_pointer_path('sess-hb-1')]
    assert pointer['isAmbiguous'] is False
    assert len(pointer['leaves']) == 1
    assert pointer['leaves']['snap-hb-1'] == '2026-08-03T10:00:00Z'


@pytest.mark.asyncio
async def test_detach_skipped_save_mutator_returns_none() -> None:
    """Verify mutator returning None skips writing completely (detach scenario)."""
    h = AdvancedHarness()
    store = h.store()

    # Attempt to update a non-existent snapshot with a mutator that returns None
    result = await store.save_snapshot(
        'snap-missing',
        lambda existing: None if existing is None else existing,
    )
    assert result is None
    assert _snap_path('snap-missing') not in h.docs
    assert _pointer_path('sess-missing') not in h.docs


@pytest.mark.asyncio
async def test_complex_state_type_reconstruction_fidelity() -> None:
    """Verify deep product correctness for complex state objects (Message parts, nested dicts, float metrics, status)."""
    h = AdvancedHarness()
    store = h.store()

    complex_state = SessionState(
        session_id='sess-complex-1',
        messages=[
            MessageData(
                role=Role.USER,
                content=[Part(root=TextPart(text='Run weather tool'))],
            ),
            MessageData(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='Calling tool weather...'))],
            ),
        ],
        custom={
            'metrics': {'latency_ms': 142.8, 'tokens': 512, 'flags': [True, False]},
            'nested': {'a': {'b': {'c': 'deep_val'}}},
        },
    )

    saved = await store.save_snapshot(
        'snap-complex-1',
        lambda _e: SessionSnapshot(
            snapshot_id='snap-complex-1',
            session_id='sess-complex-1',
            created_at='2026-08-03T12:00:00Z',
            status=SnapshotStatus.COMPLETED,
            state=complex_state,
        ),
    )
    assert saved is not None

    reconstructed = await store.get_snapshot(snapshot_id='snap-complex-1')
    assert reconstructed is not None
    assert reconstructed.state is not None
    assert reconstructed.state.session_id == 'sess-complex-1'
    assert len(reconstructed.state.messages) == 2
    assert reconstructed.state.messages[0].role == Role.USER
    assert reconstructed.state.custom['metrics']['latency_ms'] == 142.8
    assert reconstructed.state.custom['nested']['a']['b']['c'] == 'deep_val'
