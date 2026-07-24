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

"""In-memory, full-snapshot session store for local development and tests."""

from __future__ import annotations

import asyncio
from typing import Generic

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
    select_leaf,
    session_id_of,
    subscribe,
)
from genkit._core._typing import SessionSnapshot, SnapshotStatus


class InMemorySessionStore(SessionStore[StateT], SnapshotSubscriber, Generic[StateT]):
    """In-memory snapshot store. State is lost when the process exits."""

    def __init__(self, *, reject_ambiguous_session: bool = False) -> None:
        """Create the store.

        When ``reject_ambiguous_session`` is set, a ``session_id`` lookup on a
        history that has forked (more than one leaf) raises instead of picking
        the most recent branch — useful when accidental branching should surface
        loudly.
        """
        self.reject_ambiguous = reject_ambiguous_session
        self.snapshots: dict[str, SessionSnapshot] = {}
        self.subs: Subs = {}

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Return a snapshot by id, or the session's latest leaf, as a deep copy."""
        require_one_selector(snapshot_id=snapshot_id, session_id=session_id)
        async with self.lock:
            if snapshot_id is not None:
                snap = self.snapshots.get(snapshot_id)
                return snap.model_copy(deep=True) if snap is not None else None

            assert session_id is not None
            owned = [snap for snap in self.snapshots.values() if session_id_of(snap) == session_id]
            leaf = select_leaf(snapshots=owned, session_id=session_id, reject_ambiguous=self.reject_ambiguous)
            return leaf.model_copy(deep=True) if leaf is not None else None

    async def save_snapshot(self, snapshot_id: str | None, fn: SaveFn) -> SessionSnapshot | None:
        """Read-modify-write a snapshot in memory and notify status subscribers."""
        async with self.lock:
            existing = self.snapshots.get(snapshot_id) if snapshot_id is not None else None
            next_snapshot = apply_save(existing=existing, snapshot_id=snapshot_id, fn=fn)
            if next_snapshot is None:
                return None
            self.snapshots[next_snapshot.snapshot_id] = next_snapshot.model_copy(deep=True)
            notify(subs=self.subs, snapshot_id=next_snapshot.snapshot_id, status=next_snapshot.status)
            return next_snapshot

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Return a queue that receives this snapshot's status changes."""
        async with self.lock:
            return await subscribe(subs=self.subs, snapshot_id=snapshot_id, current=self.snapshots.get(snapshot_id))
