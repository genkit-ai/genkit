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

"""BranchingSessionStore implementation and variants."""

from __future__ import annotations

import asyncio
import copy
import json
import os
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._ai._session import SessionStore, SnapshotAborter
from genkit._core._error import GenkitError
from genkit._core._typing import JsonPatchOperation, SessionSnapshot, SessionState, SnapshotEvent, SnapshotStatus

StateT = TypeVar('StateT')


class BranchRecord(BaseModel):
    """Storage model for a single node in a branching conversation tree."""

    snapshot_id: str
    parent_id: str | None = None
    session_id: str
    depth: int
    kind: str  # 'checkpoint' | 'diff'
    state_or_patch: Any  # SessionState dict or list of JsonPatchOperation dicts
    status: SnapshotStatus
    created_at: str
    finish_reason: str | None = None
    error: Any | None = None


class BranchingSessionStore(SessionStore, SnapshotAborter):
    """Abstract SessionStore variant that supports full branching trees of snapshots."""

    def __init__(self, checkpoint_interval: int = 10) -> None:
        self.checkpoint_interval = checkpoint_interval
        self._lock = asyncio.Lock()
        self._subs: dict[str, list[asyncio.Queue[SnapshotStatus | None]]] = {}

    @abstractmethod
    async def _append_child(self, session_id: str, parent_id: str | None, record: BranchRecord) -> None:
        """Atomically append a child record to parent and update active leaves list."""
        ...

    @abstractmethod
    async def _update_record(self, snapshot_id: str, record: BranchRecord) -> None:
        """Update an existing record in place."""
        ...

    @abstractmethod
    async def _read_record(self, snapshot_id: str) -> BranchRecord | None:
        """Read a record by snapshot_id."""
        ...

    @abstractmethod
    async def _read_leaves(self, session_id: str) -> list[str]:
        """Read list of active leaf snapshot IDs for a session."""
        ...

    async def _reconstruct_state(self, record: BranchRecord) -> SessionState:
        path = [record]
        curr = record
        while curr.kind != 'checkpoint' and curr.parent_id:
            parent = await self._read_record(curr.parent_id)
            if parent is None:
                raise ValueError(f'Missing parent record {curr.parent_id}')
            path.append(parent)
            curr = parent

        if curr.kind != 'checkpoint':
            raise ValueError(f'No checkpoint found in history chain for snapshot {record.snapshot_id}')

        state_dict = copy.deepcopy(curr.state_or_patch)

        for i in range(len(path) - 2, -1, -1):
            ops_data = path[i].state_or_patch
            ops = [JsonPatchOperation.model_validate(op) for op in ops_data]
            state_dict = apply_json_patch(state_dict, ops)

        return SessionState.model_validate(state_dict)

    async def _reconstruct_snapshot(self, record: BranchRecord) -> SessionSnapshot:
        state = await self._reconstruct_state(record)
        return SessionSnapshot(
            snapshot_id=record.snapshot_id,
            parent_id=record.parent_id,
            created_at=record.created_at,
            event=SnapshotEvent.TURNEND if record.depth > 0 else SnapshotEvent.RUNSTART,
            state=state,
            status=record.status,
            finish_reason=record.finish_reason,
            error=record.error,
        )

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        if bool(snapshot_id) == bool(session_id):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    "get_snapshot requires exactly one of 'snapshot_id' or "
                    f"'session_id' (got {'snapshot_id' if snapshot_id else 'neither'}"
                    f'{" and session_id" if session_id else ""}).'
                ),
            )

        async with self._lock:
            if snapshot_id is not None:
                record = await self._read_record(snapshot_id)
                if record is None:
                    return None
                return await self._reconstruct_snapshot(record)

            assert session_id is not None
            leaves = await self._read_leaves(session_id)
            if not leaves:
                return None
            if len(leaves) == 1:
                record = await self._read_record(leaves[0])
                if record is None:
                    return None
                return await self._reconstruct_snapshot(record)

            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    f"Session '{session_id}' has branching snapshots ({len(leaves)} "
                    'leaves), so there is no single latest snapshot. Resume by '
                    'snapshotId instead.'
                ),
            )

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[[SessionSnapshot | None], SessionSnapshot | None],
    ) -> SessionSnapshot | None:
        async with self._lock:
            if snapshot_id is not None:
                record = await self._read_record(snapshot_id)
                if record is None:
                    return None

                snap = await self._reconstruct_snapshot(record)
                next_snap = fn(snap)
                if next_snap is None:
                    return None

                next_snap.snapshot_id = snapshot_id

                if record.depth == 0 or record.depth % self.checkpoint_interval == 0:
                    record.state_or_patch = next_snap.state.model_dump(by_alias=True)
                else:
                    parent_rec = await self._read_record(record.parent_id)
                    assert parent_rec is not None
                    parent_state = await self._reconstruct_state(parent_rec)
                    ops = diff_json(parent_state.model_dump(by_alias=True), next_snap.state.model_dump(by_alias=True))
                    record.state_or_patch = [op.model_dump(by_alias=True) for op in ops]

                record.status = next_snap.status or SnapshotStatus.DONE
                record.finish_reason = next_snap.finish_reason
                record.error = next_snap.error

                await self._update_record(snapshot_id, record)
                self._notify_locked(snapshot_id, record.status)
                return next_snap
            else:
                sid = str(uuid4())
                next_snap = fn(None)
                if next_snap is None:
                    return None

                next_snap.snapshot_id = sid
                if not next_snap.created_at:
                    next_snap.created_at = datetime.now(timezone.utc).isoformat()
                if not next_snap.status:
                    next_snap.status = SnapshotStatus.DONE

                session_id = next_snap.state.session_id
                parent_id = next_snap.parent_id

                if not parent_id:
                    depth = 0
                else:
                    parent_rec = await self._read_record(parent_id)
                    if parent_rec is None:
                        raise ValueError(f'Parent snapshot {parent_id} not found')
                    session_id = parent_rec.session_id
                    depth = parent_rec.depth + 1

                if depth == 0 or depth % self.checkpoint_interval == 0:
                    kind = 'checkpoint'
                    state_or_patch = next_snap.state.model_dump(by_alias=True)
                else:
                    assert parent_rec is not None
                    parent_state = await self._reconstruct_state(parent_rec)
                    ops = diff_json(parent_state.model_dump(by_alias=True), next_snap.state.model_dump(by_alias=True))
                    kind = 'diff'
                    state_or_patch = [op.model_dump(by_alias=True) for op in ops]

                record = BranchRecord(
                    snapshot_id=sid,
                    parent_id=parent_id,
                    session_id=session_id,
                    depth=depth,
                    kind=kind,
                    state_or_patch=state_or_patch,
                    status=next_snap.status,
                    created_at=next_snap.created_at,
                    finish_reason=next_snap.finish_reason,
                    error=next_snap.error,
                )

                await self._append_child(session_id, parent_id, record)
                self._notify_locked(sid, next_snap.status)
                return next_snap

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        async with self._lock:
            record = await self._read_record(snapshot_id)
            if record is None:
                return None
            if record.status == SnapshotStatus.PENDING:
                record.status = SnapshotStatus.ABORTED
                await self._update_record(snapshot_id, record)
                self._notify_locked(snapshot_id, record.status)
            return record.status

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
        async with self._lock:
            record = await self._read_record(snapshot_id)
            if record is None:
                await q.put(None)
                return q
            await q.put(record.status)
            self._subs.setdefault(snapshot_id, []).append(q)
        return q

    def _notify_locked(self, snapshot_id: str, status: SnapshotStatus) -> None:
        for q in self._subs.get(snapshot_id, []):
            try:
                q.put_nowait(status)
            except asyncio.QueueFull:
                pass


class InMemoryBranchingSessionStore(BranchingSessionStore):
    """Thread-safe, in-memory implementation of BranchingSessionStore."""

    def __init__(self, checkpoint_interval: int = 10) -> None:
        super().__init__(checkpoint_interval)
        self._records: dict[str, BranchRecord] = {}
        self._leaves: dict[str, list[str]] = {}  # session_id -> list of leaf snapshot_ids

    async def _append_child(self, session_id: str, parent_id: str | None, record: BranchRecord) -> None:
        self._records[record.snapshot_id] = record.model_copy(deep=True)

        leaves = self._leaves.setdefault(session_id, [])
        if parent_id in leaves:
            leaves.remove(parent_id)
        leaves.append(record.snapshot_id)

    async def _update_record(self, snapshot_id: str, record: BranchRecord) -> None:
        self._records[snapshot_id] = record.model_copy(deep=True)

    async def _read_record(self, snapshot_id: str) -> BranchRecord | None:
        rec = self._records.get(snapshot_id)
        return rec.model_copy(deep=True) if rec is not None else None

    async def _read_leaves(self, session_id: str) -> list[str]:
        return list(self._leaves.get(session_id, []))


class FileBranchingSessionStore(BranchingSessionStore):
    """Persistent, file-based implementation of BranchingSessionStore."""

    def __init__(self, directory: str, checkpoint_interval: int = 10) -> None:
        super().__init__(checkpoint_interval)
        self.directory = directory
        os.makedirs(os.path.join(directory, 'snapshots'), exist_ok=True)
        os.makedirs(os.path.join(directory, 'sessions'), exist_ok=True)

    def _snapshot_path(self, snapshot_id: str) -> str:
        return os.path.join(self.directory, 'snapshots', f'{snapshot_id}.json')

    def _leaves_path(self, session_id: str) -> str:
        return os.path.join(self.directory, 'sessions', session_id, 'leaves.json')

    async def _append_child(self, session_id: str, parent_id: str | None, record: BranchRecord) -> None:
        # Write record
        temp_path = self._snapshot_path(record.snapshot_id) + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(record.model_dump_json(indent=2))
        os.replace(temp_path, self._snapshot_path(record.snapshot_id))

        # Update leaves file
        os.makedirs(os.path.dirname(self._leaves_path(session_id)), exist_ok=True)
        leaves = await self._read_leaves(session_id)
        if parent_id in leaves:
            leaves.remove(parent_id)
        leaves.append(record.snapshot_id)

        leaves_temp = self._leaves_path(session_id) + '.tmp'
        with open(leaves_temp, 'w', encoding='utf-8') as f:
            json.dump(leaves, f, indent=2)
        os.replace(leaves_temp, self._leaves_path(session_id))

    async def _update_record(self, snapshot_id: str, record: BranchRecord) -> None:
        temp_path = self._snapshot_path(snapshot_id) + '.tmp'
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(record.model_dump_json(indent=2))
        os.replace(temp_path, self._snapshot_path(snapshot_id))

    async def _read_record(self, snapshot_id: str) -> BranchRecord | None:
        path = self._snapshot_path(snapshot_id)
        if not os.path.exists(path):
            return None
        with open(path, encoding='utf-8') as f:
            return BranchRecord.model_validate(json.load(f))

    async def _read_leaves(self, session_id: str) -> list[str]:
        path = self._leaves_path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, encoding='utf-8') as f:
            return list(json.load(f))
