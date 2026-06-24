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

"""LinearSessionStore implementation and variants."""

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

from genkit._ai._agents._session import SessionStore, SnapshotAborter
from genkit._ai._json_patch import apply_json_patch, diff_json
from genkit._core._error import GenkitError
from genkit._core._typing import (
    AgentFinishReason,
    JsonPatchOperation,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)

StateT = TypeVar('StateT')


class TurnRecord(BaseModel):
    """Storage model for a single sequence turn in a linear timeline."""

    session_id: str
    seq: int
    snapshot_id: str
    parent_id: str | None = None
    kind: str  # 'checkpoint' | 'diff'
    state_or_patch: Any  # SessionState dict or list of JsonPatchOperation dicts
    status: SnapshotStatus
    created_at: str
    finish_reason: str | None = None
    error: Any | None = None


class LinearSessionStore(SessionStore, SnapshotAborter):
    """Abstract SessionStore variant storing turns as a chain of incremental diffs."""

    def __init__(self, checkpoint_interval: int = 10) -> None:
        self.checkpoint_interval = checkpoint_interval
        self._lock = asyncio.Lock()
        self._subs: dict[str, list[asyncio.Queue[SnapshotStatus | None]]] = {}

    @abstractmethod
    async def _append_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        """Atomically append a turn record and update leaf index."""
        ...

    @abstractmethod
    async def _truncate_to(self, session_id: str, seq: int) -> None:
        """Atomically set session leaf seq to seq, and delete all turns after it."""
        ...

    @abstractmethod
    async def _update_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        """Update an existing turn record in place (updating status and final state)."""
        ...

    @abstractmethod
    async def _read_leaf_seq(self, session_id: str) -> int | None:
        """Read the sequence number of the current leaf for a session."""
        ...

    @abstractmethod
    async def _read_turn(self, session_id: str, seq: int) -> TurnRecord | None:
        """Read a turn record by sequence number."""
        ...

    @abstractmethod
    async def _read_turn_by_snapshot(self, snapshot_id: str) -> TurnRecord | None:
        """Read a turn record by snapshot_id."""
        ...

    async def _reconstruct_state(self, session_id: str, seq: int) -> SessionState:
        turns: list[TurnRecord] = []
        for s in range(seq + 1):
            t = await self._read_turn(session_id, s)
            if t is None:
                raise ValueError(f'Missing sequence {s} for session {session_id}')
            turns.append(t)

        checkpoint_idx = -1
        for i in range(seq, -1, -1):
            if turns[i].kind == 'checkpoint':
                checkpoint_idx = i
                break

        if checkpoint_idx == -1:
            raise ValueError(f'No checkpoint found in history for sequence {seq}')

        state_dict = copy.deepcopy(turns[checkpoint_idx].state_or_patch)

        for i in range(checkpoint_idx + 1, seq + 1):
            ops_data = turns[i].state_or_patch
            ops = [JsonPatchOperation.model_validate(op) for op in ops_data]
            state_dict = apply_json_patch(state_dict, ops)

        return SessionState.model_validate(state_dict)

    async def _reconstruct_snapshot(self, record: TurnRecord) -> SessionSnapshot:
        state = await self._reconstruct_state(record.session_id, record.seq)
        return SessionSnapshot(
            snapshot_id=record.snapshot_id,
            parent_id=record.parent_id,
            created_at=record.created_at,
            state=state,
            status=record.status,
            finish_reason=AgentFinishReason(record.finish_reason) if record.finish_reason else None,
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
                record = await self._read_turn_by_snapshot(snapshot_id)
                if record is None:
                    return None
                return await self._reconstruct_snapshot(record)

            assert session_id is not None
            leaf_seq = await self._read_leaf_seq(session_id)
            if leaf_seq is None:
                return None
            record = await self._read_turn(session_id, leaf_seq)
            if record is None:
                return None
            return await self._reconstruct_snapshot(record)

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[[SessionSnapshot | None], SessionSnapshot | None],
    ) -> SessionSnapshot | None:
        async with self._lock:
            if snapshot_id is not None:
                record = await self._read_turn_by_snapshot(snapshot_id)
                if record is None:
                    return None

                snap = await self._reconstruct_snapshot(record)
                next_snap = fn(snap)
                if next_snap is None:
                    return None

                next_snap.snapshot_id = snapshot_id

                # Update the record in place
                assert next_snap.state is not None
                if record.seq == 0 or record.seq % self.checkpoint_interval == 0:
                    record.state_or_patch = next_snap.state.model_dump(by_alias=True)
                else:
                    parent_state = await self._reconstruct_state(record.session_id, record.seq - 1)
                    ops = diff_json(parent_state.model_dump(by_alias=True), next_snap.state.model_dump(by_alias=True))
                    record.state_or_patch = [op.model_dump(by_alias=True) for op in ops]

                record.status = next_snap.status or SnapshotStatus.COMPLETED
                record.finish_reason = next_snap.finish_reason
                record.error = next_snap.error

                await self._update_turn(record.session_id, record.seq, record)
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
                    next_snap.status = SnapshotStatus.COMPLETED
                assert next_snap.state is not None
                session_id = next_snap.state.session_id
                parent_id = next_snap.parent_id

                if not parent_id:
                    seq = 0
                else:
                    parent_rec = await self._read_turn_by_snapshot(parent_id)
                    if parent_rec is None:
                        raise ValueError(f'Parent snapshot {parent_id} not found')
                    session_id = parent_rec.session_id
                    parent_seq = parent_rec.seq
                    leaf_seq = await self._read_leaf_seq(session_id)
                    if leaf_seq is not None and parent_seq < leaf_seq:
                        # Rollback: Truncate history after parent_seq
                        await self._truncate_to(session_id, parent_seq)

                    seq = parent_seq + 1

                assert session_id is not None
                if seq == 0 or seq % self.checkpoint_interval == 0:
                    kind = 'checkpoint'
                    state_or_patch = next_snap.state.model_dump(by_alias=True)
                else:
                    parent_state = await self._reconstruct_state(session_id, seq - 1)
                    ops = diff_json(parent_state.model_dump(by_alias=True), next_snap.state.model_dump(by_alias=True))
                    kind = 'diff'
                    state_or_patch = [op.model_dump(by_alias=True) for op in ops]

                record = TurnRecord(
                    session_id=session_id,
                    seq=seq,
                    snapshot_id=sid,
                    parent_id=parent_id,
                    kind=kind,
                    state_or_patch=state_or_patch,
                    status=next_snap.status,
                    created_at=next_snap.created_at,
                    finish_reason=next_snap.finish_reason,
                    error=next_snap.error,
                )

                await self._append_turn(session_id, seq, record)
                self._notify_locked(sid, next_snap.status)
                return next_snap

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        async with self._lock:
            record = await self._read_turn_by_snapshot(snapshot_id)
            if record is None:
                return None
            if record.status == SnapshotStatus.PENDING:
                record.status = SnapshotStatus.ABORTED
                await self._update_turn(record.session_id, record.seq, record)
                self._notify_locked(snapshot_id, record.status)
            return record.status

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
        async with self._lock:
            record = await self._read_turn_by_snapshot(snapshot_id)
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


class InMemoryLinearSessionStore(LinearSessionStore):
    """Thread-safe, in-memory implementation of LinearSessionStore."""

    def __init__(self, checkpoint_interval: int = 10) -> None:
        super().__init__(checkpoint_interval)
        self._turns: dict[str, dict[int, TurnRecord]] = {}

    async def _append_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        self._turns.setdefault(session_id, {})[seq] = record.model_copy(deep=True)

    async def _truncate_to(self, session_id: str, seq: int) -> None:
        turns = self._turns.get(session_id, {})
        for s in list(turns.keys()):
            if s > seq:
                turns.pop(s)

    async def _update_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        self._turns.setdefault(session_id, {})[seq] = record.model_copy(deep=True)

    async def _read_leaf_seq(self, session_id: str) -> int | None:
        turns = self._turns.get(session_id)
        if not turns:
            return None
        return max(turns.keys())

    async def _read_turn(self, session_id: str, seq: int) -> TurnRecord | None:
        rec = self._turns.get(session_id, {}).get(seq)
        return rec.model_copy(deep=True) if rec is not None else None

    async def _read_turn_by_snapshot(self, snapshot_id: str) -> TurnRecord | None:
        for session_turns in self._turns.values():
            for turn in session_turns.values():
                if turn.snapshot_id == snapshot_id:
                    return turn.model_copy(deep=True)
        return None


class FileLinearSessionStore(LinearSessionStore):
    """Persistent, file-based implementation of LinearSessionStore."""

    def __init__(self, directory: str, checkpoint_interval: int = 10) -> None:
        super().__init__(checkpoint_interval)
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _session_dir(self, session_id: str) -> str:
        return os.path.join(self.directory, session_id)

    def _turn_path(self, session_id: str, seq: int) -> str:
        return os.path.join(self._session_dir(session_id), f'{seq}.json')

    def _leaf_path(self, session_id: str) -> str:
        return os.path.join(self._session_dir(session_id), 'leaf.ptr')

    def _pointer_path(self, snapshot_id: str) -> str:
        return os.path.join(self.directory, f'{snapshot_id}.ptr')

    async def _append_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        def sync_op() -> None:
            os.makedirs(self._session_dir(session_id), exist_ok=True)

            # Write turn file
            temp_path = self._turn_path(session_id, seq) + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(record.model_dump_json(indent=2))
            os.replace(temp_path, self._turn_path(session_id, seq))

            # Write index pointer snapshot_id -> session_id:seq
            with open(self._pointer_path(record.snapshot_id), 'w', encoding='utf-8') as f:
                f.write(f'{session_id}:{seq}')

            # Update leaf sequence pointer
            leaf_temp = self._leaf_path(session_id) + '.tmp'
            with open(leaf_temp, 'w', encoding='utf-8') as f:
                f.write(str(seq))
            os.replace(leaf_temp, self._leaf_path(session_id))

        await asyncio.to_thread(sync_op)

    async def _truncate_to(self, session_id: str, seq: int) -> None:
        leaf_seq = await self._read_leaf_seq(session_id)
        if leaf_seq is None:
            return

        def sync_op(curr_leaf_seq: int) -> None:
            # Delete turns and index pointers from seq + 1 to leaf_seq
            for s in range(seq + 1, curr_leaf_seq + 1):
                t_path = self._turn_path(session_id, s)
                if os.path.exists(t_path):
                    # read snapshot_id to clean its pointer file
                    with open(t_path, encoding='utf-8') as f:
                        rec = json.load(f)
                    ptr_path = self._pointer_path(rec['snapshot_id'])
                    if os.path.exists(ptr_path):
                        os.remove(ptr_path)
                    os.remove(t_path)

            # Update leaf pointer
            with open(self._leaf_path(session_id), 'w', encoding='utf-8') as f:
                f.write(str(seq))

        await asyncio.to_thread(sync_op, leaf_seq)

    async def _update_turn(self, session_id: str, seq: int, record: TurnRecord) -> None:
        def sync_op() -> None:
            temp_path = self._turn_path(session_id, seq) + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(record.model_dump_json(indent=2))
            os.replace(temp_path, self._turn_path(session_id, seq))

        await asyncio.to_thread(sync_op)

    async def _read_leaf_seq(self, session_id: str) -> int | None:
        def sync_op() -> int | None:
            path = self._leaf_path(session_id)
            if not os.path.exists(path):
                return None
            with open(path, encoding='utf-8') as f:
                return int(f.read().strip())

        return await asyncio.to_thread(sync_op)

    async def _read_turn(self, session_id: str, seq: int) -> TurnRecord | None:
        def sync_op() -> TurnRecord | None:
            path = self._turn_path(session_id, seq)
            if not os.path.exists(path):
                return None
            with open(path, encoding='utf-8') as f:
                return TurnRecord.model_validate(json.load(f))

        return await asyncio.to_thread(sync_op)

    async def _read_turn_by_snapshot(self, snapshot_id: str) -> TurnRecord | None:
        def sync_op() -> str | None:
            ptr_path = self._pointer_path(snapshot_id)
            if not os.path.exists(ptr_path):
                return None
            with open(ptr_path, encoding='utf-8') as f:
                return f.read().strip()

        ref = await asyncio.to_thread(sync_op)
        if ref is None:
            return None
        session_id, seq_str = ref.split(':')
        return await self._read_turn(session_id, int(seq_str))
