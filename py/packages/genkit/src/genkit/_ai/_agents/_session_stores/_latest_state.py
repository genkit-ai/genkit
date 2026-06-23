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

"""LatestStateStore implementation and variants."""

from __future__ import annotations

import asyncio
import json
import os
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel

from genkit._ai._agents._session import SessionStore, SnapshotAborter
from genkit._core._error import GenkitError
from genkit._core._typing import SessionSnapshot, SnapshotStatus

StateT = TypeVar('StateT')


class LatestRecord(BaseModel):
    """Container for the latest state and optional pending slot of a session."""

    session_id: str
    last_good: SessionSnapshot | None = None
    pending: SessionSnapshot | None = None


class LatestStateStore(SessionStore, SnapshotAborter):
    """Abstract SessionStore variant that keeps only the latest state (+ 1 pending slot)."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._subs: dict[str, list[asyncio.Queue[SnapshotStatus | None]]] = {}

    @abstractmethod
    async def _put_record(self, record: LatestRecord) -> None:
        """Atomically persist or update a LatestRecord."""
        ...

    @abstractmethod
    async def _read_record_by_session(self, session_id: str) -> LatestRecord | None:
        """Lookup a LatestRecord by session_id."""
        ...

    @abstractmethod
    async def _read_record_by_snapshot(self, snapshot_id: str) -> LatestRecord | None:
        """Lookup a LatestRecord by snapshot_id (checking both slots)."""
        ...

    @abstractmethod
    async def _delete_record(self, session_id: str) -> None:
        """Delete a LatestRecord by session_id."""
        ...

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
                record = await self._read_record_by_snapshot(snapshot_id)
                if record is None:
                    return None
                if record.last_good and record.last_good.snapshot_id == snapshot_id:
                    return record.last_good.model_copy(deep=True)
                if record.pending and record.pending.snapshot_id == snapshot_id:
                    return record.pending.model_copy(deep=True)
                return None

            assert session_id is not None
            record = await self._read_record_by_session(session_id)
            if record is None or record.last_good is None:
                return None
            return record.last_good.model_copy(deep=True)

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[[SessionSnapshot | None], SessionSnapshot | None],
    ) -> SessionSnapshot | None:
        async with self._lock:
            if snapshot_id is not None:
                record = await self._read_record_by_snapshot(snapshot_id)
                if record is None:
                    return None

                is_pending = record.pending and record.pending.snapshot_id == snapshot_id
                target = record.pending if is_pending else record.last_good
                assert target is not None

                next_snap = fn(target.model_copy(deep=True))
                if next_snap is None:
                    return None

                next_snap.snapshot_id = snapshot_id

                if is_pending:
                    if next_snap.status != SnapshotStatus.PENDING:
                        record.last_good = next_snap
                        record.pending = None
                    else:
                        record.pending = next_snap
                else:
                    record.last_good = next_snap

                await self._put_record(record)
                self._notify_locked(snapshot_id, next_snap.status or SnapshotStatus.DONE)
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
                if not session_id:
                    raise ValueError('session_id must be populated on new snapshot')

                record = await self._read_record_by_session(session_id)
                if record is None:
                    record = LatestRecord(session_id=session_id)

                if next_snap.status == SnapshotStatus.PENDING:
                    record.pending = next_snap
                else:
                    record.last_good = next_snap
                    record.pending = None

                await self._put_record(record)
                self._notify_locked(sid, next_snap.status)
                return next_snap

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        async with self._lock:
            record = await self._read_record_by_snapshot(snapshot_id)
            if record is None:
                return None

            # Abort can only flip PENDING -> ABORTED
            if record.pending and record.pending.snapshot_id == snapshot_id:
                if record.pending.status == SnapshotStatus.PENDING:
                    record.pending.status = SnapshotStatus.ABORTED
                    await self._put_record(record)
                    self._notify_locked(snapshot_id, record.pending.status)
                    return record.pending.status
                return record.pending.status
            return record.last_good.status if record.last_good else None

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
        async with self._lock:
            record = await self._read_record_by_snapshot(snapshot_id)
            if record is None:
                await q.put(None)
                return q

            status = None
            if record.last_good and record.last_good.snapshot_id == snapshot_id:
                status = record.last_good.status
            elif record.pending and record.pending.snapshot_id == snapshot_id:
                status = record.pending.status

            await q.put(status)
            self._subs.setdefault(snapshot_id, []).append(q)
        return q

    def _notify_locked(self, snapshot_id: str, status: SnapshotStatus) -> None:
        for q in self._subs.get(snapshot_id, []):
            try:
                q.put_nowait(status)
            except asyncio.QueueFull:
                pass


class InMemoryLatestStateStore(LatestStateStore):
    """Thread-safe, in-memory implementation of LatestStateStore."""

    def __init__(self) -> None:
        super().__init__()
        self._records: dict[str, LatestRecord] = {}

    async def _put_record(self, record: LatestRecord) -> None:
        self._records[record.session_id] = record.model_copy(deep=True)

    async def _read_record_by_session(self, session_id: str) -> LatestRecord | None:
        rec = self._records.get(session_id)
        return rec.model_copy(deep=True) if rec is not None else None

    async def _read_record_by_snapshot(self, snapshot_id: str) -> LatestRecord | None:
        for rec in self._records.values():
            if rec.last_good and rec.last_good.snapshot_id == snapshot_id:
                return rec.model_copy(deep=True)
            if rec.pending and rec.pending.snapshot_id == snapshot_id:
                return rec.model_copy(deep=True)
        return None

    async def _delete_record(self, session_id: str) -> None:
        self._records.pop(session_id, None)


class FileLatestStateStore(LatestStateStore):
    """Persistent, file-based implementation of LatestStateStore."""

    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _session_path(self, session_id: str) -> str:
        return os.path.join(self.directory, f'{session_id}.json')

    def _pointer_path(self, snapshot_id: str) -> str:
        return os.path.join(self.directory, f'{snapshot_id}.ptr')

    async def _put_record(self, record: LatestRecord) -> None:
        # Save pointer paths first, then write the main record
        # Note: Delete old pointers first to avoid leakage
        old_record = await self._read_record_by_session(record.session_id)

        def sync_op() -> None:
            if old_record:
                if old_record.last_good:
                    old_good_ptr = self._pointer_path(old_record.last_good.snapshot_id)
                    if os.path.exists(old_good_ptr):
                        os.remove(old_good_ptr)
                if old_record.pending:
                    old_pending_ptr = self._pointer_path(old_record.pending.snapshot_id)
                    if os.path.exists(old_pending_ptr):
                        os.remove(old_pending_ptr)

            # Write new pointer files
            if record.last_good:
                with open(self._pointer_path(record.last_good.snapshot_id), 'w', encoding='utf-8') as f:
                    f.write(record.session_id)
            if record.pending:
                with open(self._pointer_path(record.pending.snapshot_id), 'w', encoding='utf-8') as f:
                    f.write(record.session_id)

            # Write JSON record
            temp_path = self._session_path(record.session_id) + '.tmp'
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(record.model_dump_json(indent=2))
            os.replace(temp_path, self._session_path(record.session_id))

        await asyncio.to_thread(sync_op)

    async def _read_record_by_session(self, session_id: str) -> LatestRecord | None:
        def sync_op() -> LatestRecord | None:
            path = self._session_path(session_id)
            if not os.path.exists(path):
                return None
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
            return LatestRecord.model_validate(data)

        return await asyncio.to_thread(sync_op)

    async def _read_record_by_snapshot(self, snapshot_id: str) -> LatestRecord | None:
        def sync_op() -> str | None:
            ptr_path = self._pointer_path(snapshot_id)
            if not os.path.exists(ptr_path):
                return None
            with open(ptr_path, encoding='utf-8') as f:
                return f.read().strip()

        session_id = await asyncio.to_thread(sync_op)
        if session_id is None:
            return None
        return await self._read_record_by_session(session_id)

    async def _delete_record(self, session_id: str) -> None:
        record = await self._read_record_by_session(session_id)

        def sync_op() -> None:
            if record:
                if record.last_good:
                    good_ptr = self._pointer_path(record.last_good.snapshot_id)
                    if os.path.exists(good_ptr):
                        os.remove(good_ptr)
                if record.pending:
                    pending_ptr = self._pointer_path(record.pending.snapshot_id)
                    if os.path.exists(pending_ptr):
                        os.remove(pending_ptr)
            path = self._session_path(session_id)
            if os.path.exists(path):
                os.remove(path)

        await asyncio.to_thread(sync_op)
