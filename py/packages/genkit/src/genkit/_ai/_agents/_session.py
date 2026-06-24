# Copyright 2025 Google LLC
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

"""Agent session state and snapshot persistence."""

from __future__ import annotations

import asyncio
import copy
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel

from genkit._core._error import GenkitError, StatusCodes
from genkit._core._typing import (
    Artifact,
    MessageData,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)

# Bare RFC-4122 UUID — session ids from useChat must match this shape.
SESSION_ID_PATTERN = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
    re.IGNORECASE,
)

StateT = TypeVar('StateT')
SessionContextT = TypeVar('SessionContextT')


class SnapshotContext(BaseModel):
    """Passed to SnapshotCallback to decide whether to snapshot."""

    state: SessionState
    prev_state: SessionState | None = None
    turn_index: int = 0


SnapshotCallback = Callable[[SnapshotContext], bool]


@runtime_checkable
class SessionStore(Protocol):
    """Structural interface for snapshot persistence backends.

    Minimum: ``get_snapshot`` + ``save_snapshot``.
    Optional abort lifecycle: implement ``SnapshotAborter`` as well.
    """

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Retrieve a snapshot by id or the latest leaf for a session."""
        ...

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[
            [SessionSnapshot | None],
            SessionSnapshot | None,
        ],
    ) -> SessionSnapshot | None:
        """Atomically read-modify-write a snapshot.

        fn receives the existing snapshot (or None for new) and returns the
        snapshot to persist, or None to skip. fn must be side-effect free —
        stores may call it more than once under contention.

        The store populates snapshot_id and created_at, and defaults status to
        done when fn leaves it empty.
        """
        ...


@runtime_checkable
class SnapshotAborter(Protocol):
    """Optional abort/status subscription surface for server-managed agents."""

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Flip PENDING → ABORTED. Return resulting status or None if not found."""
        ...

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Queue that receives status changes; None sentinel when done."""
        ...


def supports_abort(store: SessionStore) -> bool:
    """True when the store implements abort + status subscription."""
    return isinstance(store, SnapshotAborter)


def assert_valid_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.match(session_id):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(f"Invalid sessionId '{session_id}': must be a bare UUID (e.g. from crypto.randomUUID())."),
        )


def select_leaf_snapshot(
    snapshots: list[SessionSnapshot],
    session_id: str,
) -> SessionSnapshot | None:
    if not snapshots:
        return None

    parent_ids = {snap.parent_id for snap in snapshots if snap.parent_id}
    leaves = [snap for snap in snapshots if snap.snapshot_id not in parent_ids]

    if len(leaves) == 1:
        return leaves[0]

    if not leaves:
        raise GenkitError(
            status=StatusCodes.FAILED_PRECONDITION,
            message=(
                f"Session '{session_id}' has no leaf snapshot (corrupt or cyclic "
                'history). Resume by snapshot_id instead.'
            ),
        )

    raise GenkitError(
        status=StatusCodes.FAILED_PRECONDITION,
        message=(
            f"Session '{session_id}' has branching snapshots ({len(leaves)} "
            'leaves), so there is no single latest snapshot. This happens when a '
            'conversation is branched (e.g. regenerate). Resume by '
            'snapshot_id instead.'
        ),
        details={
            'type': 'ambiguous_branch',
            'sessionId': session_id,
            'leaves': [snap.snapshot_id for snap in leaves],
        },
    )


class InMemorySessionStore:
    """Thread-safe in-memory snapshot store for dev/tests.

    Implements SessionStore + SnapshotAborter. State lost on process exit.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._snapshots: dict[str, SessionSnapshot] = {}
        self._subs: dict[str, list[asyncio.Queue[SnapshotStatus | None]]] = {}

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
                snap = self._snapshots.get(snapshot_id)
                return snap.model_copy(deep=True) if snap is not None else None

            assert session_id is not None
            assert_valid_session_id(session_id)
            owned = [
                snap
                for snap in self._snapshots.values()
                if snap.state is not None and snap.state.session_id == session_id
            ]
            leaf = select_leaf_snapshot(owned, session_id)
            return leaf.model_copy(deep=True) if leaf is not None else None

    async def latest_snapshot(self) -> SessionSnapshot | None:
        """Most-recent snapshot by created_at. Not part of SessionStore interface."""
        async with self._lock:
            if not self._snapshots:
                return None
            latest = max(self._snapshots.values(), key=lambda s: s.created_at)
            return latest.model_copy(deep=True)

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[
            [SessionSnapshot | None],
            SessionSnapshot | None,
        ],
    ) -> SessionSnapshot | None:
        async with self._lock:
            sid = snapshot_id or str(uuid4())
            existing = self._snapshots.get(sid)
            existing_copy = existing.model_copy(deep=True) if existing is not None else None

            next_snap = fn(existing_copy)
            if next_snap is None:
                return None

            now = datetime.now(timezone.utc).isoformat()
            next_snap.snapshot_id = sid
            if existing is not None:
                next_snap.created_at = existing.created_at
            elif not next_snap.created_at:
                next_snap.created_at = now
            if not next_snap.status:
                next_snap.status = SnapshotStatus.COMPLETED

            self._snapshots[sid] = next_snap.model_copy(deep=True)

            if existing is None or existing.status != next_snap.status:
                self.notify_locked(sid, next_snap.status or SnapshotStatus.COMPLETED)

            return next_snap

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        async with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                return None
            if snap.status == SnapshotStatus.PENDING:
                snap.status = SnapshotStatus.ABORTED
                self.notify_locked(snapshot_id, snap.status)
            return snap.status

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        q: asyncio.Queue[SnapshotStatus | None] = asyncio.Queue()
        async with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                await q.put(None)
                return q
            await q.put(snap.status)
            self._subs.setdefault(snapshot_id, []).append(q)
        return q

    def notify_locked(self, snapshot_id: str, status: SnapshotStatus) -> None:
        for q in self._subs.get(snapshot_id, []):
            try:
                q.put_nowait(status)
            except asyncio.QueueFull:
                pass


class Session(Generic[StateT]):
    """Holds conversation state with asyncio-safe read/write access.

    Parameterize with a custom-state type when agents carry typed ``custom``
    blobs (``Session[MyState]``). Wire storage stays ``SessionState.custom``.

    ``version`` bumps on every mutation so the runtime can skip redundant
    snapshot writes without deep-comparing state.
    """

    def __init__(
        self,
        initial_state: SessionState | None = None,
        store: SessionStore | None = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._state: SessionState = initial_state or SessionState()
        self._store = store
        self.version: int = 0
        self._custom_changed_listeners: list[Callable[[], Awaitable[None]]] = []
        self._artifact_changed_listeners: list[Callable[[Artifact], Awaitable[None]]] = []

    def on_custom_changed(self, listener: Callable[[], Awaitable[None]]) -> None:
        """Register a callback invoked after ``update_custom`` mutates state."""
        self._custom_changed_listeners.append(listener)

    def on_artifact_changed(self, listener: Callable[[Artifact], Awaitable[None]]) -> None:
        """Register a callback invoked after ``add_artifacts`` mutates state."""
        self._artifact_changed_listeners.append(listener)

    async def notify_custom_changed(self) -> None:
        for listener in self._custom_changed_listeners:
            await listener()

    async def notify_artifact_changed(self, artifact: Artifact) -> None:
        for listener in self._artifact_changed_listeners:
            await listener(artifact)

    async def state(self) -> SessionState:
        """Deep copy of current state."""
        async with self._lock:
            copied = self._state.model_copy(deep=True)
            if self._state.custom is not None:
                copied.custom = copy.deepcopy(self._state.custom)
            return copied

    async def get_messages(self) -> list[MessageData]:
        async with self._lock:
            return list(self._state.messages or [])

    async def add_messages(self, *messages: MessageData) -> None:
        async with self._lock:
            if self._state.messages is None:
                self._state.messages = []
            self._state.messages.extend(messages)
            self.version += 1

    async def set_messages(self, messages: list[MessageData]) -> None:
        async with self._lock:
            self._state.messages = list(messages)
            self.version += 1

    async def update_messages(self, fn: Callable[[list[MessageData]], list[MessageData]]) -> None:
        async with self._lock:
            self._state.messages = fn(list(self._state.messages or []))
            self.version += 1

    async def get_custom(self) -> StateT | None:
        async with self._lock:
            return cast(StateT | None, self._state.custom)

    async def update_custom(self, fn: Callable[[StateT | None], StateT]) -> None:
        async with self._lock:
            self._state.custom = fn(cast(StateT | None, self._state.custom))
            self.version += 1
        await self.notify_custom_changed()

    async def get_artifacts(self) -> list[Artifact]:
        async with self._lock:
            return list(self._state.artifacts or [])

    async def add_artifacts(self, *artifacts: Artifact) -> None:
        """Append artifacts; replace by name if artifact.name already exists."""
        changed: list[Artifact] = []
        async with self._lock:
            if self._state.artifacts is None:
                self._state.artifacts = []
            for art in artifacts:
                replaced = False
                if art.name:
                    for i, existing in enumerate(self._state.artifacts):
                        if existing.name == art.name:
                            self._state.artifacts[i] = art
                            replaced = True
                            break
                if not replaced:
                    self._state.artifacts.append(art)
                changed.append(art)
            self.version += 1
        for art in changed:
            await self.notify_artifact_changed(art)

    async def update_artifacts(self, fn: Callable[[list[Artifact]], list[Artifact]]) -> None:
        async with self._lock:
            self._state.artifacts = fn(list(self._state.artifacts or []))
            self.version += 1


# ---------------------------------------------------------------------------
# Session context (async-local binding for middleware and tools)
# ---------------------------------------------------------------------------

current_session: ContextVar[Session[Any] | None] = ContextVar('genkit.session', default=None)


def get_current_session() -> Session[Any] | None:
    """Return the session bound by :func:`run_with_session`, if any."""
    return current_session.get()


async def run_with_session(
    session: Session[StateT],
    coro: Awaitable[SessionContextT],
) -> SessionContextT:
    """Run ``coro`` with ``session`` available via :func:`get_current_session`."""
    token = current_session.set(session)
    try:
        return await coro
    finally:
        current_session.reset(token)
