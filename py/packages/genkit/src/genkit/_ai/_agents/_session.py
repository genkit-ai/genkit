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
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from enum import Enum
from typing import Any, Generic, Protocol, cast, runtime_checkable

from typing_extensions import TypeVar as TypeVarExt

from genkit._core._error import GenkitError
from genkit._core._typing import (
    Artifact,
    MessageData,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)


class SessionErrorType(str, Enum):
    """Types of session-related errors returned in GenkitError details."""

    AMBIGUOUS_BRANCH = 'ambiguousBranch'


StateT = TypeVarExt('StateT', default=Any)
SessionContextT = TypeVarExt('SessionContextT', default=Any)
# A store only ever hands custom state back out (it's a phantom over the wire
# format), so its parameter is covariant.
StateT_co = TypeVarExt('StateT_co', covariant=True, default=Any)


@runtime_checkable
class SessionStore(Protocol, Generic[StateT_co]):
    """Structural interface for snapshot persistence backends.

    Minimum: ``get_snapshot`` + ``save_snapshot``.
    Optional detach/abort support: implement ``SnapshotSubscriber`` as well.

    The ``StateT`` parameter names the custom-state shape a store round-trips,
    so a typed store agrees with its agent's ``state_schema``. It's a phantom
    over the snapshot wire format (which stays plain JSON), so leaving it off
    just defaults to ``Any``.
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
class SnapshotSubscriber(Protocol):
    """Optional capability that makes a store's snapshots abortable/detachable.

    Aborting itself is just a ``save_snapshot`` that flips a pending snapshot to
    aborted — there's no separate abort method. This is the other half: a way to
    *notice* that flip (e.g. when a different request aborts a detached turn
    that's still running) so the runtime can cancel the background work. A store
    that can't signal status changes can't support detach.
    """

    async def on_snapshot_status_change(self, snapshot_id: str) -> asyncio.Queue[SnapshotStatus | None]:
        """Queue that receives status changes; None sentinel when done."""
        ...


def supports_abort(store: SessionStore) -> bool:
    """True when the store can signal status changes, and so supports detach/abort."""
    return isinstance(store, SnapshotSubscriber)


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
            status='FAILED_PRECONDITION',
            message=(
                f"Session '{session_id}' has no leaf snapshot (corrupt or cyclic "
                'history). Resume by snapshot_id instead.'
            ),
        )

    raise GenkitError(
        status='FAILED_PRECONDITION',
        message=(
            f"Session '{session_id}' has branching snapshots ({len(leaves)} "
            'leaves), so there is no single latest snapshot. This happens when a '
            'conversation is branched (e.g. regenerate). Resume by '
            'snapshot_id instead.'
        ),
        details={
            'type': SessionErrorType.AMBIGUOUS_BRANCH,
            'sessionId': session_id,
            'leaves': [snap.snapshot_id for snap in leaves],
        },
    )


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
        self.lock = asyncio.Lock()
        self.session_state: SessionState = initial_state or SessionState()
        self.store = store
        self.version: int = 0
        self.custom_changed_listeners: list[Callable[[], Awaitable[None]]] = []
        self.artifact_changed_listeners: list[Callable[[Artifact], Awaitable[None]]] = []

    def on_custom_changed(self, listener: Callable[[], Awaitable[None]]) -> None:
        """Register a callback invoked after ``update_custom`` mutates state."""
        self.custom_changed_listeners.append(listener)

    def on_artifact_changed(self, listener: Callable[[Artifact], Awaitable[None]]) -> None:
        """Register a callback invoked after ``add_artifacts`` mutates state."""
        self.artifact_changed_listeners.append(listener)

    async def notify_custom_changed(self) -> None:
        for listener in self.custom_changed_listeners:
            await listener()

    async def notify_artifact_changed(self, artifact: Artifact) -> None:
        for listener in self.artifact_changed_listeners:
            await listener(artifact)

    async def state(self) -> SessionState:
        """Deep copy of current state."""
        async with self.lock:
            copied = self.session_state.model_copy(deep=True)
            if self.session_state.custom is not None:
                copied.custom = copy.deepcopy(self.session_state.custom)
            return copied

    async def get_messages(self) -> list[MessageData]:
        async with self.lock:
            return list(self.session_state.messages or [])

    async def add_messages(self, *messages: MessageData) -> None:
        async with self.lock:
            if self.session_state.messages is None:
                self.session_state.messages = []
            self.session_state.messages.extend(messages)
            self.version += 1

    async def set_messages(self, messages: list[MessageData]) -> None:
        async with self.lock:
            self.session_state.messages = list(messages)
            self.version += 1

    async def update_messages(self, fn: Callable[[list[MessageData]], list[MessageData]]) -> None:
        async with self.lock:
            self.session_state.messages = fn(list(self.session_state.messages or []))
            self.version += 1

    async def get_custom(self) -> StateT | None:
        async with self.lock:
            return cast(StateT | None, self.session_state.custom)

    async def update_custom(self, fn: Callable[[StateT | None], StateT]) -> None:
        async with self.lock:
            self.session_state.custom = fn(cast(StateT | None, self.session_state.custom))
            self.version += 1
        await self.notify_custom_changed()

    async def get_artifacts(self) -> list[Artifact]:
        async with self.lock:
            return list(self.session_state.artifacts or [])

    async def add_artifacts(self, *artifacts: Artifact) -> None:
        """Append artifacts; replace by name if artifact.name already exists."""
        changed: list[Artifact] = []
        async with self.lock:
            if self.session_state.artifacts is None:
                self.session_state.artifacts = []
            for art in artifacts:
                replaced = False
                if art.name:
                    for i, existing in enumerate(self.session_state.artifacts):
                        if existing.name == art.name:
                            self.session_state.artifacts[i] = art
                            replaced = True
                            break
                if not replaced:
                    self.session_state.artifacts.append(art)
                changed.append(art)
            self.version += 1
        for art in changed:
            await self.notify_artifact_changed(art)

    async def update_artifacts(self, fn: Callable[[list[Artifact]], list[Artifact]]) -> None:
        async with self.lock:
            self.session_state.artifacts = fn(list(self.session_state.artifacts or []))
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
