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

"""In-process agent transport factory: executes the agent action directly in the same process."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

from genkit._ai._agents._types import StateManagement
from genkit._core._action import BidiAction
from genkit._core._channel import CloseableQueue
from genkit._core._typing import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    SessionSnapshot,
    SnapshotStatus,
)


class GetSnapshotFn(Protocol):
    async def __call__(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None: ...


AbortSnapshotFn = Callable[[str], Awaitable[SnapshotStatus | None]]


class InProcessTransport:
    """In-process transport: runs the agent bidi action in-process, no HTTP."""

    def __init__(
        self,
        action: BidiAction,
        *,
        get_snapshot: GetSnapshotFn | None = None,
        abort_snapshot: AbortSnapshotFn | None = None,
        state_management: StateManagement,
    ) -> None:
        self.action = action
        self._get_snapshot = get_snapshot
        self._abort_snapshot = abort_snapshot
        self.state_management: StateManagement = state_management
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def run_turn(
        self,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Run a single turn and return the stream and output awaitables.

        Each turn opens its own bidi connection seeded from ``init``, sends the
        one input, and closes the send side. The server holds the session for
        that turn and hands back the whole conversation on the turn's
        ``AgentOutput``, so the boundary of the connection is the boundary of
        the turn — the same shape as the stateless HTTP transport.
        """
        conn = await self.action.stream_bidi(init)
        await conn.send(agent_input)
        # One input per connection: closing the send side now tells the server
        # this turn has no follow-on inputs, so it can finalize and resolve
        # output().
        await conn.close()

        output_future: asyncio.Future[AgentOutput] = asyncio.Future()
        stream_queue = CloseableQueue[AgentStreamChunk | BaseException]()

        # Aborting a turn is a client-side detach: the caller stops listening,
        # but this drain keeps running to completion so the in-flight turn's work
        # and any snapshot still land. Halting server-side work is a separate
        # operation (abort_snapshot), not part of running a turn.
        async def drain_connection() -> None:
            try:
                async for chunk in conn.receive():
                    stream_queue.put_nowait(chunk)
                if not output_future.done():
                    output_future.set_result(await conn.output())
            except BaseException as e:
                if not output_future.done():
                    output_future.set_exception(e)
                stream_queue.put_nowait(e)
            finally:
                stream_queue.close()

        task = asyncio.create_task(drain_connection())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            async for chunk in stream_queue:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return stream_generator(), output_future

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        if self._get_snapshot is None:
            return None
        return await self._get_snapshot(snapshot_id=snapshot_id, session_id=session_id)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        if self._abort_snapshot is None:
            return None
        return await self._abort_snapshot(snapshot_id)

    async def close(self) -> None:
        """No persistent connection to tear down: each turn owns its own."""
        return None
