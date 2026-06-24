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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import TypeVar

from genkit._ai._agents._session import SessionStore, SnapshotAborter
from genkit._core._action import BidiAction, BidiConnection
from genkit._core._typing import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    SessionSnapshot,
    SnapshotStatus,
)

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')


class InProcessTransport:
    """In-process transport representing the trivial case where there is no physical transport.

    This runs the agent's bidirectional action directly in the same process and interacts
    directly with the session store that the agent writes to. The store is encapsulated
    and not exposed as a public field.
    """

    def __init__(
        self,
        action: BidiAction,
        store: SessionStore | None,
    ) -> None:
        """Initialise; store is captured privately and not exposed as a public field."""
        self._action = action
        self._conn: BidiConnection[AgentInput, AgentStreamChunk, AgentOutput] | None = None
        self._final_output: AgentOutput | None = None
        self._store = store

    def __copy__(self) -> InProcessTransport:
        """Create a shallow copy of the transport, resetting connection state to None."""
        return InProcessTransport(
            action=self._action,
            store=self._store,
        )

    async def run_turn(
        self,
        input: AgentInput,  # noqa: A002
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Run a single turn and return the stream and output awaitables."""
        if self._conn is None:
            self._conn = await self._action.stream_bidi(init)
        conn = self._conn

        await conn.send(input)

        output_future = asyncio.get_event_loop().create_future()
        stream_queue = asyncio.Queue()

        async def watch_abort() -> None:
            if abort_event is not None:
                await abort_event.wait()
                await conn.close()
                self._conn = None
                try:
                    await conn.output()
                except asyncio.CancelledError:
                    pass
                if not output_future.done():
                    output_future.set_exception(asyncio.CancelledError())

        abort_task = asyncio.create_task(watch_abort())
        output_future.add_done_callback(lambda _: abort_task.cancel())

        async def drain_connection() -> None:
            try:
                async for chunk in conn.receive():
                    if chunk.turn_end:
                        snapshot_id = chunk.turn_end.snapshot_id
                        state = None
                        message = None
                        artifacts = []
                        if snapshot_id:
                            snap = await self.get_snapshot(snapshot_id)
                            if snap and snap.state:
                                state = snap.state
                                if snap.state.messages:
                                    message = snap.state.messages[-1]
                                if snap.state.artifacts:
                                    artifacts = list(snap.state.artifacts)

                        output = AgentOutput(
                            snapshot_id=snapshot_id,
                            finish_reason=chunk.turn_end.finish_reason,
                            error=getattr(chunk.turn_end, 'error', None),
                            state=state,
                            message=message,
                            artifacts=artifacts,
                        )
                        if not output_future.done():
                            output_future.set_result(output)

                    stream_queue.put_nowait(chunk)
                    if chunk.turn_end:
                        break

                if not output_future.done():
                    out = await conn.output()
                    output_future.set_result(out)
            except BaseException as e:
                if not output_future.done():
                    output_future.set_exception(e)
                stream_queue.put_nowait(e)
            finally:
                stream_queue.put_nowait(None)

        asyncio.create_task(drain_connection())

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            while True:
                item = await stream_queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Retrieve a session snapshot via the store captured at construction."""
        if self._store is None:
            return None
        return await self._store.get_snapshot(snapshot_id=snapshot_id)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Abort a snapshot via the store captured at construction."""
        if not isinstance(self._store, SnapshotAborter):
            return None
        return await self._store.abort_snapshot(snapshot_id)

    async def close(self) -> None:
        """Close the underlying bidi connection."""
        if self._conn is not None:
            await self._conn.close()
            try:
                self._final_output = await self._conn.output()
            except Exception:  # noqa: BLE001
                self._final_output = None
            self._conn = None
