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

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import Any, TypeVar

import websockets

from genkit._ai._agents._client import AgentTransport
from genkit._core._typing import AgentInit, AgentInput, AgentOutput, AgentStreamChunk, SessionSnapshot, SnapshotStatus

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')

_SENTINEL = object()


class WebSocketAgentTransport(AgentTransport[StateT, StreamT]):
    """Client-side agent transport that talks to a remote agent over WebSockets."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._ws: Any = None
        self._lock = asyncio.Lock()  # Guarantees sequential turn execution over the shared socket

    async def run_turn(
        self,
        input: AgentInput,  # noqa: A002
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        # 1. Acquire the lock to guarantee sequential access to the shared socket
        await self._lock.acquire()

        lock_released = False

        def release_lock(*args: object) -> None:
            nonlocal lock_released
            if not lock_released:
                lock_released = True
                if self._lock.locked():
                    self._lock.release()

        try:
            # 2. Check for closed socket and reconnect if needed
            if self._ws is None or self._ws.closed:
                self._ws = await websockets.connect(self.url)
            ws = self._ws
            if ws is None:
                raise RuntimeError('WebSocket failed to connect')

            await ws.send(
                json.dumps({
                    'init': init.model_dump(by_alias=True),
                    'input': input.model_dump(by_alias=True),
                })
            )

            output_future = asyncio.get_event_loop().create_future()
            stream_queue: asyncio.Queue[Any] = asyncio.Queue()

            # Auto-release the lock when the turn output resolves or fails
            output_future.add_done_callback(release_lock)

            async def watch_abort() -> None:
                if abort_event is not None:
                    await abort_event.wait()
                    await self.close()
                    if not output_future.done():
                        output_future.set_exception(asyncio.CancelledError())

            abort_task = asyncio.create_task(watch_abort())
            output_future.add_done_callback(lambda _: abort_task.cancel())

            # 3. Background reader task that consumes messages from the socket
            async def drain_ws() -> None:
                try:
                    async for message in ws:
                        data = json.loads(message)
                        if 'chunk' in data:
                            chunk = AgentStreamChunk.model_validate(data['chunk'])
                            stream_queue.put_nowait(chunk)
                        if 'output' in data:
                            output = AgentOutput.model_validate(data['output'])
                            if not output_future.done():
                                output_future.set_result(output)
                            if stream_queue is not None:
                                stream_queue.put_nowait(_SENTINEL)
                            break
                    else:
                        # Normal socket termination without receiving final output (Premature Close)
                        if not output_future.done():
                            output_future.set_exception(websockets.exceptions.ConnectionClosedOK(None, None))
                        stream_queue.put_nowait(websockets.exceptions.ConnectionClosedOK(None, None))
                except BaseException as e:
                    if not output_future.done():
                        output_future.set_exception(e)
                    stream_queue.put_nowait(e)

            asyncio.create_task(drain_ws())

            async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
                try:
                    while True:
                        item = await stream_queue.get()
                        if item is _SENTINEL:
                            break
                        if isinstance(item, BaseException):
                            raise item
                        yield item
                finally:
                    # Release the lock if the stream finishes before the output future
                    release_lock()

            return stream_generator(), output_future

        except Exception as e:
            release_lock()
            raise e

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        return None

    async def close(self) -> AgentOutput | None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        return None
