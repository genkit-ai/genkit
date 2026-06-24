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
#
# SPDX-License-Identifier: Apache-2.0

"""Channel for async streaming with final value, and uvloop-aware runner."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Coroutine
from typing import Any, Generic, TypeVar

from typing_extensions import TypeVar as TypeVarExt

from genkit._core._logger import get_logger

from ._compat import wait_for

logger = get_logger(__name__)

T = TypeVar('T')
T_co = TypeVarExt('T_co')
R = TypeVarExt('R', default=Any)


class Channel(Generic[T_co, R]):
    """Async channel for streaming values with a final result when closed."""

    def __init__(self, timeout: float | int | None = None) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError('Timeout must be non-negative')
        self.queue: asyncio.Queue[T_co] = asyncio.Queue()
        self.closed: asyncio.Future[R] = asyncio.Future()
        self._close_future: asyncio.Future[R] | None = None
        self._timeout = timeout

    def __aiter__(self) -> AsyncIterator[T_co]:
        return self

    async def __anext__(self) -> T_co:
        if not self.queue.empty():
            return self.queue.get_nowait()

        pop_task = asyncio.ensure_future(self._pop())
        if not self._close_future:
            return await wait_for(pop_task, timeout=self._timeout)

        finished, _ = await asyncio.wait(
            [pop_task, self._close_future],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self._timeout,
        )

        if not finished:
            _ = pop_task.cancel()
            raise TimeoutError('Channel timeout exceeded')

        if pop_task in finished:
            return pop_task.result()

        if self._close_future in finished:
            _ = pop_task.cancel()
            raise StopAsyncIteration

        return await wait_for(pop_task, timeout=self._timeout)

    def send(self, value: T_co) -> None:
        """Send a value into the channel."""
        self.queue.put_nowait(value)

    def set_close_future(self, future: asyncio.Future[R]) -> None:
        """Set a future that closes the channel when completed."""
        if future is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError('Cannot set a None future')  # pyright: ignore[reportUnreachable]

        def _handle_done(v: asyncio.Future[R]) -> None:
            if v.cancelled():
                _ = self.closed.cancel()
            elif (exc := v.exception()) is not None:
                self.closed.set_exception(exc)
            else:
                self.closed.set_result(v.result())

        self._close_future = asyncio.ensure_future(future)
        if self._close_future is not None:  # pyright: ignore[reportUnnecessaryComparison]
            self._close_future.add_done_callback(_handle_done)

    async def _pop(self) -> T_co:
        r = await self.queue.get()
        self.queue.task_done()
        if r is None:
            raise StopAsyncIteration
        return r


def run_loop(coro: Coroutine[object, object, T], *, debug: bool | None = None) -> T:
    """Run a coroutine using uvloop if available, otherwise asyncio."""
    try:
        import uvloop  # noqa: PLC0415

        logger.debug('Using uvloop (recommended)')
        return uvloop.run(coro, debug=debug)
    except ImportError as e:
        logger.debug('Using asyncio (install uvloop for better performance)', error=e)
        return asyncio.run(coro, debug=debug)


class QueueShutDown(Exception):  # noqa: N818
    """Exception raised when attempting to interact with a closed CloseableQueue."""

    pass


class CloseableQueue(asyncio.Queue[T]):
    """An asyncio.Queue subclass that supports native, synchronous close() semantics.

    Compatible with Python 3.10 and 3.11, emulating the Queue.shutdown() feature
    introduced in Python 3.12. Once closed, put() raises QueueShutDown, and get()
    raises QueueShutDown once the queue is completely drained. Supports native async iteration.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        """Synchronously close the queue. Non-blocking and thread-safe on the event loop.

        This method transitions the queue to a closed state. It immediately wakes up
        all pending getters and putters, raising a QueueShutDown exception inside them.
        This provides a purely synchronous, non-blocking way to close the queue,
        which is mathematically guaranteed to prevent deadlocks during teardown.
        """
        if self._closed:
            return
        self._closed = True

        # Wake up all pending readers/getters with QueueShutDown exception
        getters = getattr(self, '_getters', None)
        if getters:
            while getters:
                getter = getters.popleft()
                if not getter.done():
                    getter.set_exception(QueueShutDown())

        # Wake up all pending writers/putters with QueueShutDown exception
        putters = getattr(self, '_putters', None)
        if putters:
            while putters:
                putter = putters.popleft()
                if not putter.done():
                    putter.set_exception(QueueShutDown())

    def is_closed(self) -> bool:
        return self._closed

    async def put(self, item: T) -> None:
        if self._closed:
            raise QueueShutDown('Queue is closed')
        await super().put(item)

    def put_nowait(self, item: T) -> None:
        if self._closed:
            raise QueueShutDown('Queue is closed')
        super().put_nowait(item)

    async def get(self) -> T:
        # If the queue is closed and empty, raise QueueShutDown to signal end of stream
        if self._closed and self.empty():
            raise QueueShutDown('Queue is closed and empty')
        return await super().get()

    def get_nowait(self) -> T:
        if self._closed and self.empty():
            raise QueueShutDown('Queue is closed and empty')
        return super().get_nowait()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.get()
        except QueueShutDown:
            raise StopAsyncIteration from None
