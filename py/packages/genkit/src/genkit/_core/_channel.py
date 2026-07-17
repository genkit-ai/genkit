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
import sys
from collections.abc import AsyncIterator, Coroutine
from typing import Any, Generic, TypeVar

from typing_extensions import TypeVar as TypeVarExt

from genkit._core._logger import get_logger

from ._compat import wait_for

if sys.version_info >= (3, 13):
    # Reuse the stdlib exception so a queue closed via native shutdown() and one
    # closed via the emulated path raise the exact same type, and so callers
    # can catch either interchangeably.
    from asyncio import QueueShutDown
else:

    class QueueShutDown(Exception):  # noqa: N818
        """Raised when interacting with a closed CloseableQueue."""


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


class CloseableQueue(asyncio.Queue[T]):
    """An asyncio.Queue subclass with a synchronous, idempotent close().

    Once closed, put()/put_nowait() raise QueueShutDown immediately, while
    get()/get_nowait() drain any buffered items first and then raise
    QueueShutDown once the queue is empty. close() also wakes coroutines that
    are already blocked in get() (and put() on a bounded queue). Supports async
    iteration via ``async for``.

    Python 3.13+ ships this as Queue.shutdown(), so we delegate to it there. On
    3.10-3.12 the same behavior is emulated by hand.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        """Close the queue synchronously and idempotently.

        Stops accepting new items, lets buffered items drain, then makes blocked
        and future getters raise QueueShutDown. Must be called on the event loop
        thread: asyncio.Queue is loop-affine and not thread-safe.
        """
        if self._closed:
            return
        self._closed = True

        if sys.version_info >= (3, 13):
            # native shutdown rejects new puts, leaves buffered items to drain,
            # and wakes blocked getters so they raise once the queue is empty.
            super().shutdown(immediate=False)
            return

        # wake anyone blocked in get() so they observe the closed-and-empty
        # state; the base queue tracks its waiters as plain deques.
        getters = getattr(self, '_getters', None)
        while getters:
            getter = getters.popleft()
            if not getter.done():
                getter.set_exception(QueueShutDown())

        # wake anyone blocked in put() on a bounded queue.
        putters = getattr(self, '_putters', None)
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
