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
