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

"""Asyncio helpers for asynchronous communication and data flow control."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Generic

from typing_extensions import TypeVar

from ._compat import wait_for

T = TypeVar('T')  # Type of items in the channel
R = TypeVar('R', default=Any)  # Type of the close future result (defaults to Any)


class Channel(Generic[T, R]):
    """An asynchronous channel for sending and receiving values.

    This class provides an asynchronous queue-like interface, allowing values to
    be sent and received between different parts of an asynchronous program. It
    supports both sending values and closing the channel, which will signal to
    any receivers that no more values will be sent.

    The Channel class implements the async iterator protocol, allowing it to be
    used in async for loops.

    Typical usage:
        ```python
        channel: Channel[int, int] = Channel(timeout=0.1)

        # Send values to the channel
        channel.send(1)
        channel.send(2)

        # Receive values from the channel in another task
        async for value in channel:
            print(value)  # Will print 1, then 2
        ```

    """

    def __init__(self, timeout: float | int | None = None) -> None:
        """Initializes a new Channel.

        The channel is initialized with an internal queue to store values, a
        future to signal when the channel is closed, and an internal close
        future that will be set when the channel should be closed.

        Args:
            timeout: The timeout in seconds for the __anext__ method.  If None,
                waits indefinitely (default).

        Raises:
            ValueError: If the timeout is negative.
        """
        if timeout is not None and timeout < 0:
            raise ValueError('Timeout must be non-negative')

        self.queue: asyncio.Queue[T] = asyncio.Queue()
        self.closed: asyncio.Future[R] = asyncio.Future()
        self._close_future: asyncio.Future[R] | None = None
        self._timeout: float | int | None = timeout

    def __aiter__(self) -> AsyncIterator[T]:
        """Returns the asynchronous iterator for the channel.

        Returns:
            AsyncIterator[T]: The channel object itself, which implements the
            `__anext__` method required for async iteration.
        """
        return self

    async def __anext__(self) -> T:
        """Retrieves the next value from the channel.

        If the queue is not empty, the value is returned immediately.
        Otherwise, it waits until a value is available or the channel is closed.

        Implements the `__anext__` method required for async iteration.

        Raises:
            StopAsyncIteration: If the channel is closed and no more values are
                available, signaling the end of the iteration.
            TimeoutError: If the timeout is exceeded while waiting for a
                value and a timeout has been specified.

        Returns:
            T: The next value from the channel.
        """
        # If the queue has values, return the next value immediately.
        # Otherwise, wait for a value to be available or the channel to close or
        # a timeout to occur.
        if not self.queue.empty():
            return self.queue.get_nowait()

        # Create the task to retrieve the next value
        pop_task = asyncio.ensure_future(self._pop())
        if not self._close_future:
            # Wait for the pop task with a timeout, raise TimeoutError if a
            # timeout is specified and is exceeded and automatically cancel the
            # pending task.
            return await wait_for(pop_task, timeout=self._timeout)

        # Wait for either the pop task or the close future to complete.  A
        # timeout is added to prevent indefinite blocking, unless
        # specifically set to None.
        # NOTE: asyncio.wait does not cancel tasks on timeout by default.
        finished, _pending = await asyncio.wait(
            [pop_task, self._close_future],
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self._timeout,
        )

        # If timeout occurred (nothing finished), cancel pop task and raise
        # Note: Don't cancel _close_future as it's owned by external code
        if not finished:
            _ = pop_task.cancel()
            raise TimeoutError('Channel timeout exceeded')

        # If the pop task completed, return its result.
        if pop_task in finished:
            return pop_task.result()

        # If the close future completed, raise StopAsyncIteration.
        if self._close_future in finished:
            # Cancel pop task if we're done, avoid warnings.
            _ = pop_task.cancel()
            raise StopAsyncIteration

        # Wait for the pop task with a timeout, raise TimeoutError if a timeout
        # is specified and is exceeded and automatically cancel the pending
        # task.
        return await wait_for(pop_task, timeout=self._timeout)

    def send(self, value: T) -> None:
        """Sends a value into the channel.

        The value is added to the internal queue for consumers to retrieve.
        This is a non-blocking operation.

        Args:
            value: The value to send through the channel.

        Raises:
            asyncio.QueueFull: If the channel's internal queue is full.

        Returns:
            None.
        """
        self.queue.put_nowait(value)

    def set_close_future(self, future: asyncio.Future[R]) -> None:
        """Sets a future that, when completed, will close the channel.

        When the provided future completes, the channel will be marked as
        closed, signaling to consumers that no more values will be sent.

        Args:
            future: The future to monitor for channel closure.  When this future
            completes, the channel will be closed.

        Raises:
            ValueError: If the provided future is None.
        """
        if future is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError('Cannot set a None future')  # pyright: ignore[reportUnreachable]

        def _handle_done(v: asyncio.Future[R]) -> None:
            """Handle future completion, propagating results or errors to self.closed.

            This callback ensures proper propagation of the future's final state
            (success, exception, or cancellation) to the channel's closed future,
            allowing consumers to properly handle completion or errors.
            """
            # Propagate cancellation to notify consumers that the operation was cancelled
            if v.cancelled():
                _ = self.closed.cancel()
            elif (exc := v.exception()) is not None:
                self.closed.set_exception(exc)
            else:
                self.closed.set_result(v.result())

        self._close_future = asyncio.ensure_future(future)
        if self._close_future is not None:  # pyright: ignore[reportUnnecessaryComparison]
            self._close_future.add_done_callback(_handle_done)

    async def _pop(self) -> T:
        """Asynchronously retrieves a value from the internal queue.

        This is an internal method used by the async iterator implementation to
        wait until a value is available in the queue.

        Raises:
            StopAsyncIteration: If a None value is retrieved, indicating the
                channel is closed.

        Returns:
            T: The retrieved value from the channel.
        """
        r = await self.queue.get()
        self.queue.task_done()
        if not r:
            raise StopAsyncIteration
        return r
