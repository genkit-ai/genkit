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

"""Asyncio loop utilities."""

import asyncio
import threading
from asyncio import AbstractEventLoop
from collections.abc import AsyncIterable, Callable, Coroutine, Iterable
from typing import TypeVar

from genkit.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


def create_loop() -> AbstractEventLoop:
    """Creates a new asyncio event loop or returns the current one.

    This function attempts to get the current event loop. If no current loop
    exists (e.g., in a new thread), it creates and returns a new event loop.

    Returns:
        An asyncio event loop instance.
    """
    try:
        return asyncio.get_event_loop()
    except Exception:
        return asyncio.new_event_loop()


def run_async(loop: asyncio.AbstractEventLoop, fn: Callable[[], Coroutine[object, object, T]]) -> T | None:
    """Runs an async callable on the given event loop and blocks until completion.

    If the loop is already running (e.g., called from within an async context),
    it schedules the callable using `asyncio.run_coroutine_threadsafe` and uses
    a threading lock to block until the callable finishes.

    If the loop is not running, it uses `loop.run_until_complete`.

    Args:
        loop: The asyncio event loop to run the callable on.
        fn: The async callable (e.g., a coroutine function) to execute.

    Returns:
        The result returned by the callable `fn`.

    Raises:
        Any exception raised by the callable `fn`.
    """
    if loop.is_running():
        output: T | None = None
        error: Exception | None = None
        lock = threading.Lock()
        _ = lock.acquire()

        async def run_fn() -> T | None:
            nonlocal lock
            nonlocal output
            nonlocal error
            try:
                output = await fn()
                return output
            except Exception as e:
                error = e
            finally:
                lock.release()
            return None

        _ = asyncio.run_coroutine_threadsafe(run_fn(), loop=loop)

        def wait_for_done() -> None:
            nonlocal lock
            _ = lock.acquire()

        thread = threading.Thread(target=wait_for_done)
        thread.start()
        thread.join()

        if error:
            raise error  # pyright: ignore[reportUnreachable]

        return output
    else:
        return loop.run_until_complete(fn())


def iter_over_async(ait: AsyncIterable[T], loop: asyncio.AbstractEventLoop) -> Iterable[T]:
    """Synchronously iterates over an AsyncIterable using a specified event loop.

    This function bridges asynchronous iteration with synchronous code by
    running the `__anext__` calls of the async iterator on the provided event
    loop and yielding the results synchronously.

    Args:
        ait: The asynchronous iterable to iterate over.
        loop: The asyncio event loop to use for running `__anext__`.

    Yields:
        Items from the asynchronous iterable.
    """
    ait_iter = ait.__aiter__()

    async def get_next() -> tuple[bool, T | None]:
        try:
            obj = await ait_iter.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        assert obj is not None  # Type narrowing: obj is T when done=False
        yield obj


def run_loop(coro: Coroutine[object, object, T], *, debug: bool | None = None) -> T:
    """Runs a coroutine using uvloop if available.

    Otherwise uses plain `asyncio.run`.

    Args:
        coro: The asynchronous coroutine to run.
        debug: If True, run in debug mode.
    """
    try:
        # Lazy import: uvloop is optional and only loaded if available
        import uvloop  # noqa: PLC0415

        logger.debug('✅ Using uvloop (recommended)')
        return uvloop.run(coro, debug=debug)
    except ImportError as e:
        logger.debug(
            '❓ Using asyncio (install uvloop for better performance)',
            error=e,
        )
        return asyncio.run(coro, debug=debug)
