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

"""Compatibility layer for asyncio."""

import asyncio
import functools
import sys
from typing import TYPE_CHECKING, Any, TypeVar, overload as overload

# StrEnum - use strenum package for cross-version compatibility
# Note: StrEnum was added to stdlib in Python 3.11, but we use strenum for 3.10 compat
# override decorator - use typing_extensions for consistency across Python versions
# Note: override was added to typing in Python 3.12, but typing_extensions has it for all versions
if sys.version_info >= (3, 11):
    from enum import StrEnum as StrEnum  # noqa: E402
else:
    from strenum import StrEnum as StrEnum  # noqa: E402
from typing_extensions import override as override  # noqa: E402

if TYPE_CHECKING:
    from genkit._core._model import Message

_T = TypeVar('_T')


def cancelled_history(exc: BaseException) -> 'list[Message] | None':
    """Return the history that generate attached to a cancel, or None.

    On a foreign cancel, generate sets ``messages`` on the ``CancelledError``
    it re-raises: the history up to the last completed round, with no
    partial turn.

    - Direct ``task.cancel()`` on 3.11+ re-raises that same ``CancelledError``.
    - ``asyncio.wait_for`` and ``asyncio.timeout`` raise ``TimeoutError``
      (``asyncio.TimeoutError`` on 3.10) with the ``CancelledError`` on
      ``__cause__``.
    - On 3.10, each task hop raises a fresh ``CancelledError`` with the
      previous one on ``__context__``.
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (TimeoutError, asyncio.TimeoutError)):
            cur = cur.__cause__
            continue
        if isinstance(cur, asyncio.CancelledError) and 'messages' in cur.__dict__:
            return cur.__dict__['messages']
        if sys.version_info >= (3, 11):
            return None
        cur = cur.__context__
    return None


def _wake_waiter(waiter: asyncio.Future[None], _task: object) -> None:
    """Resolve ``waiter`` once when the watched task finishes."""
    if not waiter.done():
        waiter.set_result(None)


async def await_unchained(task: asyncio.Task[_T]) -> _T:
    """Await ``task`` without chaining ``Task._fut_waiter`` across tasks.

    Directly running ``await task`` sets ``parent._fut_waiter = task``. Across
    N nested task hops (e.g. 400 middleware hooks in ``hop()``), ``_fut_waiter``
    forms an N-deep ``Task -> Task`` linked list. Calling ``outer.cancel()`` or
    ``repr(outer)`` walks that chain recursively on the call stack and raises
    ``RecursionError`` on Python 3.10 and 3.11.

    Awaiting a leaf ``Future`` per step breaks that link:
    1. ``parent._fut_waiter`` points to a leaf ``Future`` (O(1) stack depth for
       both ``cancel()`` and ``repr()``).
    2. If the caller is cancelled while waiting, ``task.cancel()`` forwards the
       cancellation and the loop keeps waiting on fresh leaf ``Future``s until
       ``task`` finishes cleanup, matching standard ``await task`` behavior.
    3. Once ``task`` finishes, ``task.result()`` returns its value or re-raises
       its exception / ``CancelledError``.
    """
    loop = asyncio.get_running_loop()

    while not task.done():
        # 1. Suspend on a leaf Future so `parent._fut_waiter` never points to `task`.
        waiter: asyncio.Future[None] = loop.create_future()
        on_done = functools.partial(_wake_waiter, waiter)
        task.add_done_callback(on_done)
        try:
            await waiter
        except asyncio.CancelledError as err:
            # 2. Forward cancellation to `task`. If `task` already finished on
            #    this event-loop turn (`cancel()` returns False), the cancel
            #    targets the caller itself: retrieve any task error and re-raise.
            msg: Any = err.args[0] if err.args else None
            if not task.cancel(msg):
                if not task.cancelled():
                    _ = task.exception()
                raise
        finally:
            task.remove_done_callback(on_done)

    # 3. `task` has finished; propagate its return value, exception, or cancellation.
    return task.result()
