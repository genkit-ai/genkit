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
import weakref
from collections.abc import Callable
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

# Iterative trampoline queue so nested `_UnchainedWaiter.cancel()` calls forward
# cancellation across arbitrarily deep hop chains in O(1) call-stack frames.
_ACTIVE_CANCEL_QUEUE: list[tuple['_UnchainedWaiter', Any]] | None = None

# Python 3.10 lacks `Task.cancelling()` / `Task.uncancel()`. Track active
# cancel counts for tasks cancelled via `await_unchained` so `cancelling()` and
# `uncancel()` behave consistently across Python 3.10 and 3.11+.
_PY310_CANCELLING: weakref.WeakKeyDictionary[asyncio.Task[Any], int] = weakref.WeakKeyDictionary()


def uncancel(task: asyncio.Task[Any] | None) -> int:
    """Decrement the cancellation count of an asyncio.Task."""
    if task is None:
        return 0
    if sys.version_info >= (3, 11):
        return task.uncancel()
    remaining = max(0, _PY310_CANCELLING.get(task, 0) - 1)
    if remaining == 0:
        _PY310_CANCELLING.pop(task, None)
    else:
        _PY310_CANCELLING[task] = remaining
    return remaining


def cancelling(task: asyncio.Task[Any] | None) -> int:
    """Return the cancellation count of an asyncio.Task."""
    if task is None:
        return 0
    if sys.version_info >= (3, 11):
        return task.cancelling()
    return _PY310_CANCELLING.get(task, 0)


def _cancel_task_unchained(task: asyncio.Task[Any], msg: Any = None) -> bool:  # noqa: ANN401
    """Cancel ``task`` and iteratively drain any nested ``_UnchainedWaiter`` children."""
    global _ACTIVE_CANCEL_QUEUE  # noqa: PLW0603
    if task.done():
        return False
    queue: list[tuple[_UnchainedWaiter, Any]] = []
    _ACTIVE_CANCEL_QUEUE = queue
    try:
        accepted = task.cancel(msg)
        if accepted and sys.version_info < (3, 11):
            _PY310_CANCELLING[task] = _PY310_CANCELLING.get(task, 0) + 1
        while queue:
            waiter, next_msg = queue.pop()
            child = waiter.child_task
            child_accepted = False if child.done() else child.cancel(next_msg)
            waiter.child_cancel_accepted = child_accepted
            if child_accepted and sys.version_info < (3, 11):
                _PY310_CANCELLING[child] = _PY310_CANCELLING.get(child, 0) + 1
        return accepted
    finally:
        _ACTIVE_CANCEL_QUEUE = None


class _UnchainedWaiter(asyncio.Future[None]):
    """Leaf Future that forwards cancellation iteratively without chaining ``Task._fut_waiter``."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        caller: asyncio.Task[Any] | None,
        child: asyncio.Task[Any],
    ) -> None:
        super().__init__(loop=loop)
        self._caller = caller
        self.child_task = child
        self.forwarded_to_child = False
        self.child_cancel_accepted = False

    @override
    def cancel(self, msg: Any = None) -> bool:  # noqa: ANN401
        if not super().cancel(msg):
            return False
        self.forwarded_to_child = True
        if _ACTIVE_CANCEL_QUEUE is not None:
            _ACTIVE_CANCEL_QUEUE.append((self, msg))
            return True
        if sys.version_info < (3, 11) and self._caller is not None:
            _PY310_CANCELLING[self._caller] = _PY310_CANCELLING.get(self._caller, 0) + 1
        self.child_cancel_accepted = _cancel_task_unchained(self.child_task, msg)
        return True


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


async def await_unchained(
    task: asyncio.Task[_T],
    *,
    on_cancel: Callable[[asyncio.CancelledError], None] | None = None,
) -> _T:
    """Await ``task`` without chaining ``Task._fut_waiter`` across tasks.

    Directly running ``await task`` sets ``parent._fut_waiter = task``. Across
    N nested task hops (e.g. 400 middleware hooks in ``hop()``), ``_fut_waiter``
    forms an N-deep ``Task -> Task`` linked list. Calling ``outer.cancel()`` or
    ``repr(outer)`` walks that chain recursively on the call stack and raises
    ``RecursionError`` on Python 3.10 and 3.11.

    Awaiting an ``_UnchainedWaiter`` leaf ``Future`` per step breaks that link:
    1. ``parent._fut_waiter`` points to a leaf ``Future`` (O(1) stack depth for
       both ``cancel()`` and ``repr()``).
    2. Calling ``outer.cancel()`` forwards cancellation synchronously down
       nested ``_UnchainedWaiter``s via an iterative trampoline in O(1) stack
       frames, then keeps waiting on fresh leaf ``Future``s until ``task``
       finishes cleanup, matching standard ``await task`` behavior.
    3. Once ``task`` finishes, ``task.result()`` returns its value or re-raises
       its exception / ``CancelledError``.
    """
    loop = asyncio.get_running_loop()
    caller = asyncio.current_task()

    while not task.done():
        # 1. Suspend on a leaf Future so `parent._fut_waiter` never points to `task`.
        waiter = _UnchainedWaiter(loop=loop, caller=caller, child=task)
        on_done = functools.partial(_wake_waiter, waiter)
        task.add_done_callback(on_done)
        try:
            await waiter
        except asyncio.CancelledError as err:
            if on_cancel is not None:
                on_cancel(err)
            # 2. If `waiter.cancel()` did not already forward cancellation via
            #    the iterative trampoline (e.g. `waiter` resolved in the same
            #    tick before `caller.cancel()` ran), forward it here. If `task`
            #    already finished, retrieve any task error and re-raise.
            if not waiter.forwarded_to_child:
                if sys.version_info < (3, 11) and caller is not None:
                    _PY310_CANCELLING[caller] = _PY310_CANCELLING.get(caller, 0) + 1
                msg: Any = err.args[0] if err.args else None
                if not _cancel_task_unchained(task, msg):
                    if not task.cancelled():
                        _ = task.exception()
                    raise
            elif not waiter.child_cancel_accepted:
                if not task.cancelled():
                    _ = task.exception()
                raise
        finally:
            task.remove_done_callback(on_done)

    # 3. `task` has finished; propagate its return value, exception, or cancellation.
    return task.result()
