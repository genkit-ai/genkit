# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for asyncio compatibility helpers in ``genkit._core._compat``."""

import asyncio
import gc
import sys

import pytest

from genkit import Message, Part, Role
from genkit._ai._generate import stamp_cancelled_error
from genkit._core._compat import await_unchained, cancelled_history


@pytest.mark.asyncio
async def test_await_unchained_returns_task_result() -> None:
    """Normal completion returns the child task's value."""

    async def compute() -> str:
        await asyncio.sleep(0)
        return 'smoked-salmon-tartine'

    assert await await_unchained(asyncio.create_task(compute())) == 'smoked-salmon-tartine'


@pytest.mark.asyncio
async def test_await_unchained_propagates_task_exception() -> None:
    """An unhandled exception inside the child task re-raises in the caller."""

    async def fail() -> None:
        await asyncio.sleep(0)
        raise ValueError('kitchen closed')

    with pytest.raises(ValueError, match='kitchen closed'):
        await await_unchained(asyncio.create_task(fail()))


@pytest.mark.asyncio
async def test_await_unchained_waits_for_child_cleanup_on_cancel() -> None:
    """Cancelling the caller cancels the child and waits for its finally block."""
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def child() -> None:
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished.set()

    outer = asyncio.create_task(await_unchained(asyncio.create_task(child())))
    await started.wait()
    outer.cancel('stop-order')

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await outer

    assert cleanup_finished.is_set()
    if sys.version_info >= (3, 11):
        assert exc_info.value.args == ('stop-order',)


@pytest.mark.asyncio
async def test_await_unchained_returns_value_if_child_absorbs_cancel() -> None:
    """If the child catches CancelledError and returns, await_unchained returns that value."""
    started = asyncio.Event()

    async def child() -> str:
        try:
            started.set()
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return 'fallback-menu'
        return 'unreachable'

    outer = asyncio.create_task(await_unchained(asyncio.create_task(child())))
    await started.wait()
    outer.cancel()

    assert await outer == 'fallback-menu'


@pytest.mark.asyncio
async def test_await_unchained_reraises_late_caller_cancel_after_child_finishes() -> None:
    """A cancel arriving after the child already finished still cancels the caller."""
    finish_child = asyncio.Event()
    unretrieved_errors: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: unretrieved_errors.append(ctx))

    async def child() -> None:
        await finish_child.wait()
        raise RuntimeError('child boom')

    try:
        outer = asyncio.create_task(await_unchained(asyncio.create_task(child())))
        await asyncio.sleep(0.01)
        finish_child.set()
        await asyncio.sleep(0)  # Child finishes here and schedules waiter wakeup.
        outer.cancel('caller-cancelled-late')

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await outer

        if sys.version_info >= (3, 11):
            assert exc_info.value.args == ('caller-cancelled-late',)

        gc.collect()
        await asyncio.sleep(0)
        assert unretrieved_errors == []
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_await_unchained_500_deep_cancel_and_repr_stay_bounded() -> None:
    """Cancelling and formatting 500 nested tasks succeeds under a 200-frame recursion limit."""
    started = asyncio.Event()

    async def nested(depth: int) -> None:
        await asyncio.sleep(0)
        if depth == 0:
            started.set()
            await asyncio.Event().wait()
            return
        await await_unchained(asyncio.create_task(nested(depth - 1)))

    outer = asyncio.create_task(nested(500))
    await started.wait()

    previous_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    try:
        assert repr(outer)
        outer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await outer
    finally:
        sys.setrecursionlimit(previous_limit)


def _order_history() -> list[Message]:
    return [Message(role=Role.USER, content=[Part.from_text('book dinner for 2')])]


@pytest.mark.asyncio
async def test_cancelled_history_reads_stamp_across_task_hops() -> None:
    """History stamped inside a cancelled task is found after awaiting through two task hops."""
    history = _order_history()
    started = asyncio.Event()

    async def reserve() -> None:
        try:
            started.set()
            await asyncio.sleep(10)
        except asyncio.CancelledError as exc:
            stamp_cancelled_error(exc=exc, messages=history)
            raise

    async def book() -> None:
        await asyncio.create_task(reserve())

    task = asyncio.create_task(book())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    found = cancelled_history(exc_info.value)
    assert found is not None
    assert [m.text for m in found] == ['book dinner for 2']
    assert 'message' not in exc_info.value.__dict__
    # 3.10 hands back a fresh CancelledError; 3.11+ returns the stamped one.
    assert (exc_info.value.__dict__.get('messages') is None) == (sys.version_info < (3, 11))


def test_cancelled_history_returns_none_without_stamp() -> None:
    """Unstamped cancels and unrelated errors have no history."""
    assert cancelled_history(asyncio.CancelledError()) is None
    assert cancelled_history(ValueError('kitchen closed')) is None


def test_cancelled_history_walks_context_only_on_310() -> None:
    """A stamp on __context__ is read on 3.10 only; 3.11+ reads the exception itself."""
    history = _order_history()
    stamped = asyncio.CancelledError()
    stamp_cancelled_error(exc=stamped, messages=history)
    fresh = asyncio.CancelledError()
    fresh.__context__ = stamped

    found = cancelled_history(fresh)
    if sys.version_info < (3, 11):
        assert found is not None
        assert [m.text for m in found] == ['book dinner for 2']
    else:
        assert found is None


def test_cancelled_history_stops_on_context_cycle() -> None:
    """A self-referencing __context__ chain terminates."""
    a = asyncio.CancelledError()
    b = asyncio.CancelledError()
    a.__context__ = b
    b.__context__ = a
    assert cancelled_history(a) is None


@pytest.mark.asyncio
async def test_cancelled_history_reads_stamp_from_wait_for_and_timeout() -> None:
    """wait_for and asyncio.timeout raise TimeoutError with the stamped CancelledError on __cause__."""
    history = _order_history()

    async def reserve() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError as exc:
            stamp_cancelled_error(exc=exc, messages=history)
            raise

    with pytest.raises((TimeoutError, asyncio.TimeoutError)) as wait_for_exc:
        await asyncio.wait_for(reserve(), timeout=0.01)

    found = cancelled_history(wait_for_exc.value)
    assert found is not None
    assert [m.text for m in found] == ['book dinner for 2']

    if sys.version_info >= (3, 11):
        with pytest.raises(TimeoutError) as timeout_exc:
            async with asyncio.timeout(0.01):
                await reserve()

        found_timeout = cancelled_history(timeout_exc.value)
        assert found_timeout is not None
        assert [m.text for m in found_timeout] == ['book dinner for 2']

    assert cancelled_history(TimeoutError('unrelated socket timeout')) is None
