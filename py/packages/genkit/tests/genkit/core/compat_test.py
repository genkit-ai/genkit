# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for asyncio compatibility helpers in ``genkit._core._compat``."""

import asyncio
import gc
import sys

import pytest

from genkit._core._compat import await_unchained


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
