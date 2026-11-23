# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`async for` over `my_flow.stream(x)`, and stopping it early cancels the flow."""

import asyncio
import contextlib
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from genkit import FlowContext, Genkit, GenkitError, Interrupt
from genkit._core._action import Action

T = TypeVar('T')


class SlowFlow:
    """A flow that sends 1, waits for `release`, sends 2, and returns 'done'."""

    def __init__(self, *, release: bool = False, fail_midway: bool = False) -> None:
        self.release = asyncio.Event()
        if release:
            self.release.set()
        self.fail_midway = fail_midway
        self.cancelled = asyncio.Event()

    async def __call__(self, n: int, ctx: FlowContext) -> str:
        ctx.send_chunk(1)
        if self.fail_midway:
            raise ValueError('flow blew up')
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        ctx.send_chunk(2)
        return 'done'


def setup(**kwargs: bool) -> tuple[Action, SlowFlow]:
    ai = Genkit()
    body = SlowFlow(**kwargs)

    @ai.flow()
    async def count(n: int, ctx: FlowContext) -> str:
        return await body(n, ctx)

    return count, body


async def settle(aw: Awaitable[T]) -> T:
    # guards against a hang turning into a stuck test run; never the thing that passes a test.
    return await asyncio.wait_for(aw, timeout=5)


@pytest.mark.asyncio
async def test_async_for_over_flow_stream_yields_chunks_directly() -> None:
    """`async for c in my_flow.stream(x)` yields the same chunks as `async for c in my_flow.stream(x).stream`."""
    count, _ = setup(release=True)

    direct = [chunk async for chunk in count.stream(1)]
    via_stream = [chunk async for chunk in count.stream(1).stream]

    assert direct == [1, 2]
    assert direct == via_stream


@pytest.mark.asyncio
async def test_flow_stream_accessing_stream_twice_returns_the_same_reader() -> None:
    """`aiter(s)` and `s.stream` return the same `AsyncIterator`, so two `anext()` calls get 1 then 2."""
    count, body = setup(release=True)

    s = count.stream(1)
    first = s.stream
    assert s.stream is first
    assert aiter(s) is first
    assert aiter(s.stream) is first
    c1 = await settle(anext(aiter(s)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    c2 = await settle(anext(s.stream))
    chunks = [c1, c2]

    assert chunks == [1, 2]
    assert await settle(s.response) == 'done'
    assert not body.cancelled.is_set()


@pytest.mark.asyncio
async def test_flow_stream_async_with_break_cancels_flow_and_response_raises_cancelled() -> None:
    """Exiting `async with my_flow.stream(x):` early cancels the flow; `.response` raises CANCELLED."""
    count, body = setup()

    seen = []
    async with count.stream(1) as stream:
        async for chunk in stream:
            seen.append(chunk)
            break
    with pytest.raises(GenkitError) as err:
        await settle(stream.response)

    assert seen == [1]
    assert body.cancelled.is_set()
    assert err.value.status == 'CANCELLED'


@pytest.mark.asyncio
async def test_flow_stream_bare_break_without_async_with_keeps_flow_running() -> None:
    """Bare `break` without `async with` or `aclose()` leaves the background flow running to completion."""
    count, body = setup(release=True)

    stream = count.stream(1)
    seen = []
    async for chunk in stream:
        seen.append(chunk)
        break
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert seen == [1]
    assert await settle(stream.response) == 'done'
    assert not body.cancelled.is_set()


@pytest.mark.asyncio
async def test_flow_stream_consumer_task_cancelled_under_async_with_cancels_flow() -> None:
    """Cancelling the task reading `async with my_flow.stream(x):` cancels the flow; `.response` raises CANCELLED."""
    count, body = setup()
    stream = count.stream(1)
    first_chunk = asyncio.Event()

    async def read() -> None:
        async with stream:
            async for _ in stream:
                first_chunk.set()

    reader = asyncio.create_task(read())
    await settle(first_chunk.wait())
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    with pytest.raises(GenkitError) as err:
        await settle(stream.response)

    assert body.cancelled.is_set()
    assert err.value.status == 'CANCELLED'


@pytest.mark.asyncio
async def test_flow_stream_iterated_to_end_returns_output() -> None:
    """Reading `my_flow.stream(x)` to the end yields every chunk; `.response` is the flow's return value."""
    count, body = setup(release=True)

    stream = count.stream(1)
    seen = [chunk async for chunk in stream]

    assert seen == [1, 2]
    assert await settle(stream.response) == 'done'
    assert not body.cancelled.is_set()


@pytest.mark.asyncio
async def test_flow_stream_flow_fails_midway_raises_from_loop_and_response() -> None:
    """A flow that raises after one chunk raises from the `async for` and from `.response`."""
    count, _ = setup(fail_midway=True)

    stream = count.stream(1)
    seen = []
    with pytest.raises(ValueError, match='flow blew up'):
        async for chunk in stream:
            seen.append(chunk)
    with pytest.raises(ValueError, match='flow blew up'):
        await settle(stream.response)

    assert seen == [1]


@pytest.mark.asyncio
async def test_flow_stream_await_response_without_iterating_runs_to_completion() -> None:
    """Awaiting `.response` of `my_flow.stream(x)` without reading chunks returns the flow's return value."""
    count, body = setup(release=True)

    assert await settle(count.stream(1).response) == 'done'
    assert not body.cancelled.is_set()


@pytest.mark.asyncio
async def test_flow_stream_consumer_exception_under_async_with_cancels_flow() -> None:
    """An unhandled exception inside `async with my_flow.stream(x):` closes the stream and cancels the flow."""
    count, body = setup()
    stream = count.stream(1)

    with pytest.raises(RuntimeError, match='consumer crash'):
        async with stream:
            async for _ in stream:
                raise RuntimeError('consumer crash')

    with pytest.raises(GenkitError) as err:
        await settle(stream.response)

    assert body.cancelled.is_set()
    assert err.value.status == 'CANCELLED'


@pytest.mark.asyncio
async def test_flow_stream_timeout_under_async_with_cancels_flow() -> None:
    """A `wait_for` deadline on a task inside `async with my_flow.stream(x):` cancels the flow."""
    count, body = setup()
    stream = count.stream(1)

    async def read_slowly() -> None:
        async with stream:
            async for _ in stream:
                await asyncio.sleep(1)

    # asyncio.TimeoutError, not TimeoutError: they're only the same class on 3.11+.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(read_slowly(), timeout=0.02)

    with pytest.raises(GenkitError) as err:
        await settle(stream.response)

    assert body.cancelled.is_set()
    assert err.value.status == 'CANCELLED'


@pytest.mark.asyncio
async def test_flow_stream_interrupt_outside_tool_raises_internal_not_cancelled() -> None:
    """An Interrupt from a streamed flow fails the loop and `.response` with INTERNAL, not CANCELLED.

    Interrupt and CancelledError are both BaseExceptions; the stream must not
    read one as the reader stopping.
    """
    ai = Genkit()

    @ai.flow()
    async def refund(order_id: str, ctx: FlowContext) -> str:
        ctx.send_chunk(1)
        raise Interrupt({'reason': 'needs approval', 'order_id': order_id})

    stream = refund.stream('A-17')
    seen = []
    with pytest.raises(GenkitError) as loop_err:
        async for chunk in stream:
            seen.append(chunk)
    with pytest.raises(GenkitError) as response_err:
        await settle(stream.response)

    assert seen == [1]
    assert loop_err.value.status == 'INTERNAL'
    assert response_err.value.status == 'INTERNAL'
    assert response_err.value.message == 'Interrupt raised outside a tool'


@pytest.mark.asyncio
async def test_flow_stream_anext_timeout_does_not_cancel_flow() -> None:
    """Cancelling a single `stream.stream.__anext__()` (SSE heartbeat tick) leaves the flow running."""
    count, body = setup()
    stream = count.stream(1)

    assert await settle(stream.stream.__anext__()) == 1
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.stream.__anext__(), timeout=0.01)

    assert not body.cancelled.is_set()
    body.release.set()
    assert await settle(stream.stream.__anext__()) == 2
    assert await settle(stream.response) == 'done'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['aclosing', 'async_with', 'async_with_stream'])
async def test_flow_stream_aclosing_and_async_with_preserve_awaiting_finally_cleanup(mode: str) -> None:
    """Breaking under `async with` (on the handle or `.stream`) or `aclosing` lets an awaiting `finally:` finish."""
    ai = Genkit()
    cleanup_finished = False

    @ai.flow()
    async def export_invoice(invoice_id: str, ctx: FlowContext) -> str:
        nonlocal cleanup_finished
        try:
            ctx.send_chunk(f'exporting:{invoice_id}')
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished = True
        return 'never'

    stream = export_invoice.stream('INV-42')
    if mode == 'aclosing':
        async with contextlib.aclosing(stream.stream) as chunks:
            async for _ in chunks:
                break
    elif mode == 'async_with_stream':
        async with stream.stream as chunks:
            async for _ in chunks:
                break
    else:
        async with stream as res:
            async for _ in res.stream:
                break

    assert cleanup_finished is True
    with pytest.raises(GenkitError) as err:
        await settle(stream.response)
    assert err.value.status == 'CANCELLED'
