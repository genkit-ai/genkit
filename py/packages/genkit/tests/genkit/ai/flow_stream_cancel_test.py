# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`async for` over `my_flow.stream(x)`, and stopping it early cancels the flow."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from genkit import ActionRunContext, Flow, Genkit, GenkitError

T = TypeVar('T')


class SlowFlow:
    """A flow that sends 1, waits for `release`, sends 2, and returns 'done'."""

    def __init__(self, *, release: bool = False, fail_midway: bool = False) -> None:
        self.release = asyncio.Event()
        if release:
            self.release.set()
        self.fail_midway = fail_midway
        self.cancelled = asyncio.Event()

    async def __call__(self, n: int, ctx: ActionRunContext) -> str:
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


def setup(**kwargs: bool) -> tuple[Flow, SlowFlow]:
    ai = Genkit()
    body = SlowFlow(**kwargs)

    @ai.flow()
    async def count(n: int, ctx: ActionRunContext) -> str:
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
    """Touching `s.stream` again returns the same reader, so two `s.stream.__anext__()` calls get 1 then 2."""
    count, body = setup(release=True)

    s = count.stream(1)
    first = s.stream
    assert s.stream is first
    chunks = [await settle(s.stream.__anext__()), await settle(s.stream.__anext__())]

    assert chunks == [1, 2]
    assert await settle(s.response) == 'done'
    assert not body.cancelled.is_set()


@pytest.mark.asyncio
async def test_flow_stream_break_after_first_chunk_cancels_flow_and_response_raises_cancelled() -> None:
    """`break` after the first chunk of `my_flow.stream(x)` cancels the flow; `.response` raises CANCELLED."""
    count, body = setup()

    stream = count.stream(1)
    seen = []
    async for chunk in stream:
        seen.append(chunk)
        break
    with pytest.raises(GenkitError) as err:
        await settle(stream.response)

    assert seen == [1]
    assert body.cancelled.is_set()
    assert err.value.status == 'CANCELLED'


@pytest.mark.asyncio
async def test_flow_stream_consumer_task_cancelled_cancels_flow() -> None:
    """Cancelling the task iterating `my_flow.stream(x)` cancels the flow; `.response` raises CANCELLED."""
    count, body = setup()
    stream = count.stream(1)
    first_chunk = asyncio.Event()

    async def read() -> None:
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
