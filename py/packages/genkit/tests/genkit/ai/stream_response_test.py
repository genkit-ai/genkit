# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""flow.stream() and ai.generate_stream() return the same StreamResponse shape, and both iterate directly."""

import asyncio
import contextlib
import gc
from collections.abc import Iterator
from typing import Any

import pytest
from typing_extensions import assert_type

from genkit import (
    FlowContext,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    ModelStreamResponse,
    Part,
    StreamResponse,
)
from genkit._core._typing import FinishReason, Role
from genkit.testing import define_scripted_model


async def _texts(result: StreamResponse[ModelResponseChunk[Any], ModelResponse[Any]]) -> list[str]:
    """Typed against the base class, so passing a ModelStreamResponse checks the subclassing."""
    return [chunk.text async for chunk in result]


@pytest.mark.asyncio
async def test_async_for_on_flow_stream_yields_chunks_then_response() -> None:
    """`async for update in flow.stream(...)` reads the chunks; `.response` is the return value."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        ctx.send_chunk('Out for delivery')
        return f'Order {order_id} delivered'

    result = track_order.stream('A-1042')
    updates = [update async for update in result]

    assert updates == ['Packed', 'Out for delivery']
    assert await result.response == 'Order A-1042 delivered'


@pytest.mark.asyncio
async def test_flow_stream_that_raises_raises_from_loop_and_response() -> None:
    """If the flow raises, both the `async for` and `await result.response` raise it."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        raise ValueError(f'no courier for {order_id}')

    result = track_order.stream('A-1042')

    with pytest.raises(ValueError, match='no courier for A-1042'):
        async for _ in result:
            pass
    with pytest.raises(ValueError, match='no courier for A-1042'):
        await result.response


@pytest.mark.asyncio
async def test_generate_stream_is_a_stream_response_and_iterates() -> None:
    """`ai.generate_stream()` returns a StreamResponse subclass; iterating it and `.stream` read the same chunks."""
    ai = Genkit()
    define_scripted_model(
        ai,
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('Savory soybean broth with tofu.')]),
            )
        ],
        chunks=[
            [
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Savory soybean broth')]),
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(' with tofu.')]),
            ]
        ],
    )

    result = ai.generate_stream(model='scriptedModel', prompt='Describe miso soup in one line.')
    assert_type(result, ModelStreamResponse[Any])
    assert isinstance(result, StreamResponse)

    texts = await _texts(result)
    response = await result.response

    assert texts == ['Savory soybean broth', ' with tofu.']
    assert response.text == 'Savory soybean broth with tofu.'
    assert response.finish_reason == FinishReason.STOP


@contextlib.contextmanager
def _unretrieved_future_errors() -> Iterator[list[str]]:
    """Collect asyncio 'never retrieved' reports raised while the block runs, after a GC pass."""
    loop = asyncio.get_running_loop()
    seen: list[str] = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, ctx: seen.append(str(ctx.get('message'))))
    try:
        yield seen
        gc.collect()
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_flow_stream_failure_caught_in_loop_leaves_no_unretrieved_future() -> None:
    """Catching the error from `async for` alone doesn't log 'Future exception was never retrieved'."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        raise ValueError(f'no courier for {order_id}')

    with _unretrieved_future_errors() as seen:
        with pytest.raises(ValueError, match='no courier'):
            async for _ in track_order.stream('A-1042'):
                pass
        await asyncio.sleep(0)

    assert seen == []


@pytest.mark.asyncio
async def test_generate_stream_unknown_model_leaves_no_unretrieved_future() -> None:
    """An unknown model raises NOT_FOUND from the loop and `.response`, and nothing is left unretrieved."""
    ai = Genkit()

    statuses: list[str] = []

    async def consume() -> None:
        # Kept in its own frame, and only the status strings leave it: a held
        # exception's traceback would keep the stream alive past gc.collect().
        result = ai.generate_stream(model='nope', prompt='Suggest a dish.')
        try:
            async for _ in result:
                pass
        except GenkitError as e:
            statuses.append(e.status)
        try:
            await result.response
        except GenkitError as e:
            statuses.append(e.status)

    with _unretrieved_future_errors() as seen:
        await consume()
        await asyncio.sleep(0)

    assert statuses == ['NOT_FOUND', 'NOT_FOUND']
    assert seen == []
