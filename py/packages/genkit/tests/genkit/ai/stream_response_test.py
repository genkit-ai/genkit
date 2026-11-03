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
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.model import ModelContext, ModelRequest
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


@pytest.mark.asyncio
async def test_generate_stream_first_turn_failure_ends_loop_and_resolves_failed() -> None:
    """A model that fails on turn 1 before any chunk ends the loop normally; `.response` is FAILED with error set."""
    ai = Genkit()

    async def broken(_request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        raise RuntimeError('kitchen closed')

    ai.define_model(name='broken', fn=broken)

    result = ai.generate_stream(model='broken', prompt='Suggest a dish.')
    chunks = [chunk async for chunk in result]
    response = await result.response

    assert chunks == []
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None


@pytest.mark.asyncio
async def test_generate_stream_tool_failure_resolves_failed() -> None:
    """A tool that raises ends the loop normally; `.response` is FAILED with error set."""
    ai = Genkit()

    @ai.tool()
    async def check_pantry(dish: str) -> str:
        raise RuntimeError(f'pantry offline for {dish}')

    define_scripted_model(
        ai,
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='check_pantry', input='soup', ref='r1'))],
                ),
            ),
        ],
    )

    result = ai.generate_stream(model='scriptedModel', prompt='Can you make soup?', tools=['check_pantry'])
    _ = [chunk async for chunk in result]
    response = await result.response

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None


@pytest.mark.asyncio
async def test_stream_response_chunks_are_single_pass_and_response_rereads() -> None:
    """A second `async for` yields nothing; `.response` awaits any number of times, before or after iterating."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        ctx.send_chunk('Out for delivery')
        return f'Order {order_id} delivered'

    result = track_order.stream('A-1042')
    before = await result.response
    first = [update async for update in result]
    second = [update async for update in result]

    assert before == 'Order A-1042 delivered'
    assert first == ['Packed', 'Out for delivery']
    assert second == []
    assert await result.response == 'Order A-1042 delivered'
    assert await result.response == 'Order A-1042 delivered'


@pytest.mark.asyncio
async def test_failed_stream_second_loop_reraises() -> None:
    """After a failed run, a second `async for` raises the same error again."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str) -> str:
        raise ValueError(f'no courier for {order_id}')

    result = track_order.stream('A-1042')
    for _ in range(2):
        with pytest.raises(ValueError, match='no courier'):
            async for _ in result:
                pass


@pytest.mark.asyncio
async def test_model_action_stream_returns_stream_response() -> None:
    """`Action.stream` returns a StreamResponse for non-flow kinds too, e.g. a model action."""
    ai = Genkit()

    async def chef(_request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Tartine')]))
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Tartine')]))

    action = ai.define_model(name='chef', fn=chef)
    result = action.stream(ModelRequest(messages=[]))

    assert isinstance(result, StreamResponse)
    assert [chunk.text async for chunk in result] == ['Tartine']
    assert (await result.response).text == 'Tartine'


@pytest.mark.asyncio
async def test_generate_stream_cancelling_shielded_waiter_does_not_cancel_run() -> None:
    """Cancelling a shielded waiter on stream.response does not cancel the underlying stream."""
    ai = Genkit()
    release = asyncio.Event()

    async def slow_model(_request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Simmering')]))
        await release.wait()
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Simmering done')]),
        )

    ai.define_model(name='slow', fn=slow_model)

    async with ai.generate_stream(model='slow', prompt='Cook') as stream:
        waiter = asyncio.ensure_future(asyncio.shield(stream.response))
        await asyncio.sleep(0.01)

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        release.set()
        res = await stream.response
        assert res.text == 'Simmering done'
