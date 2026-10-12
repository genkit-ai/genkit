# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""flow.stream() and ai.generate_stream() return the same StreamResponse shape, and both iterate directly."""

from typing import Any

import pytest
from typing_extensions import assert_type

from genkit import (
    FlowContext,
    Genkit,
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
