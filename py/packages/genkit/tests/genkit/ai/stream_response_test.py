# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""flow.stream() and ai.generate_stream() return the same StreamResponse shape, and both iterate directly."""

import json
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
    PublicError,
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


@pytest.mark.asyncio
async def test_flow_stream_to_sse_frames_chunks_and_result() -> None:
    """`flow.stream(...).to_sse()` yields `data: {"message": ...}` frames followed by `data: {"result": ...}`."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        ctx.send_chunk('Out for delivery')
        return f'Order {order_id} delivered'

    lines = [line async for line in track_order.stream('A-1042').to_sse()]
    assert lines == [
        'data: {"message":"Packed"}\n\n',
        'data: {"message":"Out for delivery"}\n\n',
        'data: {"result":"Order A-1042 delivered"}\n\n',
    ]


@pytest.mark.asyncio
async def test_flow_stream_to_sse_on_failure_emits_error_frame_without_result() -> None:
    """`flow.stream(...).to_sse()` frames mid-stream errors as `data: {"error": ...}` with no trailing result."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('Packed')
        raise PublicError('NOT_FOUND', f'order {order_id} missing')

    lines = [line async for line in track_order.stream('A-1042').to_sse()]
    assert len(lines) == 2
    assert lines[0] == 'data: {"message":"Packed"}\n\n'
    payload = json.loads(lines[1].removeprefix('data: ').strip())
    assert payload == {'error': {'message': 'order A-1042 missing', 'status': 'NOT_FOUND'}}


@pytest.mark.asyncio
async def test_generate_stream_to_sse_frames_model_chunks_and_response() -> None:
    """`ai.generate_stream(...).to_sse()` serializes Pydantic chunks and final response with camelCase aliases."""
    ai = Genkit()
    define_scripted_model(
        ai,
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('Miso soup')]),
            )
        ],
        chunks=[
            [
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Miso')]),
                ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(' soup')]),
            ]
        ],
    )

    lines = [line async for line in ai.generate_stream(model='scriptedModel', prompt='Soup?').to_sse()]
    events = [json.loads(line.removeprefix('data: ').strip()) for line in lines]
    assert [list(e.keys()) for e in events] == [['message'], ['message'], ['result']]
    assert events[0]['message']['content'] == [{'text': 'Miso'}]
    assert events[1]['message']['content'] == [{'text': ' soup'}]
    assert events[2]['result']['finishReason'] == 'stop'
