# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`genkit.web.to_sse` frames flow streams, model streams and agent turns; these pin its frames."""

import asyncio
import datetime
import gc
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import FlowContext, Genkit, Message, Part, PublicError
from genkit._ai._agents._client import AgentChunk, AgentResponse, AgentTurn
from genkit._core._model import AgentOutput, AgentStreamChunk, ModelResponseChunk
from genkit._core._typing import AgentFinishReason, Role, TurnEnd
from genkit.web import to_sse


class Meta(BaseModel):
    """Source info a CRM record carries."""

    source: str


class Doc(BaseModel):
    """A user model that happens to have a `raw` field."""

    title: str
    raw: Meta


class Visit(BaseModel):
    """A booking with a datetime field."""

    guest: str
    at: datetime.datetime


async def _lines(stream: AsyncIterator[str]) -> list[str]:
    return [line async for line in stream]


@pytest.mark.asyncio
async def test_flow_to_sse_keeps_user_model_raw_field() -> None:
    """A user model with a `raw` field serializes whole, in chunks and in the result."""
    ai = Genkit()

    @ai.flow()
    async def fetch_doc(doc_id: str, ctx: FlowContext[Doc]) -> Doc:
        ctx.send_chunk(Doc(title='draft', raw=Meta(source='crm')))
        return Doc(title=doc_id, raw=Meta(source='crm'))

    assert await _lines(to_sse(fetch_doc.stream('T'))) == [
        'data: {"message":{"title":"draft","raw":{"source":"crm"}}}\n\n',
        'data: {"result":{"title":"T","raw":{"source":"crm"}}}\n\n',
    ]


@pytest.mark.asyncio
async def test_flow_to_sse_encodes_lists_of_models_and_datetimes() -> None:
    """Values JSON can't hold directly (lists of models, datetimes) are encoded, not an error frame."""
    ai = Genkit()

    @ai.flow()
    async def bookings(guest: str) -> list[Visit]:
        return [Visit(guest=guest, at=datetime.datetime(2026, 3, 1, 19, 30))]

    assert await _lines(to_sse(bookings.stream('Ada'))) == [
        'data: {"result":[{"guest":"Ada","at":"2026-03-01T19:30:00"}]}\n\n',
    ]


@pytest.mark.asyncio
async def test_flow_to_sse_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The error frame is logged like the adapters log it: 4xx as a warning, without a traceback."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str) -> str:
        raise PublicError('NOT_FOUND', f'order {order_id} missing')

    with caplog.at_level(logging.WARNING, logger='genkit.web._sse'):
        lines = await _lines(to_sse(track_order.stream('A-1042')))

    assert lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ('WARNING', 'served flow stream failed: NOT_FOUND: order A-1042 missing'),
    ]


def _turn(
    chunks: list[AgentChunk[Any]],
    output: asyncio.Future[AgentResponse[Any]],
    *,
    fail: bool = False,
) -> AgentTurn[Any]:
    async def stream() -> AsyncIterator[AgentChunk[Any]]:
        for chunk in chunks:
            yield chunk
        if fail:
            raise PublicError('UNAVAILABLE', 'agent offline')

    return AgentTurn(stream=stream(), output=output)


@pytest.mark.asyncio
async def test_agent_turn_to_sse_frames_wire_shapes() -> None:
    """`to_sse(AgentTurn)` frames `raw` (AgentStreamChunk / AgentOutput), the shapes serve_agent sends."""
    wire_chunk = AgentStreamChunk(model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Hi')]))
    turn_end = AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP))
    wire_output = AgentOutput(
        snapshot_id='snap-1',
        message=Message(role=Role.MODEL, content=[Part.from_text('Hi')]),
        finish_reason=AgentFinishReason.STOP,
    )
    output: asyncio.Future[AgentResponse[Any]] = asyncio.get_running_loop().create_future()
    output.set_result(AgentResponse(raw=wire_output, messages=[]))

    turn = _turn([AgentChunk(text='Hi', raw=wire_chunk), AgentChunk(raw=turn_end)], output)

    assert await _lines(to_sse(turn)) == [
        'data: {"message":{"modelChunk":{"role":"model","content":[{"text":"Hi"}]}}}\n\n',
        'data: {"message":{"turnEnd":{"snapshotId":"snap-1","finishReason":"stop"}}}\n\n',
        'data: {"result":{"snapshotId":"snap-1","message":{"role":"model","content":[{"text":"Hi"}]},'
        '"finishReason":"stop"}}\n\n',
    ]


def _unawaited_coroutine_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if 'was never awaited' in str(w.message)]


@pytest.mark.asyncio
async def test_agent_turn_to_sse_reads_response_only_after_the_stream_drains() -> None:
    """A failed or never-iterated `to_sse()` doesn't leave an un-awaited `.response` coroutine."""
    loop = asyncio.get_running_loop()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        failed_output: asyncio.Future[AgentResponse[Any]] = loop.create_future()
        failed = _turn([], failed_output, fail=True)
        lines = await _lines(to_sse(failed))

        unread = _turn([], loop.create_future())
        _ = to_sse(unread)

        del failed, unread, _
        gc.collect()

    assert lines == ['data: {"error":{"message":"agent offline","status":"UNAVAILABLE"}}\n\n']
    assert _unawaited_coroutine_warnings(caught) == []
