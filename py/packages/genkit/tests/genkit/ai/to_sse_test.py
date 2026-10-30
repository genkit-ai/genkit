# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`StreamResponse.to_sse()` and `AgentTurn.to_sse()` share one framer; these pin its frames."""

import asyncio
import datetime
import gc
import json
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Document, FlowContext, Genkit, Message, ModelResponse, Part, PublicError
from genkit._ai._agents._client import AgentChunk, AgentResponse, AgentTurn
from genkit._core._model import AgentOutput, AgentStreamChunk, ModelResponseChunk
from genkit._core._typing import AgentFinishReason, FinishReason, Role, TurnEnd
from genkit.testing import define_scripted_model


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

    assert await _lines(fetch_doc.stream('T').to_sse()) == [
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

    assert await _lines(bookings.stream('Ada').to_sse()) == [
        'data: {"result":[{"guest":"Ada","at":"2026-03-01T19:30:00"}]}\n\n',
    ]


@pytest.mark.asyncio
async def test_flow_to_sse_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """The error frame is logged like the adapters log it: 4xx as a warning, without a traceback."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str) -> str:
        raise PublicError('NOT_FOUND', f'order {order_id} missing')

    with caplog.at_level(logging.WARNING, logger='genkit._core._action'):
        lines = await _lines(track_order.stream('A-1042').to_sse())

    assert lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ('WARNING', 'served flow stream failed: NOT_FOUND: order A-1042 missing'),
    ]


def _define_chef(ai: Genkit) -> None:
    reply = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text('Pad thai, no peanuts.')]),
    )
    define_scripted_model(
        ai,
        responses=[reply],
        chunks=[[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Pad thai')])]],
    )


@pytest.mark.asyncio
async def test_generate_stream_to_sse_drops_request_from_result() -> None:
    """The result frame leaves out the echoed request (system prompt, docs); `.response` keeps it."""
    ai = Genkit()
    _define_chef(ai)

    stream = ai.generate_stream(
        model='scriptedModel',
        system='Comp VIPs up to $50.',
        prompt='Dinner?',
        docs=[Document.from_text('Supplier contract: peanut oil, $2.10/L')],
    )
    lines = await _lines(stream.to_sse())

    events = [json.loads(line.removeprefix('data: ')) for line in lines]
    assert [next(iter(e)) for e in events] == ['message', 'result']
    assert 'request' not in events[1]['result']
    assert events[1]['result']['message'] == {'role': 'model', 'content': [{'text': 'Pad thai, no peanuts.'}]}
    assert 'Comp VIPs' not in lines[1]
    assert 'Supplier contract' not in lines[1]
    response = await stream.response
    assert response.request is not None
    assert response.request.docs is not None


@pytest.mark.asyncio
async def test_flow_to_sse_sends_the_flow_result_as_returned() -> None:
    """A flow's result goes out as the flow returned it, same as serve_flow's JSON `result`."""
    ai = Genkit()
    _define_chef(ai)

    @ai.flow()
    async def ask_chef(prompt: str) -> ModelResponse:
        return await ai.generate(model='scriptedModel', system='Comp VIPs up to $50.', prompt=prompt)

    lines = await _lines(ask_chef.stream('Dinner?').to_sse())

    assert 'request' in json.loads(lines[-1].removeprefix('data: '))['result']


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
    """`AgentTurn.to_sse()` frames `raw` (AgentStreamChunk / AgentOutput), the shapes serve_agent sends."""
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

    assert await _lines(turn.to_sse()) == [
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
        lines = await _lines(failed.to_sse())

        unread = _turn([], loop.create_future())
        _ = unread.to_sse()

        del failed, unread, _
        gc.collect()

    assert lines == ['data: {"error":{"message":"agent offline","status":"UNAVAILABLE"}}\n\n']
    assert _unawaited_coroutine_warnings(caught) == []
