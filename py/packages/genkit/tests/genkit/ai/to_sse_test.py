# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`genkit.web.to_sse` frames flow streams, model streams and agent turns; these pin its frames."""

import asyncio
import datetime
import gc
import logging
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

import genkit.web._sse as _sse_mod
from genkit import FlowContext, Genkit, Interrupt, Message, ModelResponse, Part, PublicError
from genkit._ai._agents._client import AgentChunk, AgentResponse, AgentTurn
from genkit._core._model import AgentOutput, AgentStreamChunk, ModelResponseChunk, ToolRequest
from genkit._core._typing import AgentFinishReason, FinishReason, Role, TurnEnd
from genkit.model import ModelContext, ModelRequest
from genkit.plugin_api import provider_error
from genkit.testing import define_scripted_model
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

    with caplog.at_level(logging.WARNING, logger='genkit.web'):
        lines = await _lines(to_sse(track_order.stream('A-1042')))

    assert lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert [(r.levelname, r.getMessage()) for r in caplog.records] == [
        ('WARNING', 'served flow stream failed: NOT_FOUND: order A-1042 missing'),
    ]


@pytest.mark.asyncio
async def test_generate_stream_and_prompt_stream_to_sse_redact_and_log_provider_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed Vertex calls on ai.generate_stream() and prompt.stream() log 5xx and emit Internal Error, not result."""
    ai = Genkit()

    async def vertex_denied(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        raise provider_error(
            RuntimeError(
                'Permission denied on resource project secret-proj-123 '
                'for projects/secret-proj-123/locations/us-central1/publishers/google/models/gemini-2.5-flash'
            ),
            http_status=403,
        )

    ai.define_model(name='vertex_denied', fn=vertex_denied)
    menu_prompt = ai.define_prompt(name='menu_prompt', model='vertex_denied', prompt='Suggest dinner.')

    with caplog.at_level(logging.ERROR, logger='genkit.web'):
        gen_lines = await _lines(to_sse(ai.generate_stream(model='vertex_denied', prompt='Suggest dinner.')))
        prompt_lines = await _lines(to_sse(menu_prompt.stream()))

    expected = ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert gen_lines == expected
    assert prompt_lines == expected
    assert 'secret-proj-123' not in gen_lines[0]
    assert 'projects/' not in gen_lines[0]
    assert [(r.name, r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('genkit.web', 'ERROR', 'served flow stream failed', True),
        ('genkit.web', 'ERROR', 'served flow stream failed', True),
    ]
    for record in caplog.records:
        assert record.exc_info is not None
        assert 'secret-proj-123' in str(record.exc_info[1])


@pytest.mark.asyncio
async def test_generate_stream_to_sse_surfaces_tool_public_error_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool raising PublicError inside ai.generate_stream() emits a 4xx error frame and logs a warning."""
    ai = Genkit()

    @ai.tool()
    async def lookup_order(order_id: str) -> str:
        """Look up an order by ID."""
        raise PublicError('NOT_FOUND', f'order {order_id} missing')

    define_scripted_model(
        ai,
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='lookup_order', input='A-1042'))],
                ),
            )
        ],
    )

    with caplog.at_level(logging.WARNING, logger='genkit.web'):
        lines = await _lines(
            to_sse(ai.generate_stream(model='scriptedModel', prompt='Where is A-1042?', tools=['lookup_order']))
        )

    assert lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert [(r.name, r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('genkit.web', 'WARNING', 'served flow stream failed: NOT_FOUND: order A-1042 missing', False),
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


class _SpyTurn(AgentTurn[Any]):
    """AgentTurn recording every `.response` access."""

    def __init__(self, *, stream: AsyncIterator[AgentChunk[Any]], output: asyncio.Future[AgentResponse[Any]]) -> None:
        super().__init__(stream=stream, output=output)
        self.response_reads = 0

    @property
    def response(self) -> Any:
        self.response_reads += 1
        return super().response


@pytest.mark.asyncio
async def test_agent_turn_to_sse_reads_response_only_after_the_stream_drains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed, abandoned, or never-iterated `to_sse()` never reads `.response` or leaks a coroutine."""
    loop = asyncio.get_running_loop()
    wire_chunk = AgentStreamChunk(model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Hi')]))
    wire_output = AgentOutput(
        snapshot_id='snap-1',
        message=Message(role=Role.MODEL, content=[Part.from_text('Hi')]),
        finish_reason=AgentFinishReason.STOP,
    )

    def spy_turn(
        chunks: list[AgentChunk[Any]],
        output: asyncio.Future[AgentResponse[Any]],
        *,
        fail: bool = False,
    ) -> _SpyTurn:
        async def stream() -> AsyncIterator[AgentChunk[Any]]:
            for chunk in chunks:
                yield chunk
            if fail:
                raise PublicError('UNAVAILABLE', 'agent offline')

        return _SpyTurn(stream=stream(), output=output)

    # Disable _sse._logger so pytest's root LogCaptureHandlers don't pin the
    # exception traceback (and to_sse's locals) past gc.collect().
    monkeypatch.setattr(_sse_mod._logger, 'disabled', True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')

        failed = spy_turn([], loop.create_future(), fail=True)
        lines = await _lines(to_sse(failed))
        failed_reads = failed.response_reads

        abandoned = spy_turn([AgentChunk(text='Hi', raw=wire_chunk)], loop.create_future())
        agen = cast(AsyncGenerator[str, None], to_sse(abandoned))
        first = await agen.__anext__()
        mid_stream_reads = abandoned.response_reads
        await agen.aclose()
        abandoned_reads = abandoned.response_reads

        unread = spy_turn([], loop.create_future())
        unused = to_sse(unread)
        unread_reads = unread.response_reads

        settled: asyncio.Future[AgentResponse[Any]] = loop.create_future()
        settled.set_result(AgentResponse(raw=wire_output, messages=[]))
        drained = spy_turn([AgentChunk(text='Hi', raw=wire_chunk)], settled)
        drained_gen = to_sse(drained)
        _ = await drained_gen.__anext__()
        before_drain_reads = drained.response_reads
        _ = await _lines(drained_gen)
        after_drain_reads = drained.response_reads

        del failed, abandoned, agen, unread, unused, drained, drained_gen
        gc.collect()

    assert _unawaited_coroutine_warnings(caught) == []
    assert (failed_reads, mid_stream_reads, abandoned_reads, unread_reads, before_drain_reads, after_drain_reads) == (
        0,
        0,
        0,
        0,
        0,
        1,
    )
    assert first == 'data: {"message":{"modelChunk":{"role":"model","content":[{"text":"Hi"}]}}}\n\n'
    assert lines == ['data: {"error":{"message":"agent offline","status":"UNAVAILABLE"}}\n\n']


class _RawStreamHandle:
    """Custom StreamHandle raising an arbitrary BaseException from ``.stream``."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def stream(self) -> AsyncIterator[object]:
        async def _gen() -> AsyncIterator[object]:
            if False:
                yield None
            raise self._exc

        return _gen()

    @property
    async def response(self) -> object:
        return None


@pytest.mark.asyncio
async def test_to_sse_frames_escaping_interrupt_as_internal_error_and_propagates_cancel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An escaping ``Interrupt`` (a ``BaseException``) frames as 500 Internal Error; ``CancelledError`` propagates."""
    with caplog.at_level(logging.ERROR, logger='genkit.web'):
        lines = await _lines(to_sse(_RawStreamHandle(Interrupt({'secret_order': 'A-17'}))))

    assert lines == ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert 'A-17' not in lines[0]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('ERROR', 'served flow stream failed', True),
    ]

    with pytest.raises(asyncio.CancelledError):
        await _lines(to_sse(_RawStreamHandle(asyncio.CancelledError())))
