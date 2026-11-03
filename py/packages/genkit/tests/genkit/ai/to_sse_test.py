# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""`genkit.web.to_sse` frames flow streams, model streams and agent turns; these pin its frames."""

import asyncio
import copy
import datetime
import gc
import json
import logging
import pickle  # noqa: S403
import warnings
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import pytest
from pydantic import BaseModel

import genkit.web._sse as _sse_mod
from genkit import (
    FlowContext,
    Genkit,
    Interrupt,
    Message,
    ModelResponse,
    Part,
    PublicError,
    StreamResponse,
    ToolContext,
)
from genkit._ai._agents._client import AgentChunk, AgentResponse, AgentTurn
from genkit._ai._generate import box_dead_turn, box_from_exc
from genkit._core._error import BoxedFailure, Failure, GenkitError
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
    """Named action streams log 'served action %r stream failed'; unnamed handles log 'served stream failed'."""
    ai = Genkit()

    @ai.flow()
    async def track_order(order_id: str) -> str:
        raise PublicError('NOT_FOUND', f'order {order_id} missing')

    @ai.flow()
    async def charge_card(_order_id: str) -> str:
        raise RuntimeError('gateway timeout')

    @ai.tool()
    async def lookup_allergens(dish: str) -> str:
        """Look up allergens for a dish."""
        raise PublicError('INVALID_ARGUMENT', f'unknown dish {dish}')

    unnamed_custom = _RawStreamHandle(PublicError('NOT_FOUND', 'order A-1042 missing'))
    raw_for_sr = _RawStreamHandle(PublicError('NOT_FOUND', 'order A-1042 missing'))
    settled_response: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    settled_response.set_result(None)
    unnamed_sr = StreamResponse(stream=raw_for_sr.stream, response=settled_response)

    with caplog.at_level(logging.WARNING):
        not_found_lines = await _lines(to_sse(track_order.stream('A-1042')))
        internal_lines = await _lines(to_sse(charge_card.stream('A-1042')))
        tool_lines = await _lines(to_sse(lookup_allergens.stream('pad-thai')))
        custom_lines = await _lines(to_sse(unnamed_custom))
        sr_lines = await _lines(to_sse(unnamed_sr))

    assert not_found_lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert internal_lines == ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert tool_lines == ['data: {"error":{"message":"unknown dish pad-thai","status":"INVALID_ARGUMENT"}}\n\n']
    assert custom_lines == not_found_lines
    assert sr_lines == not_found_lines
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('WARNING', "served action 'track_order' stream failed: NOT_FOUND: order A-1042 missing", False),
        ('ERROR', "served action 'charge_card' stream failed", True),
        ('WARNING', "served action 'lookup_allergens' stream failed: INVALID_ARGUMENT: unknown dish pad-thai", False),
        ('WARNING', 'served stream failed: NOT_FOUND: order A-1042 missing', False),
        ('WARNING', 'served stream failed: NOT_FOUND: order A-1042 missing', False),
    ]
    assert [getattr(r, 'genkit_action', None) for r in caplog.records] == [
        'track_order',
        'charge_card',
        'lookup_allergens',
        None,
        None,
    ]


@pytest.mark.asyncio
async def test_generate_stream_and_prompt_stream_to_sse_redact_and_log_provider_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failed Vertex calls keep provider text in-process, while to_sse() logs 5xx and emits Internal Error."""
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

    in_proc = await ai.generate_stream(model='vertex_denied', prompt='Suggest dinner.').response
    assert in_proc.finish_reason == FinishReason.FAILED
    assert 'secret-proj-123' in (in_proc.finish_message or '')

    with caplog.at_level(logging.ERROR):
        gen_lines = await _lines(to_sse(ai.generate_stream(model='vertex_denied', prompt='Suggest dinner.')))
        prompt_lines = await _lines(to_sse(menu_prompt.stream()))

    expected = ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert gen_lines == expected
    assert prompt_lines == expected
    assert 'secret-proj-123' not in gen_lines[0]
    assert 'projects/' not in gen_lines[0]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('ERROR', 'served stream failed', True),
        ('ERROR', 'served stream failed', True),
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

    with caplog.at_level(logging.WARNING):
        lines = await _lines(
            to_sse(ai.generate_stream(model='scriptedModel', prompt='Where is A-1042?', tools=['lookup_order']))
        )

    assert lines == ['data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND"}}\n\n']
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('WARNING', 'served stream failed: NOT_FOUND: order A-1042 missing', False),
    ]


class Prep(BaseModel):
    """Kitchen progress a tool streams while it runs."""

    stage: str
    percent: int


@pytest.mark.asyncio
async def test_generate_stream_to_sse_frames_partial_tool_response_chunks() -> None:
    """ToolContext.send_chunk progress goes out as a tool-role message frame marked partial, before the final one."""
    ai = Genkit()

    @ai.tool()
    async def plate_order(order_id: str, ctx: ToolContext[Prep]) -> str:
        """Plate an order, streaming kitchen progress."""
        ctx.send_chunk(Prep(stage='searing', percent=40))
        return f'{order_id} plated'

    define_scripted_model(
        ai,
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='plate_order', ref='call-1', input='T-7'))],
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('Served.')]),
            ),
        ],
    )

    lines = await _lines(to_sse(ai.generate_stream(model='scriptedModel', prompt='Plate T-7.', tools=['plate_order'])))
    # The partial frame carries the tool's chunk as output, tagged partial; the
    # final tool response follows untagged; the result frame closes the stream.
    assert lines[:2] == [
        'data: {"message":{"role":"tool","index":0.0,"content":[{"toolResponse":{"ref":"call-1",'
        '"name":"plate_order","output":{"stage":"searing","percent":40}},"metadata":{"partial":true}}]}}\n\n',
        'data: {"message":{"role":"tool","index":0.0,"content":[{"toolResponse":{"ref":"call-1",'
        '"name":"plate_order","output":"T-7 plated"}}]}}\n\n',
    ]
    assert len(lines) == 3
    result = json.loads(lines[2].removeprefix('data: '))['result']
    assert result['message'] == {'role': 'model', 'content': [{'text': 'Served.'}]}
    assert result['finishReason'] == 'stop'


@pytest.mark.asyncio
async def test_generate_stream_to_sse_undeclared_tool_and_bare_failed_turn_redact_without_result(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """FAILED turns with _failure=None (undeclared tool or bare FAILED) emit Internal Error, not result."""
    ai = Genkit()
    undeclared_turn = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='internal_billing_tool', input={'acct': '42'}))],
        ),
    )
    define_scripted_model(
        ai,
        responses=[
            undeclared_turn,
            undeclared_turn,
            ModelResponse(finish_reason=FinishReason.FAILED),
        ],
    )

    in_proc = await ai.generate_stream(model='scriptedModel', prompt='Check billing.').response
    assert in_proc.finish_reason == FinishReason.FAILED
    assert in_proc._failure is None
    assert in_proc.finish_message == 'Tool internal_billing_tool not found'

    with caplog.at_level(logging.ERROR):
        missing_tool_lines = await _lines(to_sse(ai.generate_stream(model='scriptedModel', prompt='Check billing.')))
        bare_failed_lines = await _lines(to_sse(ai.generate_stream(model='scriptedModel', prompt='Check billing.')))

    expected = ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert missing_tool_lines == expected
    assert bare_failed_lines == expected
    assert 'internal_billing_tool' not in missing_tool_lines[0]
    assert 'not found' not in missing_tool_lines[0]
    assert 'result' not in missing_tool_lines[0]
    assert 'result' not in bare_failed_lines[0]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('ERROR', 'served stream failed', True),
        ('ERROR', 'served stream failed', True),
    ]
    assert caplog.records[0].exc_info is not None
    assert 'Tool internal_billing_tool not found' in str(caplog.records[0].exc_info[1])
    assert caplog.records[1].exc_info is not None
    assert 'Model generation failed' in str(caplog.records[1].exc_info[1])


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


@pytest.mark.asyncio
async def test_to_sse_frames_escaping_interrupt_as_internal_error_and_propagates_cancel(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An escaping ``Interrupt`` (a ``BaseException``) frames as 500 Internal Error; ``CancelledError`` propagates."""
    with caplog.at_level(logging.ERROR):
        lines = await _lines(to_sse(_RawStreamHandle(Interrupt({'secret_order': 'A-17'}))))

    assert lines == ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert 'A-17' not in lines[0]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('ERROR', 'served stream failed', True),
    ]

    with pytest.raises(asyncio.CancelledError):
        await _lines(to_sse(_RawStreamHandle(asyncio.CancelledError())))


@pytest.mark.asyncio
async def test_prompt_stream_to_sse_frames_chunks_and_result() -> None:
    """`to_sse(prompt.stream())` emits `message` frames per chunk and a final `result` frame."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.chunks = [
        [
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Smoked ')]),
            ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Salmon')]),
        ]
    ]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon')]),
        )
    ]
    menu_prompt = ai.define_prompt(name='lunch_prompt', model='scriptedModel', prompt='Suggest lunch.')

    lines = await _lines(to_sse(menu_prompt.stream()))

    assert len(lines) == 3
    frames = [json.loads(line.removeprefix('data: ').strip()) for line in lines]
    assert frames[0]['message']['content'] == [{'text': 'Smoked '}]
    assert frames[1]['message']['content'] == [{'text': 'Salmon'}]
    assert frames[2]['result']['finishReason'] == 'stop'
    assert frames[2]['result']['message']['content'] == [{'text': 'Smoked Salmon'}]


@pytest.mark.asyncio
async def test_generate_stream_to_sse_public_error_details_keep_failure_immutable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PublicError details survive SSE framing without mutating BoxedFailure."""
    ai = Genkit()

    @ai.tool()
    async def lookup_order(order_id: str) -> str:
        """Look up an order by ID."""
        raise PublicError(
            'NOT_FOUND',
            f'order {order_id} missing',
            details={'order': Meta(source='crm'), 'stack': 'internal'},
        )

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

    handle = ai.generate_stream(model='scriptedModel', prompt='Where is A-1042?', tools=['lookup_order'])
    with caplog.at_level(logging.WARNING):
        lines = await _lines(to_sse(handle))

    result = await handle.response
    assert lines == [
        'data: {"error":{"message":"order A-1042 missing","status":"NOT_FOUND",'
        '"details":{"order":{"source":"crm"}}}}\n\n'
    ]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('WARNING', 'served stream failed: NOT_FOUND: order A-1042 missing', False),
    ]
    assert result._failure is not None
    assert result._failure.details == {'order': {'source': 'crm'}}
    assert result._failure == copy.deepcopy(result._failure)


@pytest.mark.asyncio
async def test_box_dead_turn_clears_stale_boxed_failure_from_copied_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dead turns (undeclared tool or caller abort) clear any stale `_failure` inherited via `model_copy()`."""
    stale = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='internal_billing_tool', input={'acct': '42'}))],
        ),
    )
    stale._failure = BoxedFailure.from_exc(PublicError('NOT_FOUND', 'stale public error'))

    dead = box_dead_turn(
        response=stale,
        messages=[],
        finish_reason=FinishReason.FAILED,
        finish_message='Tool internal_billing_tool not found',
        error=Failure(status='NOT_FOUND', message='Tool internal_billing_tool not found'),
    )
    assert dead._failure is None

    aborted = box_from_exc(
        response=stale,
        messages=[],
        exc=RuntimeError('cancelled'),
        caller_stopped=True,
    )
    assert aborted.finish_reason == FinishReason.ABORTED
    assert aborted._failure is None

    ai = Genkit()

    async def model_returning_copied_stale_failure(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return stale.model_copy()

    ai.define_model(name='copied_stale', fn=model_returning_copied_stale_failure)

    with caplog.at_level(logging.WARNING):
        lines = await _lines(to_sse(ai.generate_stream(model='copied_stale', prompt='Check billing.')))

    assert lines == ['data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n']
    assert 'stale public error' not in lines[0]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('ERROR', 'served stream failed', True),
    ]


def test_boxed_failure_round_trips_source_and_cause_without_holding_the_exception() -> None:
    """GenkitError's source prefix and cause text survive BoxedFailure; nothing live is kept."""

    cause = GenkitError(status='NOT_FOUND', message='order A-17')
    err = GenkitError(status='INTERNAL', message='refund failed', source='billing', cause=cause)
    boxed = BoxedFailure.from_exc(err)

    assert boxed == BoxedFailure(
        status='INTERNAL',
        message='refund failed: NOT_FOUND: order A-17',
        source='billing',
    )
    assert str(boxed.to_exception()) == str(err) == 'billing: INTERNAL: refund failed: NOT_FOUND: order A-17'
    assert pickle.loads(pickle.dumps(boxed)) == boxed == copy.deepcopy(boxed)  # noqa: S301

    plain = BoxedFailure.from_exc(GenkitError(status='UNAVAILABLE', message='kitchen offline'))
    assert (plain.source, str(plain.to_exception())) == (None, 'UNAVAILABLE: kitchen offline')

    public = BoxedFailure.from_exc(PublicError('NOT_FOUND', 'no table 12', details={'table': 12}))
    rebuilt = public.to_exception()
    assert (type(rebuilt), rebuilt.status, rebuilt.message, rebuilt.details) == (
        PublicError,
        'NOT_FOUND',
        'no table 12',
        {'table': 12},
    )
