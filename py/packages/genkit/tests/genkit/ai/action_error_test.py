# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What a caller catches when a flow, tool, or embedder body raises."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from genkit import (
    FinishReason,
    Genkit,
    GenkitError,
    Interrupt,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    PublicError,
    Role,
)
from genkit._ai._generate import generate_action
from genkit._ai._testing import define_programmable_model
from genkit._core._model import GenerateActionOptions
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._typing import ToolRequest
from genkit.embedder import EmbedRequest, EmbedResponse


class AccountLockedError(Exception):
    """A caller's own exception type."""


def _model_calls_tool(*, name: str, ref: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name=name, input={}, ref=ref))]),
    )


def _define_missing(ai: Genkit):  # noqa: ANN202
    @ai.flow()
    async def missing(account: str) -> str:
        raise ValueError('no such account')

    return missing


@pytest.mark.asyncio
async def test_await_flow_that_raises_value_error_raises_value_error() -> None:
    """`await missing('acme')` raises the body's ValueError."""
    missing = _define_missing(Genkit())

    with pytest.raises(ValueError, match='no such account') as exc:
        await missing('acme')

    assert type(exc.value) is ValueError


@pytest.mark.asyncio
async def test_await_flow_that_raises_custom_exception_raises_same_instance() -> None:
    """A user exception class comes back as the same object the body raised."""
    ai = Genkit()
    raised = AccountLockedError('acme is locked')

    @ai.flow()
    async def charge(account: str) -> str:
        raise raised

    with pytest.raises(AccountLockedError) as exc:
        await charge('acme')

    assert exc.value is raised


@pytest.mark.asyncio
async def test_await_flow_that_raises_public_error_keeps_status_and_message() -> None:
    """A PublicError('NOT_FOUND', 'no order 99') surfaces unchanged."""
    ai = Genkit()

    @ai.flow()
    async def order(order_id: str) -> str:
        raise PublicError('NOT_FOUND', f'no order {order_id}')

    with pytest.raises(PublicError) as exc:
        await order('99')

    assert exc.value.status == 'NOT_FOUND'
    assert exc.value.original_message == 'no order 99'


@pytest.mark.asyncio
async def test_await_flow_that_raises_genkit_error_keeps_status() -> None:
    """A GenkitError(status='FAILED_PRECONDITION') surfaces with that status."""
    ai = Genkit()

    @ai.flow()
    async def ship(order_id: str) -> str:
        raise GenkitError(status='FAILED_PRECONDITION', message='order is not paid')

    with pytest.raises(GenkitError) as exc:
        await ship('99')

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert exc.value.original_message == 'order is not paid'
    assert exc.value.cause is None


@pytest.mark.asyncio
async def test_await_outer_flow_calling_failing_inner_flow_raises_inner_value_error() -> None:
    """A ValueError from a nested flow reaches the outer caller unwrapped."""
    ai = Genkit()
    missing = _define_missing(ai)

    @ai.flow()
    async def statement(account: str) -> str:
        return await missing(account)

    with pytest.raises(ValueError, match='no such account') as exc:
        await statement('acme')

    assert type(exc.value) is ValueError


@pytest.mark.asyncio
async def test_await_tool_that_raises_value_error_raises_value_error() -> None:
    """Calling a tool directly raises the tool's ValueError."""
    ai = Genkit()

    @ai.tool()
    async def balance(account: str) -> int:
        """Look up an account balance."""
        raise ValueError(f'no account {account}')

    with pytest.raises(ValueError, match='no account acme') as exc:
        await balance('acme')

    assert type(exc.value) is ValueError


@pytest.mark.asyncio
async def test_ai_embed_with_failing_embedder_raises_original_exception() -> None:
    """`ai.embed` raises the embedder's own exception."""
    ai = Genkit()

    async def down(request: EmbedRequest) -> EmbedResponse:
        raise ConnectionError('embedding server refused the connection')

    ai.define_embedder(name='down', fn=down)

    with pytest.raises(ConnectionError, match='refused the connection') as exc:
        await ai.embed(embedder='down', content='hello')

    assert type(exc.value) is ConnectionError


@pytest.mark.asyncio
async def test_flow_error_with_tracing_on_has_no_trace_id(hex_ids: None) -> None:
    """With tracing on, the raised ValueError still has no `trace_id`; the trace keeps it."""
    missing = _define_missing(Genkit())

    with pytest.raises(ValueError) as exc:
        await missing('acme')

    assert not hasattr(exc.value, 'trace_id')


@pytest.mark.asyncio
async def test_flow_genkit_error_with_tracing_on_keeps_trace_id_unset(hex_ids: None) -> None:
    """With tracing on, a GenkitError the body raised comes back with `trace_id` still None."""
    ai = Genkit()

    @ai.flow()
    async def ship(order_id: str) -> str:
        raise GenkitError(status='FAILED_PRECONDITION', message='order is not paid')

    with pytest.raises(GenkitError) as exc:
        await ship('99')

    assert exc.value.trace_id is None
    assert exc.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_flow_reraising_shared_exception_leaves_it_untouched(hex_ids: None) -> None:
    """One exception instance raised by two runs comes back unchanged both times."""
    ai = Genkit()
    locked = AccountLockedError('acme is locked')

    @ai.flow()
    async def charge(account: str) -> str:
        raise locked

    for _ in range(2):
        with pytest.raises(AccountLockedError) as exc:
            await charge('acme')
        assert exc.value is locked
        assert 'trace_id' not in vars(locked)


@pytest.mark.asyncio
async def test_flow_error_keeps_existing_trace_id_attribute(hex_ids: None) -> None:
    """An exception that already carries `trace_id` keeps its own value."""
    ai = Genkit()

    class UpstreamError(Exception):
        trace_id = 'upstream-trace'

    @ai.flow()
    async def relay(account: str) -> str:
        raise UpstreamError('upstream failed')

    with pytest.raises(UpstreamError) as exc:
        await relay('acme')

    assert exc.value.trace_id == 'upstream-trace'


@pytest.mark.asyncio
async def test_dev_ui_run_of_failing_flow_returns_message_and_trace_id(hex_ids: None) -> None:
    """A Dev UI runAction of `missing` returns the body's message, code INTERNAL, and the run's trace id."""
    ai = Genkit()
    _define_missing(ai)
    app = create_reflection_asgi_app(ai.registry)

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/runAction', json={'key': '/flow/missing', 'input': 'acme'})

    error = response.json()['error']
    assert error['message'] == 'no such account'
    assert error['code'] == 13
    assert error['details']['traceId'] == response.headers['x-genkit-trace-id']
    assert len(error['details']['traceId']) == 32


@pytest.mark.asyncio
async def test_dev_ui_run_of_public_error_flow_returns_its_status_and_trace_id(hex_ids: None) -> None:
    """A Dev UI runAction of a flow raising PublicError NOT_FOUND returns its message, code, and the run's trace id."""
    ai = Genkit()

    @ai.flow()
    async def order(order_id: str) -> str:
        raise PublicError('NOT_FOUND', f'no order {order_id}')

    app = create_reflection_asgi_app(ai.registry)

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        response = await client.post('/api/runAction', json={'key': '/flow/order', 'input': '99'})

    error = response.json()['error']
    assert error['message'] == 'no order 99'
    assert error['code'] == 5
    assert error['details']['traceId'] == response.headers['x-genkit-trace-id']


@pytest.mark.asyncio
async def test_generate_with_failing_tool_still_returns_internal_error() -> None:
    """A tool raising ValueError still fails the response with INTERNAL and `finish_message == 'internal error'`."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise ValueError('db password is hunter2')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='weather?', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.error.message == 'internal error'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_public_error_tool_returns_its_message() -> None:
    """A tool raising PublicError still puts its message on `finish_message`."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        raise PublicError('NOT_FOUND', 'no forecast for Atlantis')

    pm.responses = [_model_calls_tool(name='lookup', ref='r1')]

    response = await ai.generate(prompt='weather?', tools=['lookup'])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'no forecast for Atlantis'
    assert response.error is not None
    assert response.error.status == 'NOT_FOUND'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]


@pytest.mark.asyncio
async def test_generate_with_interrupting_tool_still_returns_interrupted() -> None:
    """A tool raising Interrupt still yields an interrupted response with the tool request."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)

    @ai.tool(name='transfer')
    async def transfer() -> str:
        raise Interrupt({'reason': 'needs_approval'})

    pm.responses = [_model_calls_tool(name='transfer', ref='r1')]

    response = await ai.generate(prompt='send $100', tools=['transfer'])

    assert response.finish_reason == FinishReason.INTERRUPTED
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    [interrupt] = response.interrupts
    assert interrupt.tool_request is not None
    assert interrupt.tool_request.ref == 'r1'
    assert interrupt.metadata is not None
    assert interrupt.metadata['interrupt'] == {'reason': 'needs_approval'}


@pytest.mark.asyncio
async def test_generate_with_failing_streaming_callback_returns_callback_message() -> None:
    """An `on_chunk` that raises still yields its own message on the failed response."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai)
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('partial')])]]

    def on_chunk(_: ModelResponseChunk) -> None:
        raise RuntimeError('model sink closed')

    response = await generate_action(
        ai.registry,
        GenerateActionOptions(
            model='programmableModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        ),
        on_chunk=on_chunk,
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'model sink closed'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER]
