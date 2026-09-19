# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""generate() / generate_operation() against a define_background_model fake."""

from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import ActionKind, Document, Genkit, Message, Part
from genkit._core._action import ActionRunContext, _action_context
from genkit._core._error import GenkitError, RuntimeErrorReason
from genkit._core._middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext, ModelHookParams
from genkit._core._model import ModelRequest, ModelResponse, ModelResponseChunk
from genkit._core._typing import (
    Error,
    FinishReason,
    Operation,
    Role,
    ToolRequest,
)


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


def register_bg_model(
    ai: Genkit,
    *,
    op_id: str = 'bg-op-123',
    starts: list[str] | None = None,
    checks: list[str] | None = None,
) -> None:
    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        if starts is not None:
            starts.append('start')
        return Operation(id=op_id, done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        if checks is not None:
            checks.append('check')
        return op

    ai.define_background_model(
        name='bg-model',
        start=start,
        check=check,
    )


@pytest.mark.asyncio
async def test_generate_returns_operation_for_background_model(ai: Genkit) -> None:
    """generate() wraps the start handle. message stays empty."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing')

    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.done is False
    assert response.operation.action == '/background-model/bg-model'
    assert response.message is None


@pytest.mark.asyncio
async def test_generate_operation_with_background_model(ai: Genkit) -> None:
    """generate_operation() returns that same handle."""
    register_bg_model(ai, op_id='bg-op-456')

    operation = await ai.generate_operation(model='bg-model', prompt='a cat surfing')

    assert isinstance(operation, Operation)
    assert operation.id == 'bg-op-456'
    assert operation.action == '/background-model/bg-model'


@pytest.mark.asyncio
async def test_generate_returns_the_job_without_polling(ai: Genkit) -> None:
    """generate() hands back the job now. It does not wait until the job is done.

    A background model (video, and anything registered with
    ``define_background_model``) starts a job and returns a handle. You
    poll later with ``check_operation``. ``generate()`` and
    ``generate_operation()`` only start; they must not call ``check``
    on the way out, or a long render would block the first call.
    """
    checks: list[str] = []
    register_bg_model(ai, checks=checks)

    response = await ai.generate(model='bg-model', prompt='a cat surfing')
    operation = await ai.generate_operation(model='bg-model', prompt='a cat surfing')

    assert response.operation is not None
    assert response.operation.done is False
    assert operation.done is False
    assert checks == []


class ReadsMessage(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        response = await next_fn(params, ctx)
        _ = response.message
        return response


@pytest.mark.asyncio
async def test_generate_boxes_before_wrap_model(ai: Genkit) -> None:
    """wrap_model sees a ModelResponse, so reading .message does not crash."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing', use=[ReadsMessage()])

    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.message is None


class DropsOperation(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp = await next_fn(params, ctx)
        return ModelResponse(message=resp.message, finish_reason=resp.finish_reason)


@pytest.mark.asyncio
async def test_generate_keeps_operation_when_wrap_model_drops_ticket(ai: Genkit) -> None:
    """A hook that drops a billed ticket still leaves the handle to check or cancel."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing', use=[DropsOperation()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return an operation' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'a cat surfing'
    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.error is not None
    assert response.operation.error.message == response.finish_message


@pytest.mark.asyncio
async def test_generate_operation_raises_when_wrap_model_drops_ticket(ai: Genkit) -> None:
    """generate_operation still wants a live ticket, not a dropped handle."""
    register_bg_model(ai)

    with pytest.raises(GenkitError, match='did not return an operation') as operation_info:
        await ai.generate_operation(model='bg-model', prompt='a cat surfing', use=[DropsOperation()])
    assert operation_info.value.status == 'FAILED_PRECONDITION'


class DropsGenerate(BaseMiddleware):
    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        resp = await next_fn(params, ctx)
        return ModelResponse(message=resp.message, finish_reason=resp.finish_reason)


@pytest.mark.asyncio
async def test_generate_keeps_operation_when_wrap_generate_drops_ticket(ai: Genkit) -> None:
    """A wrap_generate that rebuilds without the ticket still leaves the handle."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing', use=[DropsGenerate()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'did not return an operation' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'a cat surfing'
    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.error is not None
    assert response.operation.error.message == response.finish_message


@pytest.mark.asyncio
async def test_generate_operation_raises_when_wrap_generate_drops_ticket(ai: Genkit) -> None:
    """generate_operation still wants a live ticket if wrap_generate dropped it."""
    register_bg_model(ai)

    with pytest.raises(GenkitError, match='did not return an operation') as operation_info:
        await ai.generate_operation(model='bg-model', prompt='a cat surfing', use=[DropsGenerate()])
    assert operation_info.value.status == 'FAILED_PRECONDITION'


class RaisesAfterStart(BaseMiddleware):
    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        await next_fn(params, ctx)
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='hook after start',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )


@pytest.mark.asyncio
async def test_generate_keeps_operation_when_wrap_generate_raises_after_start(ai: Genkit) -> None:
    """A hook that starts then raises still leaves the ticket to check or cancel."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing', use=[RaisesAfterStart()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook after start'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert response.message is None
    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.action == '/background-model/bg-model'
    assert response.operation.error is not None
    assert response.operation.error.message == 'hook after start'


class RaisesAfterStartModel(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        await next_fn(params, ctx)
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='hook after start',
            reason=RuntimeErrorReason.INVALID_INPUT,
        )


@pytest.mark.asyncio
async def test_generate_keeps_operation_when_wrap_model_raises_after_start(ai: Genkit) -> None:
    """A wrap_model that starts then raises still leaves the ticket to check or cancel."""
    register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing', use=[RaisesAfterStartModel()])
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'hook after start'
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.message is None
    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.action == '/background-model/bg-model'
    assert response.operation.error is not None
    assert response.operation.error.message == 'hook after start'


@pytest.mark.asyncio
async def test_generate_operation_failed_start_is_not_a_clean_start(ai: Genkit) -> None:
    """generate_operation does not hand back a live-looking ticket after a failed start."""
    register_bg_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate_operation(model='bg-model', prompt='a cat surfing', use=[RaisesAfterStart()])
    assert raised.value.status == 'FAILED_PRECONDITION'
    assert raised.value.reason is RuntimeErrorReason.INVALID_INPUT
    assert raised.value.original_message == 'hook after start'


class SwallowsStart(BaseMiddleware):
    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        try:
            return await next_fn(params, ctx)
        except GenkitError:
            return ModelResponse(
                message=Message(role=Role.MODEL, content=[Part.from_text('FLASH')]),
                finish_reason=FinishReason.STOP,
            )


@pytest.mark.asyncio
async def test_generate_keeps_fallback_answer_when_start_raises(ai: Genkit) -> None:
    """A hook that substitutes another model's answer is not a missing handle."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        raise GenkitError(status='UNAVAILABLE', message='veo capacity exhausted')

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg-model', start=start, check=check)

    response = await ai.generate(model='bg-model', prompt='a cat', use=[SwallowsStart()])

    assert response.text == 'FLASH'
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_persists_clean_history_without_injected_docs(ai: Genkit) -> None:
    """Injected RAG text stays off response.request.messages."""
    register_bg_model(ai)

    response = await ai.generate(
        model='bg-model',
        prompt='render a cat',
        docs=[Document.from_text('SECRET-CONTEXT-DOC')],
    )

    assert response.request is not None
    dumped = ' '.join(m.text for m in response.request.messages)
    assert 'SECRET-CONTEXT-DOC' not in dumped
    assert 'render a cat' in dumped


@pytest.mark.asyncio
async def test_generate_rejects_resume_on_background_model(ai: Genkit) -> None:
    """A video start cannot satisfy an interrupt resume. Don't bill start()."""
    starts: list[str] = []
    register_bg_model(ai, starts=starts)

    with pytest.raises(GenkitError, match='Cannot resume background model') as exc_info:
        await ai.generate(
            model='bg-model',
            messages=[
                Message(role=Role.USER, content=[Part.from_text('hi')]),
                Message(
                    role=Role.MODEL,
                    content=[
                        Part(tool_request=ToolRequest(name='ping', input={}, ref='1')),
                    ],
                ),
            ],
            resume_respond=[Part.from_tool_response(name='ping', ref='1', output='ok')],
        )

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert exc_info.value.reason is RuntimeErrorReason.INVALID_RESUME
    assert 'INVALID_RESUME' not in exc_info.value.original_message
    assert starts == []


@pytest.mark.asyncio
async def test_generate_stamps_latency_from_action_run(ai: Genkit) -> None:
    """Action.run's clock lands on the boxed response for both registration shapes."""
    register_bg_model(ai)
    wrapped = await ai.generate(model='bg-model', prompt='a cat')
    assert wrapped.latency_ms is not None
    assert wrapped.latency_ms >= 0

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='raw-1', done=False)

    _register_raw_background(ai, name='raw-bg', start=start)
    raw = await ai.generate(model='raw-bg', prompt='a cat')
    assert raw.latency_ms is not None
    assert raw.latency_ms >= 0


def _register_raw_background(ai: Genkit, *, name: str, start: Callable[..., Awaitable[object]]) -> None:
    ai.registry.register_action(name=name, kind=ActionKind.BACKGROUND_MODEL, fn=start)


@pytest.mark.asyncio
async def test_box_stamps_operation_action_so_check_can_poll(ai: Genkit) -> None:
    """A raw start that forgot action can still be polled after generate()."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='raw-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op.id, done=True, action=op.action)

    _register_raw_background(ai, name='raw-bg', start=start)
    ai.registry.register_action(name='raw-bg/check', kind=ActionKind.CHECK_OPERATION, fn=check)

    response = await ai.generate(model='raw-bg', prompt='a cat')
    assert response.operation is not None
    assert response.operation.action == '/background-model/raw-bg'

    updated = await ai.check_operation(response.operation)
    assert updated.id == 'raw-1'
    assert updated.done is True


@pytest.mark.asyncio
async def test_generate_background_start_without_operation_returns_closed_history(ai: Genkit) -> None:
    """start() that returns a ModelResponse leaves the prompt and no ticket."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(operation=Operation(id='boxed-1', done=False))

    _register_raw_background(ai, name='boxed-bg', start=start)

    response = await ai.generate(model='boxed-bg', prompt='a cat')
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "'boxed-bg' did not return an operation" in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'a cat'
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_operation_raises_when_start_returns_no_ticket(ai: Genkit) -> None:
    """generate_operation still raises when start() did not return a ticket."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(operation=Operation(id='boxed-1', done=False))

    _register_raw_background(ai, name='boxed-bg', start=start)

    with pytest.raises(GenkitError, match="'boxed-bg' did not return an operation") as exc_info:
        await ai.generate_operation(model='boxed-bg', prompt='a cat')

    assert exc_info.value.status == 'FAILED_PRECONDITION'


class DenyStart(BaseMiddleware):
    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        raise RuntimeError('nope')

    async def wrap_model(
        self,
        params: ModelHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next_fn(params, ctx)


@pytest.mark.asyncio
async def test_generate_operation_raises_the_boxed_error_when_start_never_ran(ai: Genkit) -> None:
    """A boxed failure with no ticket is that failure, not 'did not return an operation'."""
    register_bg_model(ai)

    boxed = await ai.generate(model='bg-model', prompt='a cat', use=[DenyStart()])
    assert boxed.error is not None
    assert boxed.operation is None
    assert boxed.finish_message == 'internal error'

    with pytest.raises(GenkitError) as ei:
        await ai.generate_operation(model='bg-model', prompt='a cat', use=[DenyStart()])

    assert ei.value.original_message == boxed.error.message
    assert ei.value.status == boxed.error.status
    assert 'did not return an operation' not in ei.value.original_message


@pytest.mark.asyncio
async def test_generate_chat_model_returning_operation_returns_closed_history(ai: Genkit) -> None:
    """A chat model that returns a bare Operation leaves the prompt and no model message."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='sneaky', done=False)

    ai.define_model(name='plain', fn=cast(Any, model_fn))

    response = await ai.generate(model='plain', prompt='hi')
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'define_background_model' in response.finish_message
    assert 'plain' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_chat_model_returning_dict_returns_closed_history(ai: Genkit) -> None:
    """A chat model that returns a dict leaves the prompt and no model message."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> dict[str, object]:
        return {'operation': {'id': 'op-1', 'done': False}}

    ai.define_model(name='plain-dict', fn=model_fn)

    response = await ai.generate(model='plain-dict', prompt='hi')
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Model 'plain-dict' did not return a ModelResponse" in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_chat_model_returning_none_returns_closed_history(ai: Genkit) -> None:
    """A chat model that returns None leaves the prompt and no model message."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> None:
        return None

    ai.define_model(name='plain-none', fn=model_fn)

    response = await ai.generate(model='plain-none', prompt='hi')
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert "Model 'plain-none' did not return a ModelResponse" in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_chat_model_returning_response_with_operation_returns_closed_history(
    ai: Genkit,
) -> None:
    """A chat model that stuffs a handle onto ModelResponse does not keep that ticket."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text('Started')]),
            operation=Operation(id='lro-1', done=False),
        )

    ai.define_model(name='plain', fn=model_fn)

    response = await ai.generate(model='plain', prompt='hi')
    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert 'define_background_model' in response.finish_message
    assert response.message is None
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is None
    assert response.error.message == response.finish_message
    assert [message.role for message in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'
    assert response.operation is None


def test_model_response_messages_sees_request_set_after_first_read() -> None:
    """History is request + message. A first read before request is attached must not stick."""
    resp = ModelResponse(operation=Operation(id='x'))
    assert resp.messages == []
    resp.request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('a cat')])],
    )
    assert [m.text for m in resp.messages] == ['a cat']


def test_model_response_views_see_message_set_after_first_read() -> None:
    """A hook that touches text/interrupts/output before message is set must not freeze empty."""
    resp = ModelResponse()
    assert resp.text == ''
    assert resp.interrupts == []
    assert resp.tool_requests == []
    assert resp.media == []
    assert resp.output is None

    resp.message = Message(
        role=Role.MODEL,
        content=[
            Part(tool_request=ToolRequest(name='ping', input={}), metadata={'interrupt': True}),
            Part.from_text('{"ok": true}'),
        ],
    )
    assert resp.text == '{"ok": true}'
    assert len(resp.interrupts) == 1
    assert len(resp.tool_requests) == 1
    assert resp.output == {'ok': True}


def test_model_response_eq_uses_operation_snapshot() -> None:
    """Same job and poll state match even when start timing differs."""
    a = ModelResponse(operation=Operation(id='unique-a', metadata={'latencyMs': 0.166}))
    b = ModelResponse(operation=Operation(id='unique-a', metadata={'latencyMs': 0.002}))
    c = ModelResponse(operation=Operation(id='unique-b'))
    assert a == b
    assert a != c

    in_flight = ModelResponse(operation=Operation(id='job1', done=False))
    finished = ModelResponse(operation=Operation(id='job1', done=True, output={'url': 'x'}))
    failed = ModelResponse(operation=Operation(id='job1', done=True, error=Error(message='boom')))
    assert in_flight != finished
    assert finished != failed


@pytest.mark.asyncio
async def test_started_operation_dump_round_trips_through_check(ai: Genkit) -> None:
    """The dump of a real start() handle reloads and polls without edits."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-123', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op.id, done=True)

    ai.define_background_model(
        name='bg-model',
        start=start,
        check=check,
    )
    started = await ai.generate_operation(model='bg-model', prompt='a cat video')
    dumped = started.model_dump(by_alias=True)
    assert 'latencyMs' not in dumped
    assert dumped.get('metadata') in (None, {})

    reloaded = Operation.model_validate(dumped)
    updated = await ai.check_operation(reloaded)

    assert updated.id == 'bg-op-123'
    assert updated.done is True
    assert updated.action == '/background-model/bg-model'


@pytest.mark.asyncio
async def test_generate_operation_passes_context_to_start(ai: Genkit) -> None:
    """generate_operation(context=) already reaches start. Keep that hop."""
    seen: dict[str, Any] = {}

    async def start(_request: ModelRequest, ctx: ActionRunContext) -> Operation:
        seen['start'] = dict(ctx.context)
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg-model', start=start, check=check)

    await ai.generate_operation(
        model='bg-model',
        prompt='a cat',
        context={'secrets': {'api_key': 'sk-start'}},
    )

    assert seen['start']['secrets']['api_key'] == 'sk-start'


@pytest.mark.asyncio
async def test_background_action_start_passes_context(ai: Genkit) -> None:
    """Holding the wrapper and calling start() still delivers the tenant key."""
    seen: dict[str, Any] = {}

    async def start(_request: ModelRequest, ctx: ActionRunContext) -> Operation:
        seen['start'] = dict(ctx.context)
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    bg = ai.define_background_model(name='bg-model', start=start, check=check)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('a cat')])],
    )

    await bg.start(request, context={'secrets': {'api_key': 'tenant'}})

    assert seen['start']['secrets']['api_key'] == 'tenant'


@pytest.mark.asyncio
async def test_check_and_cancel_pass_folded_context(ai: Genkit) -> None:
    """check/cancel hand the plugin secrets and fold config= into context['config']."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['check'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['cancel'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check, cancel=cancel)
    op = await ai.generate_operation(model='bg-model', prompt='a cat')
    context = {'secrets': {'api_key': 'sk-live'}, 'config': {'base_url': 'https://old'}}
    config = MappingProxyType({'base_url': 'https://x'})

    await ai.check_operation(
        op,
        context=context,
        config=config,
    )
    await ai.cancel_operation(
        op,
        context={'secrets': {'api_key': 'sk-live'}},
        config=config,
    )

    assert seen['check']['secrets']['api_key'] == 'sk-live'
    assert seen['check']['config']['base_url'] == 'https://x'
    assert seen['cancel']['secrets']['api_key'] == 'sk-live'
    assert seen['cancel']['config']['base_url'] == 'https://x'
    assert context == {'secrets': {'api_key': 'sk-live'}, 'config': {'base_url': 'https://old'}}
    assert config == {'base_url': 'https://x'}


@pytest.mark.asyncio
async def test_bare_check_operation_still_works(ai: Genkit) -> None:
    """The plugin-key loop does not require context= or config=."""
    seen: dict[str, object] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['check'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check)
    op = await ai.generate_operation(model='bg-model', prompt='a cat')

    updated = await ai.check_operation(op)

    assert updated.done is True
    assert seen['check'] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['check_operation', 'cancel_operation'])
@pytest.mark.parametrize('config', [{'base_url': 'https://x'}, {}], ids=['configured', 'empty'])
async def test_operation_config_keeps_ambient_context(
    ai: Genkit,
    method_name: str,
    config: dict[str, str],
) -> None:
    """Connection knobs don't discard the tenant context of the current call."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check, cancel=cancel)
    token = _action_context.set({'secrets': {'api_key': 'sk-ambient'}, 'locale': 'en-US'})
    try:
        op = await ai.generate_operation(model='bg-model', prompt='a cat')
        await getattr(ai, method_name)(op, config=MappingProxyType(config))
    finally:
        _action_context.reset(token)

    assert seen['operation'] == {
        'secrets': {'api_key': 'sk-ambient'},
        'locale': 'en-US',
        'config': config,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['check_operation', 'cancel_operation'])
async def test_empty_context_replaces_ambient(ai: Genkit, method_name: str) -> None:
    """An explicit empty context means plugin-key / no tenant, not a no-op."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check, cancel=cancel)
    token = _action_context.set({'secrets': {'api_key': 'sk-ambient'}})
    try:
        op = await ai.generate_operation(model='bg-model', prompt='a cat')
        await getattr(ai, method_name)(op, context={})
    finally:
        _action_context.reset(token)

    assert seen['operation'] == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['check_operation', 'cancel_operation'])
async def test_empty_context_with_config_replaces_ambient(ai: Genkit, method_name: str) -> None:
    """Connection knobs don't undo an explicit request to clear context."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check, cancel=cancel)
    token = _action_context.set({'secrets': {'api_key': 'sk-ambient'}})
    try:
        op = await ai.generate_operation(model='bg-model', prompt='a cat')
        await getattr(ai, method_name)(op, context={}, config={'base_url': 'https://x'})
    finally:
        _action_context.reset(token)

    assert seen['operation'] == {'config': {'base_url': 'https://x'}}


@pytest.mark.asyncio
@pytest.mark.parametrize('method_name', ['check_operation', 'cancel_operation'])
async def test_bare_operation_inherits_ambient_context(ai: Genkit, method_name: str) -> None:
    """A poll inside the current request sees its tenant without extra plumbing."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    async def cancel(op: Operation, ctx: ActionRunContext) -> Operation:
        seen['operation'] = dict(ctx.context)
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check, cancel=cancel)
    token = _action_context.set({'secrets': {'api_key': 'sk-ambient'}})
    try:
        op = await ai.generate_operation(model='bg-model', prompt='a cat')
        await getattr(ai, method_name)(op)
    finally:
        _action_context.reset(token)

    assert seen['operation'] == {'secrets': {'api_key': 'sk-ambient'}}


@pytest.mark.asyncio
async def test_generate_operation_empty_context_replaces_ambient(ai: Genkit) -> None:
    """generate_operation(..., context={}) is plugin-key this time, not ambient."""
    seen: dict[str, dict[str, Any]] = {}

    async def start(_request: ModelRequest, ctx: ActionRunContext) -> Operation:
        seen['start'] = dict(ctx.context)
        return Operation(id='bg-op-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai.define_background_model(name='bg-model', start=start, check=check)
    token = _action_context.set({'secrets': {'api_key': 'sk-ambient'}})
    try:
        await ai.generate_operation(model='bg-model', prompt='a cat', context={})
    finally:
        _action_context.reset(token)

    assert seen['start'] == {}


class RerouteConfig(BaseModel):
    to: str
    turn: int | None = None


class Reroute(BaseMiddleware[RerouteConfig]):
    """Tries to swap ``params.options.model``. The door already bound the action."""

    async def wrap_generate(
        self,
        params: GenerateHookParams,
        ctx: GenerateMiddlewareContext,
        next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self.config.turn is not None and params.iteration != self.config.turn:
            return await next_fn(params, ctx)
        options = params.options.model_copy(update={'model': self.config.to})
        return await next_fn(params.model_copy(update={'options': options}), ctx)


def register_plain(ai: Genkit, *, name: str = 'plain', text: str = 'from-plain') -> None:
    async def model_fn(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(text)]))
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))

    ai.define_model(name=name, fn=model_fn)


def register_tool_caller(ai: Genkit, *, name: str = 'flash') -> None:
    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='ping', input={}, ref='1'))],
            )
        )

    ai.define_model(name=name, fn=model_fn)


def interrupted_history() -> list[Message]:
    return [
        Message(role=Role.USER, content=[Part.from_text('hi')]),
        Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='ping', input={}, ref='1'))],
        ),
    ]


def respond_ping() -> Part:
    return Part.from_tool_response(name='ping', ref='1', output='ok')


def restart_ping() -> Part:
    return Part.from_tool_request(name='ping', input={}, ref='1')


def register_ping(ai: Genkit, runs: list[str] | None = None) -> None:
    @ai.tool(name='ping')
    async def ping() -> str:
        if runs is not None:
            runs.append('ping')
        return 'pong'


def assert_chat(response: ModelResponse, *, text: str, roles: list[Role] | None = None) -> None:
    assert response.operation is None
    assert response.message is not None
    assert response.messages[-1] == response.message
    assert response.text == text
    if roles is not None:
        assert [m.role for m in response.messages] == roles


def assert_ticket(response: ModelResponse, *, op_id: str = 'bg-op-123') -> None:
    assert response.operation is not None
    assert response.operation.id == op_id
    assert response.operation.done is False
    assert response.operation.action == '/background-model/bg-model'
    assert response.message is None
    assert response.text == ''


@pytest.mark.asyncio
async def test_wrap_generate_reroute_to_background_keeps_the_named_model(ai: Genkit) -> None:
    """A hook that writes a Veo id does not start Veo. They named a chat model."""
    register_bg_model(ai)
    register_plain(ai)

    response = await ai.generate(model='plain', prompt='a cat', use=[Reroute(to='bg-model')])

    assert_chat(response, text='from-plain')


@pytest.mark.asyncio
async def test_wrap_generate_reroute_from_background_still_starts(ai: Genkit) -> None:
    """They named a background model. A hook cannot turn that into a chat call."""
    starts: list[str] = []

    register_bg_model(ai, starts=starts)
    register_plain(ai)

    response = await ai.generate(model='bg-model', prompt='a cat', use=[Reroute(to='plain')])

    assert_ticket(response)
    assert starts == ['start']


@pytest.mark.asyncio
async def test_wrap_generate_reroute_same_kind_keeps_the_named_model(ai: Genkit) -> None:
    """flash→pro in a hook still runs flash."""
    register_plain(ai, name='flash', text='from-flash')
    register_plain(ai, name='pro', text='from-pro')

    response = await ai.generate(model='flash', prompt='hi', use=[Reroute(to='pro')])

    assert_chat(response, text='from-flash')


@pytest.mark.asyncio
async def test_wrap_generate_reroute_to_missing_model_keeps_the_named_model(ai: Genkit) -> None:
    """A swap to a missing name is ignored. The model they named still runs."""
    register_plain(ai)

    response = await ai.generate(model='plain', prompt='hi', use=[Reroute(to='no-such-model')])

    assert_chat(response, text='from-plain')


@pytest.mark.asyncio
async def test_wrap_generate_in_place_model_assignment_keeps_the_named_model(ai: Genkit) -> None:
    """Writing options.model in place does not change which action runs."""
    register_plain(ai, name='flash', text='from-flash')
    register_plain(ai, name='pro', text='from-pro')

    class InPlace(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            params.options.model = 'pro'
            return await next_fn(params, ctx)

    response = await ai.generate(model='flash', prompt='hi', use=[InPlace()])

    assert_chat(response, text='from-flash')


@pytest.mark.asyncio
async def test_wrap_generate_cannot_rescue_a_missing_model_name(ai: Genkit) -> None:
    """A missing name fails at the door, before a hook can rewrite it."""
    register_plain(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='no-such-model', prompt='hi', use=[Reroute(to='plain')])

    assert raised.value.status == 'NOT_FOUND'
    assert raised.value.reason is RuntimeErrorReason.MODEL_NOT_FOUND


@pytest.mark.asyncio
async def test_resume_restart_on_video_without_swap_does_not_run_the_tool(ai: Genkit) -> None:
    """Still on Veo after the hook: reject before a resume restart runs the tool."""
    starts: list[str] = []
    runs: list[str] = []

    register_bg_model(ai, starts=starts)
    register_ping(ai, runs)

    with pytest.raises(GenkitError, match='Cannot resume background model') as raised:
        await ai.generate(
            model='bg-model',
            messages=interrupted_history(),
            resume_restart=[restart_ping()],
            tools=['ping'],
        )

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert starts == []
    assert runs == []


@pytest.mark.asyncio
async def test_wrap_generate_cannot_swap_off_video_to_resume(ai: Genkit) -> None:
    """They named Veo. A hook cannot turn resume into a chat continue."""
    starts: list[str] = []

    register_bg_model(ai, starts=starts)
    register_plain(ai)

    with pytest.raises(GenkitError, match='Cannot resume background model') as raised:
        await ai.generate(
            model='bg-model',
            messages=interrupted_history(),
            resume_respond=[respond_ping()],
            use=[Reroute(to='plain')],
        )

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert starts == []


@pytest.mark.asyncio
async def test_wrap_generate_cannot_swap_off_video_to_restart_a_tool(ai: Genkit) -> None:
    """They named Veo. A hook cannot make a resume restart run the tool."""
    starts: list[str] = []
    runs: list[str] = []

    register_bg_model(ai, starts=starts)
    register_plain(ai)
    register_ping(ai, runs)

    with pytest.raises(GenkitError, match='Cannot resume background model') as raised:
        await ai.generate(
            model='bg-model',
            messages=interrupted_history(),
            resume_restart=[restart_ping()],
            tools=['ping'],
            use=[Reroute(to='plain')],
        )

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert starts == []
    assert runs == []


@pytest.mark.asyncio
async def test_wrap_generate_resume_on_flash_ignores_swap_to_video(ai: Genkit) -> None:
    """They named a chat model. Resume continues there even if a hook writes Veo."""
    starts: list[str] = []

    register_bg_model(ai, starts=starts)
    register_plain(ai)

    response = await ai.generate(
        model='plain',
        messages=interrupted_history(),
        resume_respond=[respond_ping()],
        use=[Reroute(to='bg-model')],
    )

    assert_chat(response, text='from-plain', roles=[Role.USER, Role.MODEL, Role.TOOL, Role.MODEL])
    assert starts == []


@pytest.mark.asyncio
async def test_wrap_generate_resume_restart_on_flash_ignores_swap_to_video(ai: Genkit) -> None:
    """A resume restart on a chat model runs the tool. A Veo id in the hook does not."""
    starts: list[str] = []
    runs: list[str] = []

    register_bg_model(ai, starts=starts)
    register_plain(ai)
    register_ping(ai, runs)

    response = await ai.generate(
        model='plain',
        messages=interrupted_history(),
        resume_restart=[restart_ping()],
        tools=['ping'],
        use=[Reroute(to='bg-model')],
    )

    assert_chat(response, text='from-plain', roles=[Role.USER, Role.MODEL, Role.TOOL, Role.MODEL])
    assert starts == []
    assert runs == ['ping']


@pytest.mark.asyncio
async def test_wrap_generate_cannot_swap_model_on_the_turn_after_a_tool(ai: Genkit) -> None:
    """After a closed tool round the named model still runs, not the hook's rewrite."""
    calls = 0

    async def flash(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='ping', input={}, ref='1'))],
                )
            )
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('from-flash')]))

    ai.define_model(name='flash', fn=flash)
    register_plain(ai, name='pro', text='from-pro')
    register_ping(ai)

    response = await ai.generate(model='flash', prompt='hi', tools=['ping'], use=[Reroute(to='pro', turn=1)])

    assert_chat(response, text='from-flash', roles=[Role.USER, Role.MODEL, Role.TOOL, Role.MODEL])
    assert calls == 2


@pytest.mark.asyncio
async def test_wrap_generate_cannot_swap_to_video_on_the_turn_after_a_tool(ai: Genkit) -> None:
    """After a closed tool round a Veo id in the hook does not start a job."""
    calls = 0

    async def flash(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='ping', input={}, ref='1'))],
                )
            )
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('from-flash')]))

    ai.define_model(name='flash', fn=flash)
    register_bg_model(ai)
    register_ping(ai)

    response = await ai.generate(model='flash', prompt='hi', tools=['ping'], use=[Reroute(to='bg-model', turn=1)])

    assert_chat(response, text='from-flash', roles=[Role.USER, Role.MODEL, Role.TOOL, Role.MODEL])
    assert response.operation is None


@pytest.mark.asyncio
async def test_generate_stream_keeps_the_named_model_when_hook_swaps(ai: Genkit) -> None:
    """generate_stream uses the name they passed. Chunks and the final reply match."""
    register_plain(ai, name='flash', text='from-flash')
    register_plain(ai, name='pro', text='from-pro')

    stream = ai.generate_stream(model='flash', prompt='hi', use=[Reroute(to='pro')])
    texts: list[str] = []
    async for chunk in stream.stream:
        texts.append(chunk.text)
    response = await stream.response

    assert ''.join(texts) == 'from-flash'
    assert_chat(response, text='from-flash')


@pytest.mark.asyncio
async def test_generate_stream_keeps_chat_when_hook_writes_video(ai: Genkit) -> None:
    """A stream hook that writes Veo still streams the chat model they named."""
    register_plain(ai)
    register_bg_model(ai)

    stream = ai.generate_stream(model='plain', prompt='a cat', use=[Reroute(to='bg-model')])
    texts: list[str] = []
    async for chunk in stream.stream:
        texts.append(chunk.text)
    response = await stream.response

    assert ''.join(texts) == 'from-plain'
    assert_chat(response, text='from-plain')


@pytest.mark.asyncio
async def test_wrap_generate_on_resume_sees_the_model_message_and_resume(ai: Genkit) -> None:
    """On a resume turn the hook sees the model message and resume, once."""
    register_plain(ai)
    seen: list[dict[str, object]] = []

    class ResumeSpy(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            messages = params.options.messages
            seen.append({
                'iteration': params.iteration,
                'last_role': messages[-1].role if messages else None,
                'has_resume': params.options.resume is not None,
            })
            return await next_fn(params, ctx)

    response = await ai.generate(
        model='plain',
        messages=interrupted_history(),
        resume_respond=[respond_ping()],
        use=[ResumeSpy()],
    )

    assert_chat(response, text='from-plain', roles=[Role.USER, Role.MODEL, Role.TOOL, Role.MODEL])
    assert seen == [{'iteration': 0, 'last_role': Role.MODEL, 'has_resume': True}]


@pytest.mark.asyncio
async def test_wrap_generate_short_circuit_cannot_skip_a_missing_model(ai: Genkit) -> None:
    """A missing name fails at the door, before a hook can return a cached reply."""

    class ReturnsFlashWithoutNext(BaseMiddleware):
        async def wrap_generate(
            self,
            params: GenerateHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[GenerateHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return ModelResponse(
                message=Message(role=Role.MODEL, content=[Part.from_text('FLASH')]),
                finish_reason=FinishReason.STOP,
            )

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='no-such-model', prompt='hi', use=[ReturnsFlashWithoutNext()])

    assert raised.value.status == 'NOT_FOUND'
    assert raised.value.reason is RuntimeErrorReason.MODEL_NOT_FOUND


@pytest.mark.asyncio
async def test_generate_operation_swap_from_flash_is_not_long_running(ai: Genkit) -> None:
    """generate_operation gates on the name they passed. A swap to Veo never runs."""
    register_plain(ai, name='flash', text='from-flash')
    register_bg_model(ai)

    with pytest.raises(GenkitError, match='does not support long running operations') as raised:
        await ai.generate_operation(model='flash', prompt='a cat', use=[Reroute(to='bg-model')])

    assert raised.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_generate_operation_keeps_the_named_background_model(ai: Genkit) -> None:
    """generate_operation on Veo returns that ticket. A hook cannot run flash instead."""
    register_bg_model(ai)
    register_plain(ai)

    operation = await ai.generate_operation(model='bg-model', prompt='a cat', use=[Reroute(to='plain')])

    assert operation.id == 'bg-op-123'
    assert operation.action == '/background-model/bg-model'
