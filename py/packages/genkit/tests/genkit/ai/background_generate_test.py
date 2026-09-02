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
from typing import Any, cast

import pytest

from genkit import ActionKind, Document, Genkit, Message
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware, GenerateHookParams, GenerateMiddlewareContext, ModelHookParams
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._typing import (
    Error,
    FinishReason,
    Operation,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


def register_bg_model(ai: Genkit, *, op_id: str = 'bg-op-123') -> None:
    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op_id, done=False)

    async def check(op: Operation) -> Operation:
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
    checks = 0

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='bg-op-123', done=False)

    async def check(op: Operation) -> Operation:
        nonlocal checks
        checks += 1
        return Operation(id=op.id, done=True)

    ai.define_background_model(name='bg-model', start=start, check=check)

    response = await ai.generate(model='bg-model', prompt='a cat surfing')
    operation = await ai.generate_operation(model='bg-model', prompt='a cat surfing')

    assert response.operation is not None
    assert response.operation.done is False
    assert operation.done is False
    assert checks == 0


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
async def test_generate_fails_when_wrap_model_drops_operation(ai: Genkit) -> None:
    """A hook that drops a billed ticket is a missing handle on both doors."""
    register_bg_model(ai)

    with pytest.raises(GenkitError, match='did not return an operation') as generate_info:
        await ai.generate(model='bg-model', prompt='a cat surfing', use=[DropsOperation()])
    assert generate_info.value.status == 'FAILED_PRECONDITION'

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
async def test_generate_fails_when_wrap_generate_drops_operation(ai: Genkit) -> None:
    """Same missing-handle error if wrap_generate rebuilds the response without the ticket."""
    register_bg_model(ai)

    with pytest.raises(GenkitError, match='did not return an operation') as generate_info:
        await ai.generate(model='bg-model', prompt='a cat surfing', use=[DropsGenerate()])
    assert generate_info.value.status == 'FAILED_PRECONDITION'

    with pytest.raises(GenkitError, match='did not return an operation') as operation_info:
        await ai.generate_operation(model='bg-model', prompt='a cat surfing', use=[DropsGenerate()])
    assert operation_info.value.status == 'FAILED_PRECONDITION'


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
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='FLASH'))]),
                finish_reason=FinishReason.STOP,
            )


@pytest.mark.asyncio
async def test_generate_keeps_fallback_answer_when_start_raises(ai: Genkit) -> None:
    """A hook that substitutes another model's answer is not a missing handle."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        raise GenkitError(status='UNAVAILABLE', message='veo capacity exhausted')

    async def check(op: Operation) -> Operation:
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
    started = 0

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        nonlocal started
        started += 1
        return Operation(id='bg-op-123', done=False)

    async def check(op: Operation) -> Operation:
        return op

    ai.define_background_model(name='bg-model', start=start, check=check)

    with pytest.raises(GenkitError, match='Cannot resume background model') as exc_info:
        await ai.generate(
            model='bg-model',
            messages=[
                Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))]),
                Message(
                    role=Role.MODEL,
                    content=[
                        Part(root=ToolRequestPart(tool_request=ToolRequest(name='ping', input={}, ref='1'))),
                    ],
                ),
            ],
            resume_respond=[ToolResponsePart(tool_response=ToolResponse(name='ping', ref='1', output='ok'))],
        )

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert started == 0


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
async def test_background_start_must_return_operation(ai: Genkit) -> None:
    """start() returns an Operation. A ModelResponse is a plugin bug."""

    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(operation=Operation(id='boxed-1', done=False))

    _register_raw_background(ai, name='boxed-bg', start=start)

    with pytest.raises(GenkitError, match="'boxed-bg' did not return an operation") as exc_info:
        await ai.generate(model='boxed-bg', prompt='a cat')

    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_define_model_returning_operation_raises(ai: Genkit) -> None:
    """A chat model that returns a bare Operation is registered on the wrong kind."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id='sneaky', done=False)

    ai.define_model(name='plain', fn=cast(Any, model_fn))

    with pytest.raises(GenkitError, match='define_background_model') as exc_info:
        await ai.generate(model='plain', prompt='hi')

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'plain' in str(exc_info.value)


@pytest.mark.asyncio
async def test_define_model_returning_dict_raises(ai: Genkit) -> None:
    """A chat model that returns a dict is a plugin bug, not an AttributeError."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> dict[str, object]:
        return {'operation': {'id': 'op-1', 'done': False}}

    ai.define_model(name='plain-dict', fn=model_fn)

    with pytest.raises(GenkitError, match="Model 'plain-dict' did not return a ModelResponse") as exc_info:
        await ai.generate(model='plain-dict', prompt='hi')

    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_define_model_returning_none_raises(ai: Genkit) -> None:
    """A chat model that returns None is a plugin bug, not an AttributeError."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> None:
        return None

    ai.define_model(name='plain-none', fn=model_fn)

    with pytest.raises(GenkitError, match="Model 'plain-none' did not return a ModelResponse") as exc_info:
        await ai.generate(model='plain-none', prompt='hi')

    assert exc_info.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_define_model_returning_model_response_with_operation_raises(ai: Genkit) -> None:
    """A chat model that stuffs a handle onto ModelResponse is the same mistake."""

    async def model_fn(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Started'))]),
            operation=Operation(id='lro-1', done=False),
        )

    ai.define_model(name='plain', fn=model_fn)

    with pytest.raises(GenkitError, match='define_background_model') as exc_info:
        await ai.generate(model='plain', prompt='hi')

    assert exc_info.value.status == 'FAILED_PRECONDITION'


def test_model_response_messages_sees_request_set_after_first_read() -> None:
    """History is request + message. A first read before request is attached must not stick."""
    resp = ModelResponse(operation=Operation(id='x'))
    assert resp.messages == []
    resp.request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='a cat'))])],
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
            Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(name='ping', input={}),
                    metadata={'interrupt': True},
                )
            ),
            Part(root=TextPart(text='{"ok": true}')),
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

    async def check(op: Operation) -> Operation:
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
