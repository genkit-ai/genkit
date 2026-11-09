# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Failed turns raise by default; ``raise_on_error=False`` keeps the boxed response.

One test per symptom the 1.0 error-contract flip fixes: provider errors,
``PublicError`` from tools, guardrail exceptions in middleware, ``max_turns``,
``INVALID_OUTPUT``, a missing tool, and a bad resume. Plus the callers that
must stay boxed: ``raise_on_error=False``, a stream the caller closed, and the
Dev UI ``/util/generate`` action.
"""

import asyncio
from collections.abc import Awaitable, Callable

import pytest
from pydantic import BaseModel

from genkit import (
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    PublicError,
    Role,
)
from genkit._core._action import ActionKind
from genkit._core._error import ErrorReason, error_status
from genkit._core._model import GenerateActionOptions, ModelRequest
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import ModelContext, ToolRequest, ToolResponse


class Recipe(BaseModel):
    title: str
    steps: list[str]


def _says(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _calls(tool: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=tool, ref='1', input='A-17'))],
        ),
    )


def _model(ai: Genkit, name: str, reply: Callable[[ModelRequest], ModelResponse]) -> None:
    async def fn(request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        return reply(request)

    ai.define_model(name=name, fn=fn)


def _overloaded(_request: ModelRequest) -> ModelResponse:
    raise GenkitError(status='UNAVAILABLE', message='model overloaded')


@pytest.mark.asyncio
async def test_provider_error_raises_from_generate() -> None:
    """A model error reaches the caller with its status instead of FAILED and text ''."""
    ai = Genkit(model='flaky')
    _model(ai, 'flaky', _overloaded)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='Suggest a dish.')

    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.message == 'model overloaded'
    # Not a PublicError: a served flow redacts it to a 500, never a 200.
    assert error_status(raised.value) == 500


@pytest.mark.asyncio
async def test_provider_error_raises_from_prompt_call_and_stream() -> None:
    """Prompt __call__ and .stream follow the same contract as ai.generate."""
    ai = Genkit(model='flaky')
    _model(ai, 'flaky', _overloaded)
    menu = ai.define_prompt(name='menu', prompt='Suggest a dish.')

    with pytest.raises(GenkitError) as from_call:
        await menu()
    assert from_call.value.status == 'UNAVAILABLE'

    stream = menu.stream()
    with pytest.raises(GenkitError) as from_iter:
        async for _ in stream:
            pass
    assert from_iter.value.status == 'UNAVAILABLE'
    with pytest.raises(GenkitError) as from_response:
        await stream.response
    assert from_response.value.status == 'UNAVAILABLE'


@pytest.mark.asyncio
async def test_tool_public_error_propagates_as_public_error() -> None:
    """RBAC in a tool: PublicError keeps status and details, so a served flow returns 403."""
    ai = Genkit(model='waiter')
    turns = iter([_calls('refund_order'), _says('done')])
    _model(ai, 'waiter', lambda _r: next(turns))

    @ai.tool()
    async def refund_order(order_id: str) -> str:
        raise PublicError('PERMISSION_DENIED', 'Not your order.', details={'order_id': order_id})

    with pytest.raises(PublicError) as raised:
        await ai.generate(prompt='Refund order A-17.', tools=['refund_order'])

    assert raised.value.status == 'PERMISSION_DENIED'
    assert raised.value.message == 'Not your order.'
    assert raised.value.details == {'order_id': 'A-17'}
    assert error_status(raised.value) == 403


@pytest.mark.asyncio
async def test_guardrail_exception_in_wrap_model_raises() -> None:
    """A guardrail that raises in wrap_model fails the call instead of returning text ''."""
    ai = Genkit(model='chef')
    _model(ai, 'chef', lambda _r: _says('Peanut satay'))

    class AllergyGuard(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            response = await next_fn(params, ctx)
            if 'peanut' in response.text.lower():
                raise ValueError('contains peanuts')
            return response

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='Suggest a dish.', use=[AllergyGuard()])

    assert raised.value.status == 'INTERNAL'
    assert raised.value.message == 'ValueError: contains peanuts'


@pytest.mark.asyncio
async def test_max_turns_raises_aborted() -> None:
    """Running out of tool rounds raises ABORTED / MAX_TURNS_EXCEEDED; boxed it is finish_reason ABORTED."""
    ai = Genkit(model='looper')
    _model(ai, 'looper', lambda _r: _calls('check_stock'))

    @ai.tool()
    async def check_stock(order_id: str) -> str:
        return 'in stock'

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='Is A-17 in stock?', tools=['check_stock'], max_turns=1)

    assert raised.value.status == 'ABORTED'
    assert raised.value.reason is ErrorReason.MAX_TURNS_EXCEEDED
    assert raised.value.message == 'Exceeded maximum tool call iterations (1)'

    boxed = await ai.generate(prompt='Is A-17 in stock?', tools=['check_stock'], max_turns=1, raise_on_error=False)
    assert boxed.finish_reason == FinishReason.ABORTED
    assert boxed.error is not None
    assert boxed.error.reason is ErrorReason.MAX_TURNS_EXCEEDED


@pytest.mark.asyncio
async def test_invalid_output_raises_and_boxed_keeps_text() -> None:
    """Output that fails output_schema raises INVALID_OUTPUT; raise_on_error=False keeps the raw text."""
    ai = Genkit(model='chef')
    _model(ai, 'chef', lambda _r: _says('Just make pancakes.'))

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='A recipe.', output_schema=Recipe)

    assert raised.value.status == 'INTERNAL'
    assert raised.value.reason is ErrorReason.INVALID_OUTPUT

    boxed = await ai.generate(prompt='A recipe.', output_schema=Recipe, raise_on_error=False)
    assert boxed.finish_reason == FinishReason.STOP
    assert boxed.output is None
    assert boxed.text == 'Just make pancakes.'
    assert boxed.error is not None
    assert boxed.error.reason is ErrorReason.INVALID_OUTPUT


@pytest.mark.asyncio
async def test_missing_tool_raises_not_found() -> None:
    """The model naming a tool nobody registered raises NOT_FOUND / TOOL_NOT_FOUND."""
    ai = Genkit(model='waiter')
    _model(ai, 'waiter', lambda _r: _calls('cancel_order'))

    @ai.tool()
    async def check_stock(order_id: str) -> str:
        return 'in stock'

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='Cancel A-17.', tools=['check_stock'])

    assert raised.value.status == 'NOT_FOUND'
    assert raised.value.reason is ErrorReason.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_invalid_resume_raises() -> None:
    """Resuming when the last message has no tool request raises instead of skipping the approved tool."""
    ai = Genkit(model='waiter')
    _model(ai, 'waiter', lambda _r: _says('ok'))

    approval = Part(tool_response=ToolResponse(name='refund_order', ref='1', output={'approved': True}))
    with pytest.raises(GenkitError) as raised:
        await ai.generate(
            messages=[Message(role=Role.USER, content=[Part.from_text('Refund A-17.')])],
            resume_respond=[approval],
        )

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert raised.value.reason is ErrorReason.INVALID_RESUME


@pytest.mark.asyncio
async def test_raise_on_error_false_returns_boxed_response() -> None:
    """raise_on_error=False is JS throwOnError: false: FAILED, error set, no message."""
    ai = Genkit(model='flaky')
    _model(ai, 'flaky', _overloaded)

    response = await ai.generate(prompt='Suggest a dish.', raise_on_error=False)

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'UNAVAILABLE'
    assert response.text == ''
    assert response.message is None


@pytest.mark.asyncio
async def test_caller_closed_stream_still_resolves_aborted() -> None:
    """Closing the stream yourself is not a failure: .response resolves ABORTED, nothing raises."""
    ai = Genkit(model='slow')

    async def slow(_request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('one ')]))
        await asyncio.Event().wait()
        return _says('never')

    ai.define_model(name='slow', fn=slow)

    stream = ai.generate_stream(prompt='Talk.')
    async with stream:
        async for _ in stream:
            break

    response = await stream.response
    assert response.finish_reason == FinishReason.ABORTED


@pytest.mark.asyncio
async def test_dev_ui_util_generate_stays_boxed() -> None:
    """/util/generate returns the failed response so the Dev UI can render it."""
    ai = Genkit()
    _model(ai, 'flaky', _overloaded)
    util = await ai._registry.resolve_action(ActionKind.UTIL, 'generate')  # pyright: ignore[reportPrivateUsage]
    assert util is not None

    result = await util.run(
        GenerateActionOptions(
            model='flaky',
            messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])],
        )
    )

    response = result.response
    assert isinstance(response, ModelResponse)
    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'UNAVAILABLE'
