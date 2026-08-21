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

import pytest

from genkit import Genkit, Message
from genkit._core._action import ActionRunContext
from genkit._core._background import BackgroundAction
from genkit._core._error import GenkitError
from genkit._core._middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._typing import (
    FinishReason,
    ModelInfo,
    Operation,
    Part,
    Role,
    Supports,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


async def register_bg_model(ai: Genkit, *, op_id: str = 'bg-op-123') -> BackgroundAction:
    async def start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        return Operation(id=op_id, done=False)

    async def check(op: Operation) -> Operation:
        return op

    return ai.define_background_model(
        name='bg-model',
        start=start,
        check=check,
    )


@pytest.mark.asyncio
async def test_generate_returns_operation_for_background_model(ai: Genkit) -> None:
    """generate() wraps the start handle. message stays empty."""
    await register_bg_model(ai)

    response = await ai.generate(model='bg-model', prompt='a cat surfing')

    assert response.operation is not None
    assert response.operation.id == 'bg-op-123'
    assert response.operation.done is False
    assert response.operation.action == '/background-model/bg-model'
    assert response.message is None


@pytest.mark.asyncio
async def test_generate_operation_with_background_model(ai: Genkit) -> None:
    """generate_operation() returns that same handle."""
    await register_bg_model(ai, op_id='bg-op-456')

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
    await register_bg_model(ai)

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
    """A hook that rebuilds ModelResponse without operation must not blame the model."""
    await register_bg_model(ai)

    with pytest.raises(GenkitError, match='wrap_model returned no operation') as exc_info:
        await ai.generate(model='bg-model', prompt='a cat surfing', use=[DropsOperation()])

    assert exc_info.value.status == 'FAILED_PRECONDITION'

    with pytest.raises(GenkitError, match='wrap_model returned no operation') as op_exc:
        await ai.generate_operation(model='bg-model', prompt='a cat surfing', use=[DropsOperation()])

    assert op_exc.value.status == 'FAILED_PRECONDITION'
    assert 'did not return an operation' not in str(op_exc.value)


@pytest.mark.asyncio
async def test_generate_on_lro_define_model_still_runs_tools(ai: Genkit) -> None:
    """A define_model LRO that also returns a tool request still runs the tool loop."""
    tool_ran = 0

    @ai.tool(name='ping')
    async def ping() -> str:
        nonlocal tool_ran
        tool_ran += 1
        return 'pong'

    turns = 0

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal turns
        turns += 1
        if turns == 1:
            return ModelResponse(
                message=Message(
                    role=Role.MODEL,
                    content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='ping', input={}, ref='1')))],
                ),
                operation=Operation(id='lro-1', done=False),
                finish_reason=FinishReason.STOP,
            )
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='done'))]),
            finish_reason=FinishReason.STOP,
        )

    ai.define_model(
        name='lro-model',
        fn=model_fn,
        info=ModelInfo(supports=Supports(long_running=True, tools=True)),
    )

    response = await ai.generate(model='lro-model', prompt='x', tools=['ping'])

    assert tool_ran == 1
    assert response.text == 'done'
    assert response.operation is None


def test_model_response_eq_includes_operation() -> None:
    """Two start handles with different job ids are not the same response."""
    a = ModelResponse(operation=Operation(id='unique-a'))
    b = ModelResponse(operation=Operation(id='unique-b'))
    assert a != b
    assert a == ModelResponse(operation=Operation(id='unique-a'))
