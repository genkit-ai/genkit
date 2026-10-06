# Copyright 2025 Google LLC
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

"""Tests for Fallback middleware."""

from typing import NoReturn

import pytest
from genkit_middleware import Fallback

from genkit import ActionRunContext, FinishReason, Genkit, Message, ModelResponse, Part, Role
from genkit._core._error import GenkitError
from genkit.middleware import ModelHookParams
from genkit.model import ModelRequest


def _make_params() -> ModelHookParams:
    return ModelHookParams(request=ModelRequest(messages=[]))


def _make_fallback(**kwargs) -> Fallback:
    return Fallback(**kwargs)


@pytest.mark.asyncio
async def test_fallback_success_on_first_model(ctx) -> None:
    """Test that successful primary model calls pass through."""
    fallback = _make_fallback(models=['model2', 'model3'])

    async def next_fn(params, ctx):
        return ModelResponse(message=None)

    result = await fallback.wrap_model(_make_params(), ctx, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_fallback_on_retryable_error(ctx) -> None:
    """Test that retryable errors are classified correctly."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise GenkitError(message='Service unavailable', status='UNAVAILABLE')

    with pytest.raises(GenkitError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_retryable_error(ctx) -> None:
    """Test that non-retryable errors fail immediately."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise GenkitError(message='Invalid argument', status='INVALID_ARGUMENT')

    with pytest.raises(GenkitError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_genkit_error_raises_without_trying_next_model(ctx) -> None:
    """A raw TypeError (a bug in another middleware, say) propagates without fallback."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise TypeError("'NoneType' object is not subscriptable")

    with pytest.raises(TypeError, match='not subscriptable'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_generate_with_unavailable_model_and_fallback_tries_next_model() -> None:
    """With `Fallback(models=['backup'])`, a model raising UNAVAILABLE falls back to `backup`."""
    ai = Genkit()

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='provider is down')

    async def backup(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('from backup')]),
        )

    ai.define_model(name='primary', fn=down)
    ai.define_model(name='backup', fn=backup)

    response = await ai.generate(model='primary', prompt='hi', use=[Fallback(models=['backup'])])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from backup'
    assert response.error is None


@pytest.mark.asyncio
async def test_generate_with_model_raising_connection_error_and_fallback_keeps_the_failure() -> None:
    """An unclassified ConnectionError from the model fails the call without trying `backup`."""
    ai = Genkit()
    backup_calls = 0

    async def down(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        raise ConnectionError('connection refused')

    async def backup(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        nonlocal backup_calls
        backup_calls += 1
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('from backup')]))

    ai.define_model(name='primary', fn=down)
    ai.define_model(name='backup', fn=backup)

    response = await ai.generate(model='primary', prompt='hi', use=[Fallback(models=['backup'])])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == 'internal error'
    assert response.error is not None
    assert response.error.status == 'INTERNAL'
    assert response.message is None
    assert backup_calls == 0
