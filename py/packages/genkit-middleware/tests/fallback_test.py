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

from genkit import (
    ActionRunContext,
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
)
from genkit.middleware import GenerateMiddlewareContext, ModelHookParams
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
async def test_fallback_non_genkit_error(ctx) -> None:
    """Test that non-GenkitError exceptions fail immediately."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise ConnectionError('Network failure')

    with pytest.raises(ConnectionError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_stops_when_aborted() -> None:
    """Test that fallback halts and re-raises when the abort signal is already set."""
    ai = Genkit()
    ran: list[str] = []

    async def fail(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    async def backup1(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup1')
        raise GenkitError(status='UNAVAILABLE', message='backup1 down')

    async def backup2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup2')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('backup2')]),
        )

    ai.define_model(name='primary', fn=fail)
    ai.define_model(name='backup1', fn=backup1)
    ai.define_model(name='backup2', fn=backup2)

    ctx = GenerateMiddlewareContext(ai=ai)
    ctx.abort_signal.set()
    fallback = Fallback(models=['backup1', 'backup2'])

    async def next_fn(_params: ModelHookParams, _ctx: GenerateMiddlewareContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    with pytest.raises(GenkitError, match='primary down'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)

    assert ran == []


@pytest.mark.asyncio
async def test_fallback_halts_subsequent_models_on_abort() -> None:
    """Test that fallback stops calling subsequent backup models when aborted mid-sequence."""
    ai = Genkit()
    ran: list[str] = []

    async def backup1(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup1')
        ctx.abort_signal.set()
        raise GenkitError(status='UNAVAILABLE', message='backup1 down')

    async def backup2(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        ran.append('backup2')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('backup2')]),
        )

    ai.define_model(name='backup1', fn=backup1)
    ai.define_model(name='backup2', fn=backup2)

    ctx = GenerateMiddlewareContext(ai=ai)
    fallback = Fallback(models=['backup1', 'backup2'])

    async def next_fn(_params: ModelHookParams, _ctx: GenerateMiddlewareContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    with pytest.raises(GenkitError, match='backup1 down'):
        await fallback.wrap_model(_make_params(), ctx, next_fn)

    assert ran == ['backup1']


@pytest.mark.asyncio
async def test_fallback_streams_chunks_from_the_fallback_model() -> None:
    """Test that fallback streams chunks emitted by the fallback model."""
    ai = Genkit()

    async def fail(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        raise GenkitError(status='UNAVAILABLE', message='primary down')

    async def backup(_request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('from-backup')]))
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )

    ai.define_model(name='primary', fn=fail)
    ai.define_model(name='backup', fn=backup)

    stream = ai.generate_stream(model='primary', prompt='hi', use=[Fallback(models=['backup'])])
    texts: list[str] = []
    async for chunk in stream.stream:
        texts.append(chunk.text)
    final = await stream.response

    assert 'from-backup' in ''.join(texts)
    assert final.text == 'done'
