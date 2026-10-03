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

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._typing import FinishReason, Role
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
async def test_fallback_non_genkit_error(ctx) -> None:
    """Test that non-GenkitError exceptions fail immediately."""
    fallback = _make_fallback(models=['model2'])

    async def next_fn(params, ctx) -> NoReturn:
        raise ConnectionError('Network failure')

    with pytest.raises(ConnectionError):
        await fallback.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_fallback_streams_chunks_from_the_fallback_model() -> None:
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
