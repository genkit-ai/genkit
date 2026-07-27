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

"""Tests for Retry middleware."""

from typing import NoReturn
from unittest.mock import AsyncMock, patch

import pytest
from genkit_middleware import Retry
from pydantic import ValidationError

from genkit import ModelRequest, ModelResponse
from genkit._core._error import GenkitError
from genkit.middleware import GenerateMiddlewareContext, ModelHookParams


def _make_params() -> ModelHookParams:
    return ModelHookParams(request=ModelRequest(messages=[]))


@pytest.mark.asyncio
async def test_retry_success_on_first_attempt(ctx: GenerateMiddlewareContext) -> None:
    """Test that successful calls pass through without retry."""
    retry = Retry(max_retries=3)

    async def next_fn(params, ctx):
        return ModelResponse(message=None)

    result = await retry.wrap_model(_make_params(), ctx, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_retry_on_retryable_error(ctx: GenerateMiddlewareContext) -> None:
    """Test that retryable errors trigger retry."""
    retry = Retry(max_retries=2, initial_delay_ms=10, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise GenkitError(message='Service unavailable', status='UNAVAILABLE')
        return ModelResponse(message=None)

    result = await retry.wrap_model(_make_params(), ctx, next_fn)
    assert result is not None
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted(ctx: GenerateMiddlewareContext) -> None:
    """Test that errors are raised after max retries."""
    retry = Retry(max_retries=1, initial_delay_ms=10, no_jitter=True)

    async def next_fn(params, ctx) -> NoReturn:
        raise GenkitError(message='Service unavailable', status='UNAVAILABLE')

    with pytest.raises(GenkitError):
        await retry.wrap_model(_make_params(), ctx, next_fn)


@pytest.mark.asyncio
async def test_retry_non_retryable_error(ctx: GenerateMiddlewareContext) -> None:
    """Test that non-retryable errors fail immediately."""
    retry = Retry(max_retries=3)

    call_count = 0

    async def next_fn(params, ctx) -> NoReturn:
        nonlocal call_count
        call_count += 1
        raise GenkitError(message='Invalid argument', status='INVALID_ARGUMENT')

    with pytest.raises(GenkitError):
        await retry.wrap_model(_make_params(), ctx, next_fn)
    assert call_count == 1


def test_retry_rejects_negative_max_retries() -> None:
    """``max_retries`` must be non-negative; the wrap_model fall-through is unreachable.

    Regression: without the ``Field(ge=0)`` constraint, ``max_retries=-1`` would
    skip the for-loop entirely and trip the defensive ``AssertionError`` at the
    end of ``wrap_model``.
    """
    with pytest.raises(ValidationError):
        Retry(max_retries=-1)


@pytest.mark.asyncio
async def test_retry_non_genkit_error(ctx: GenerateMiddlewareContext) -> None:
    """Test that non-GenkitError exceptions are retried."""
    retry = Retry(max_retries=2, initial_delay_ms=10, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError('Network failure')
        return ModelResponse(message=None)

    result = await retry.wrap_model(_make_params(), ctx, next_fn)
    assert result is not None
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_after_is_delay_floor(ctx: GenerateMiddlewareContext) -> None:
    """Provider retry delay overrides a smaller local delay."""
    retry = Retry(max_retries=1, initial_delay_ms=100, max_delay_ms=10000, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': 5000},
            )
        return ModelResponse(message=None)

    with patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_local_delay_wins_when_larger_than_retry_after(ctx: GenerateMiddlewareContext) -> None:
    """Computed local delay is retained when it exceeds provider guidance."""
    retry = Retry(max_retries=1, initial_delay_ms=500, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': 10},
            )
        return ModelResponse(message=None)

    with patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_zero_retry_after_preserves_local_delay(ctx: GenerateMiddlewareContext) -> None:
    """A zero provider delay is handled as metadata while the local delay wins."""
    retry = Retry(max_retries=1, initial_delay_ms=100, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': 0},
            )
        return ModelResponse(message=None)

    with patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(0.1)


@pytest.mark.asyncio
async def test_retry_after_floor_is_applied_before_jitter(ctx: GenerateMiddlewareContext) -> None:
    """Apply jitter after the provider floor, matching the JavaScript middleware."""
    retry = Retry(max_retries=1, initial_delay_ms=100, max_delay_ms=10000)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': 5000},
            )
        return ModelResponse(message=None)

    with (
        patch('genkit_middleware._retry.random.random', return_value=0.5),
        patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep,
    ):
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(5.5)


@pytest.mark.asyncio
async def test_retry_after_is_capped_by_max_delay(ctx: GenerateMiddlewareContext) -> None:
    """A provider delay beyond the configured ceiling does not extend the wait."""
    retry = Retry(max_retries=1, initial_delay_ms=100, max_delay_ms=60000, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': 86_400_000},
            )
        return ModelResponse(message=None)

    with patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep:
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(60.0)


@pytest.mark.asyncio
@pytest.mark.parametrize('retry_after_ms', [0, 1])
async def test_small_retry_after_preserves_local_delay_cap(
    ctx: GenerateMiddlewareContext,
    retry_after_ms: float,
) -> None:
    """A small provider floor does not disable the configured local delay cap."""
    retry = Retry(max_retries=1, initial_delay_ms=100, max_delay_ms=100)

    call_count = 0

    async def next_fn(params, ctx):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenkitError(
                message='Rate limited',
                status='RESOURCE_EXHAUSTED',
                response_metadata={'retry_after_ms': retry_after_ms},
            )
        return ModelResponse(message=None)

    with (
        patch('genkit_middleware._retry.random.random', return_value=0.5),
        patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep,
    ):
        result = await retry.wrap_model(_make_params(), ctx, next_fn)

    assert result is not None
    assert call_count == 2
    sleep.assert_awaited_once_with(0.1)


@pytest.mark.asyncio
async def test_retry_does_not_retry_unauthenticated_error(ctx: GenerateMiddlewareContext) -> None:
    """Provider delay metadata does not make authentication errors retryable."""
    retry = Retry(max_retries=3, no_jitter=True)

    call_count = 0

    async def next_fn(params, ctx) -> NoReturn:
        nonlocal call_count
        call_count += 1
        raise GenkitError(
            message='Invalid API key',
            status='UNAUTHENTICATED',
            response_metadata={'retry_after_ms': 5000},
        )

    with (
        patch('genkit_middleware._retry.asyncio.sleep', new_callable=AsyncMock) as sleep,
        pytest.raises(GenkitError),
    ):
        await retry.wrap_model(_make_params(), ctx, next_fn)

    assert call_count == 1
    sleep.assert_not_awaited()
