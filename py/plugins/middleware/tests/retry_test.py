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

import pytest
from pydantic import ValidationError

from genkit._core._error import GenkitError
from genkit._core._model import ModelHookParams, ModelRequest, ModelResponse
from genkit.plugins.middleware import Retry


@pytest.mark.asyncio
async def test_retry_success_on_first_attempt() -> None:
    """Test that successful calls pass through without retry."""
    retry = Retry(max_retries=3)

    async def next_fn(params):
        return ModelResponse(message=None)

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    result = await retry.wrap_model(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_retry_on_retryable_error() -> None:
    """Test that retryable errors trigger retry."""
    retry = Retry(max_retries=2, initial_delay_ms=10, jitter=False)

    call_count = 0

    async def next_fn(params):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise GenkitError(message='Service unavailable', status='UNAVAILABLE')
        return ModelResponse(message=None)

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    result = await retry.wrap_model(params, next_fn)
    assert result is not None
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted() -> None:
    """Test that errors are raised after max retries."""
    retry = Retry(max_retries=1, initial_delay_ms=10, jitter=False)

    async def next_fn(params) -> NoReturn:
        raise GenkitError(message='Service unavailable', status='UNAVAILABLE')

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    with pytest.raises(GenkitError):
        await retry.wrap_model(params, next_fn)


@pytest.mark.asyncio
async def test_retry_non_retryable_error() -> None:
    """Test that non-retryable errors fail immediately."""
    retry = Retry(max_retries=3)

    call_count = 0

    async def next_fn(params) -> NoReturn:
        nonlocal call_count
        call_count += 1
        raise GenkitError(message='Invalid argument', status='INVALID_ARGUMENT')

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    with pytest.raises(GenkitError):
        await retry.wrap_model(params, next_fn)
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
async def test_retry_non_genkit_error() -> None:
    """Test that non-GenkitError exceptions are retried."""
    retry = Retry(max_retries=2, initial_delay_ms=10, jitter=False)

    call_count = 0

    async def next_fn(params):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError('Network failure')
        return ModelResponse(message=None)

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    result = await retry.wrap_model(params, next_fn)
    assert result is not None
    assert call_count == 2
