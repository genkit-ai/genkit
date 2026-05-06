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

from genkit._core._error import GenkitError
from genkit._core._model import ModelHookParams, ModelRequest, ModelResponse
from genkit.plugins.middleware import Fallback


@pytest.mark.asyncio
async def test_fallback_success_on_first_model() -> None:
    """Test that successful primary model calls pass through."""
    fallback = Fallback(models=['model2', 'model3'])

    async def next_fn(params):
        return ModelResponse(message=None)

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    result = await fallback.wrap_model(params, next_fn)
    assert result is not None


@pytest.mark.asyncio
async def test_fallback_on_retryable_error() -> None:
    """Test that retryable errors are classified correctly."""
    fallback = Fallback(models=['model2'])

    async def next_fn(params) -> NoReturn:
        raise GenkitError(message='Service unavailable', status='UNAVAILABLE')

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    with pytest.raises(GenkitError):
        await fallback.wrap_model(params, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_retryable_error() -> None:
    """Test that non-retryable errors fail immediately."""
    fallback = Fallback(models=['model2'])

    async def next_fn(params) -> NoReturn:
        raise GenkitError(message='Invalid argument', status='INVALID_ARGUMENT')

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    with pytest.raises(GenkitError):
        await fallback.wrap_model(params, next_fn)


@pytest.mark.asyncio
async def test_fallback_non_genkit_error() -> None:
    """Test that non-GenkitError exceptions fail immediately."""
    fallback = Fallback(models=['model2'])

    async def next_fn(params) -> NoReturn:
        raise ConnectionError('Network failure')

    params = ModelHookParams(request=ModelRequest(messages=[]), on_chunk=None, context={})
    with pytest.raises(ConnectionError):
        await fallback.wrap_model(params, next_fn)
