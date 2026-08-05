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

"""Tests for Anthropic API error handling."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import genkit_anthropic.models as anthropic_models
import httpx
import pytest
from anthropic import APIConnectionError, APIError, APIStatusError
from genkit_anthropic.models import AnthropicModel

from genkit import GenkitError, Message, ModelRequest, Part, Role, TextPart
from genkit.plugin_api import StatusName

_ERROR_MESSAGE = 'Anthropic request failed'


def _request() -> ModelRequest:
    """Create a minimal model request."""
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])],
    )


def _http_request() -> httpx.Request:
    """Create the request required by Anthropic SDK errors."""
    return httpx.Request('POST', 'https://api.anthropic.com/v1/messages')


def _status_error(status_code: int, retry_after: str | None = None) -> APIStatusError:
    """Create a real Anthropic status error."""
    request = _http_request()
    headers = {'retry-after': retry_after} if retry_after is not None else None
    response = httpx.Response(status_code, request=request, headers=headers)
    return APIStatusError(_ERROR_MESSAGE, response=response, body={'type': 'error'})


def _model_failing_with(error: Exception) -> AnthropicModel:
    """Create a model whose non-streaming request raises an error."""
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=error)
    return AnthropicModel(model_name='claude-sonnet-4', client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status_code', 'expected_status'),
    [
        (400, 'INVALID_ARGUMENT'),
        (401, 'UNAUTHENTICATED'),
        (403, 'PERMISSION_DENIED'),
        (429, 'RESOURCE_EXHAUSTED'),
        (500, 'INTERNAL'),
        (503, 'UNAVAILABLE'),
        (529, 'UNAVAILABLE'),
        (404, 'UNKNOWN'),
    ],
)
async def test_generate_maps_anthropic_status_errors(status_code: int, expected_status: StatusName) -> None:
    """Map only the status codes supported by the JavaScript adapter."""
    api_error = _status_error(status_code)
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == expected_status
    assert error.original_message == _ERROR_MESSAGE
    assert error.cause is None
    assert error.__cause__ is api_error
    assert error.response_metadata is None
    assert error.to_callable_serializable().message == _ERROR_MESSAGE
    assert error.to_serializable().message == _ERROR_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'api_error',
    [
        APIError(_ERROR_MESSAGE, _http_request(), body=None),
        APIConnectionError(message=_ERROR_MESSAGE, request=_http_request()),
    ],
    ids=['base-api-error', 'connection-error'],
)
async def test_generate_maps_anthropic_errors_without_responses_to_unknown(api_error: APIError) -> None:
    """Anthropic errors without an HTTP response map to UNKNOWN."""
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == 'UNKNOWN'
    assert error.original_message == _ERROR_MESSAGE
    assert error.cause is None
    assert error.__cause__ is api_error
    assert error.response_metadata is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('status_code', 'expected_status'),
    [
        (429, 'RESOURCE_EXHAUSTED'),
        (503, 'UNAVAILABLE'),
        (529, 'UNAVAILABLE'),
    ],
)
async def test_generate_attaches_retry_after_metadata(status_code: int, expected_status: StatusName) -> None:
    """Attach parsed retry metadata for retryable Anthropic responses."""
    api_error = _status_error(status_code, retry_after='2.5')
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == expected_status
    assert error.response_metadata == {'retry_after_ms': 2500.0}
    assert error.cause is None
    assert error.__cause__ is api_error


@pytest.mark.asyncio
async def test_generate_leaves_non_anthropic_errors_untouched() -> None:
    """Do not wrap exceptions that were not raised by the Anthropic SDK."""
    provider_error = RuntimeError('unexpected failure')
    model = _model_failing_with(provider_error)

    with pytest.raises(RuntimeError) as exc_info:
        await model.generate(_request())

    assert exc_info.value is provider_error


class _FailingStreamManager:
    """Async stream manager that raises an Anthropic error on entry."""

    def __init__(self, error: APIError) -> None:
        self.error = error

    async def __aenter__(self) -> Any:  # noqa: ANN401
        raise self.error

    async def __aexit__(self, *args: object) -> None:
        return None


@pytest.mark.asyncio
async def test_generate_maps_streaming_anthropic_errors() -> None:
    """Apply the same mapping across the streaming context lifecycle."""
    api_error = _status_error(503, retry_after='1')
    client = MagicMock()
    client.messages.stream.return_value = _FailingStreamManager(api_error)
    model = AnthropicModel(model_name='claude-sonnet-4', client=client)
    ctx = MagicMock()
    ctx.is_streaming = True

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request(), ctx)

    error = exc_info.value
    assert error.status == 'UNAVAILABLE'
    assert error.response_metadata == {'retry_after_ms': 1000.0}
    assert error.cause is None
    assert error.__cause__ is api_error


@pytest.mark.parametrize(
    ('value', 'expected_ms'),
    [
        ('2', 2000.0),
        (' 1.5 ', 1500.0),
        ('0', 0.0),
    ],
)
def test_parse_retry_after_delay_seconds(value: str, expected_ms: float) -> None:
    """Parse whole, fractional, and zero delay-seconds values."""
    assert anthropic_models._parse_retry_after_ms(value) == expected_ms


@pytest.mark.parametrize('value', ['', '   ', 'not-a-delay'])
def test_parse_retry_after_rejects_blank_and_malformed_values(value: str) -> None:
    """Do not attach metadata for blank or malformed header values."""
    assert anthropic_models._parse_retry_after_ms(value) is None


@pytest.mark.parametrize('value', ['inf', 'Infinity', 'nan', '1e999', '1e307'])
def test_parse_retry_after_rejects_non_finite_delays(value: str) -> None:
    """Reject delays that are, or scale to, non-finite milliseconds."""
    assert anthropic_models._parse_retry_after_ms(value) is None


def test_parse_retry_after_future_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert a future HTTP-date to a relative millisecond delay."""
    monkeypatch.setattr(anthropic_models.time, 'time', lambda: 1_700_000_000.0)

    assert anthropic_models._parse_retry_after_ms('Tue, 14 Nov 2023 22:13:25 GMT') == 5000.0


def test_parse_retry_after_past_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp a past HTTP-date delay to zero."""
    monkeypatch.setattr(anthropic_models.time, 'time', lambda: 1_700_000_000.0)

    assert anthropic_models._parse_retry_after_ms('Tue, 14 Nov 2023 22:13:15 GMT') == 0.0


def test_parse_retry_after_returns_none_on_timestamp_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore platform timestamp failures for parseable dates."""
    retry_at = MagicMock()
    retry_at.timestamp.side_effect = OSError
    monkeypatch.setattr(anthropic_models, 'parsedate_to_datetime', lambda _: retry_at)

    assert anthropic_models._parse_retry_after_ms('Thu, 01 Jan 1601 00:00:00') is None


@pytest.mark.asyncio
@pytest.mark.parametrize('retry_after', [None, '', '   ', 'not-a-delay', 'inf', '1e999'])
async def test_generate_omits_invalid_retry_after_metadata(retry_after: str | None) -> None:
    """Leave response metadata unset when Retry-After cannot be parsed."""
    api_error = _status_error(429, retry_after=retry_after)
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    assert exc_info.value.response_metadata is None
