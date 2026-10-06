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

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
)
from genkit_anthropic._models import AnthropicModel

from genkit import GenkitError, Message, Part, Role
from genkit._core._error import get_callable_json, get_http_status
from genkit.model import ModelRequest
from genkit.plugin_api import StatusName

_ERROR_MESSAGE = 'Anthropic request failed'


def _request() -> ModelRequest:
    """Create a minimal model request."""
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Hello')])],
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
        (404, 'NOT_FOUND'),
    ],
)
async def test_generate_maps_anthropic_status_errors(status_code: int, expected_status: StatusName) -> None:
    """HTTP codes go through the shared map; 529 stays overloaded/unavailable."""
    api_error = _status_error(status_code)
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == expected_status
    assert error.original_message == _ERROR_MESSAGE
    assert error.cause is api_error
    assert error.__cause__ is api_error
    assert error.response_metadata is None


@pytest.mark.asyncio
async def test_anthropic_api_error_is_served_as_internal_error() -> None:
    """An Anthropic 401 stays UNAUTHENTICATED in-process and serves as 500 Internal Error."""
    api_error = _status_error(401)
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == 'UNAUTHENTICATED'
    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert get_http_status(error) == 500
    assert _ERROR_MESSAGE not in str(get_callable_json(error))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'api_error',
    [
        APIError(_ERROR_MESSAGE, _http_request(), body=None),
        APIConnectionError(message=_ERROR_MESSAGE, request=_http_request()),
        APITimeoutError(request=_http_request()),
    ],
    ids=['base-api-error', 'connection-error', 'timeout'],
)
async def test_generate_leaves_errors_without_a_status_raw(api_error: APIError) -> None:
    """A transport failure has no status to report, so it reaches the caller unchanged."""
    model = _model_failing_with(api_error)

    with pytest.raises(APIError) as exc_info:
        await model.generate(_request())

    assert exc_info.value is api_error
    assert not isinstance(exc_info.value, GenkitError)


@pytest.mark.asyncio
async def test_generate_reads_error_type_when_http_status_is_unmapped() -> None:
    """A 413 has no status mapping; the body's request_too_large decides it."""
    request = _http_request()
    response = httpx.Response(413, request=request)
    api_error = APIStatusError(
        _ERROR_MESSAGE,
        response=response,
        body={'type': 'error', 'error': {'type': 'request_too_large', 'message': 'Request exceeds the maximum size'}},
    )

    with pytest.raises(GenkitError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.original_message == _ERROR_MESSAGE
    assert exc_info.value.cause is api_error


@pytest.mark.asyncio
async def test_generate_leaves_unmapped_status_without_known_type_raw() -> None:
    """An unmapped 4xx with no known error type has no real status, so it stays raw."""
    request = _http_request()
    api_error = APIStatusError(_ERROR_MESSAGE, response=httpx.Response(418, request=request), body=None)

    with pytest.raises(APIStatusError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value is api_error


@pytest.mark.asyncio
async def test_generate_marks_unreadable_response_internal() -> None:
    """A 200 whose body fails SDK validation is a malformed provider reply."""
    request = _http_request()
    api_error = APIResponseValidationError(httpx.Response(200, request=request), body={'unexpected': True})

    with pytest.raises(GenkitError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value.status == 'INTERNAL'
    assert exc_info.value.cause is api_error


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
    assert error.cause is api_error
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
    assert error.cause is api_error
    assert error.__cause__ is api_error


@pytest.mark.asyncio
@pytest.mark.parametrize('retry_after', [None, '', '   ', 'not-a-delay', 'inf', '1e999'])
async def test_generate_omits_invalid_retry_after_metadata(retry_after: str | None) -> None:
    """Leave response metadata unset when Retry-After cannot be parsed."""
    api_error = _status_error(429, retry_after=retry_after)
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    assert exc_info.value.response_metadata is None


def _sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


def _stream_failing_after_first_token(error: dict[str, Any]) -> str:
    """A 200 stream that emits one text token, then an in-band error event."""
    return ''.join([
        _sse(
            'message_start',
            {
                'type': 'message_start',
                'message': {
                    'id': 'msg_01',
                    'type': 'message',
                    'role': 'assistant',
                    'model': 'claude-sonnet-4-6',
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {'input_tokens': 12, 'output_tokens': 1},
                },
            },
        ),
        _sse(
            'content_block_start',
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        ),
        _sse(
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Smoked'}},
        ),
        _sse('error', {'type': 'error', 'error': error}),
    ])


async def _stream_generate(sse_body: str) -> tuple[list[str], BaseException]:
    """Stream through the real Anthropic SDK against a canned SSE body; return chunks and the error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=sse_body.encode())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAnthropic(api_key='test-key', http_client=http_client, max_retries=0)
    model = AnthropicModel(model_name='claude-sonnet-4-6', client=client)
    chunks: list[str] = []
    ctx = MagicMock()
    ctx.is_streaming = True
    ctx.send_chunk.side_effect = lambda chunk: chunks.append(chunk.text)

    with pytest.raises(Exception) as exc_info:
        await model.generate(_request(), ctx)
    await http_client.aclose()
    return chunks, exc_info.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error_type', 'expected_status'),
    [
        ('overloaded_error', 'UNAVAILABLE'),
        ('api_error', 'INTERNAL'),
        ('rate_limit_error', 'RESOURCE_EXHAUSTED'),
        ('invalid_request_error', 'INVALID_ARGUMENT'),
    ],
)
async def test_streaming_in_band_error_event_maps_its_type(error_type: str, expected_status: StatusName) -> None:
    """A mid-stream `error` event after a 200 is classified by its type, not by the 200."""
    chunks, error = await _stream_generate(
        _stream_failing_after_first_token({'type': error_type, 'message': 'Overloaded'})
    )

    assert chunks == ['Smoked']
    assert isinstance(error, GenkitError)
    assert error.status == expected_status
    assert error.original_message == 'Overloaded'
    assert isinstance(error.cause, APIStatusError)
    assert error.cause.status_code == 200


@pytest.mark.asyncio
async def test_streaming_in_band_error_event_with_unknown_type_stays_raw() -> None:
    """An error type the plugin does not know has no real status, so the SDK error escapes unchanged."""
    _, error = await _stream_generate(_stream_failing_after_first_token({'type': 'brand_new_error', 'message': 'x'}))

    assert isinstance(error, APIStatusError)
    assert not isinstance(error, GenkitError)


def _model_never_called() -> tuple[AnthropicModel, MagicMock]:
    client = MagicMock()
    client.messages.create = AsyncMock()
    return AnthropicModel(model_name='claude-sonnet-4', client=client), client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('part', 'message_fragment'),
    [
        (Part.from_reasoning('The guest listed a peanut allergy.'), 'require a signature'),
        (Part.from_media(url='data:image/png;base64', content_type='image/png'), 'not enough values to unpack'),
    ],
    ids=['unsigned-thinking', 'data-uri-without-payload'],
)
async def test_generate_marks_unsendable_history_invalid_argument(part: Part, message_fragment: str) -> None:
    """History the plugin cannot convert is caller input; retry must not resend it."""
    model, client = _model_never_called()
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part.from_text('Is the satay safe for me?')]),
            Message(role=Role.MODEL, content=[part]),
            Message(role=Role.USER, content=[Part.from_text('And the dessert?')]),
        ],
    )

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert message_fragment in exc_info.value.original_message
    assert isinstance(exc_info.value.cause, ValueError)
    assert exc_info.value.__cause__ is exc_info.value.cause
    client.messages.create.assert_not_called()


@pytest.mark.asyncio
async def test_generate_marks_invalid_thinking_budget_invalid_argument() -> None:
    """A non-integer thinking budget is rejected before any request is sent."""
    model, client = _model_never_called()
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Plan a gluten-free menu.')])],
        config={'thinking': {'enabled': True, 'budgetTokens': 2048.5}},
    )

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert isinstance(exc_info.value.cause, ValueError)
    client.messages.create.assert_not_called()
