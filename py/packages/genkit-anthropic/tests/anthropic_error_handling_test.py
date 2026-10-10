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


async def _generate_over_transport(handler: Any) -> GenkitError:  # noqa: ANN401
    """Run a non-streaming generate through the real SDK with a fake transport; return the raised error."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncAnthropic(api_key='test-key', http_client=http_client, max_retries=0)
    model = AnthropicModel(model_name='claude-sonnet-4-6', client=client)
    try:
        with pytest.raises(GenkitError) as exc_info:
            await model.generate(_request())
    finally:
        await http_client.aclose()
    return exc_info.value


@pytest.mark.asyncio
async def test_connection_refused_is_unavailable() -> None:
    """A refused connection is UNAVAILABLE so retry tries again."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError('[Errno 61] Connection refused', request=request)

    error = await _generate_over_transport(handler)

    assert error.status == 'UNAVAILABLE'
    assert isinstance(error.cause, APIConnectionError)
    assert error.__cause__ is error.cause
    assert get_http_status(error) == 500


@pytest.mark.asyncio
async def test_timeout_is_deadline_exceeded() -> None:
    """A request that times out is DEADLINE_EXCEEDED so retry tries again."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout('timed out', request=request)

    error = await _generate_over_transport(handler)

    assert error.status == 'DEADLINE_EXCEEDED'
    assert isinstance(error.cause, APITimeoutError)
    assert error.__cause__ is error.cause


@pytest.mark.asyncio
async def test_generate_marks_status_less_sdk_error_unknown() -> None:
    """An SDK error with no status, type, or transport failure is an UNKNOWN GenkitError."""
    api_error = APIError(_ERROR_MESSAGE, _http_request(), body=None)

    with pytest.raises(GenkitError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value.status == 'UNKNOWN'
    assert exc_info.value.original_message == _ERROR_MESSAGE
    assert exc_info.value.cause is api_error


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
async def test_generate_maps_billing_error_to_resource_exhausted() -> None:
    """A 402 billing_error is RESOURCE_EXHAUSTED, like OpenAI insufficient_quota, so Fallback can switch providers."""
    request = _http_request()
    api_error = APIStatusError(
        _ERROR_MESSAGE,
        response=httpx.Response(402, request=request),
        body={'type': 'error', 'error': {'type': 'billing_error', 'message': 'Your credit balance is too low'}},
    )

    with pytest.raises(GenkitError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value.status == 'RESOURCE_EXHAUSTED'
    assert exc_info.value.cause is api_error


@pytest.mark.asyncio
async def test_unmapped_4xx_is_unknown_genkit_error() -> None:
    """A 413 with no error type in the body is an UNKNOWN GenkitError that keeps the provider's message."""
    request = _http_request()
    api_error = APIStatusError(_ERROR_MESSAGE, response=httpx.Response(413, request=request), body=None)

    with pytest.raises(GenkitError) as exc_info:
        await _model_failing_with(api_error).generate(_request())

    assert exc_info.value.status == 'UNKNOWN'
    assert exc_info.value.original_message == _ERROR_MESSAGE
    assert exc_info.value.cause is api_error
    assert exc_info.value.__cause__ is api_error


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
async def test_429_retry_after_header_sets_retry_delay() -> None:
    """A 429 with a Retry-After header puts the delay in response_metadata['retry_after_ms']."""
    request = _http_request()
    response = httpx.Response(429, request=request, headers={'Retry-After': '2.5'})
    api_error = APIStatusError(_ERROR_MESSAGE, response=response, body={'type': 'error'})
    model = _model_failing_with(api_error)

    with pytest.raises(GenkitError) as exc_info:
        await model.generate(_request())

    error = exc_info.value
    assert error.status == 'RESOURCE_EXHAUSTED'
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
        ('billing_error', 'RESOURCE_EXHAUSTED'),
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
async def test_streaming_in_band_error_event_with_unknown_type_is_unknown() -> None:
    """A mid-stream error type the plugin does not know is an UNKNOWN GenkitError with the event's message."""
    _, error = await _stream_generate(_stream_failing_after_first_token({'type': 'brand_new_error', 'message': 'x'}))

    assert isinstance(error, GenkitError)
    assert error.status == 'UNKNOWN'
    assert error.original_message == 'x'
    assert isinstance(error.cause, APIStatusError)


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
