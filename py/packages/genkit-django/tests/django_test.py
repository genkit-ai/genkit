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


"""Tests for the Django plugin."""

import json
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest
from django.test import AsyncClient
from django.test.utils import override_settings
from django.urls import path
from genkit_django import genkit_django_handler

from genkit import ActionRunContext, Genkit, GenkitError, PublicError, RequestData


def _assert_is_error_response(parsed: dict) -> None:
    """Assert parsed dict has a callable error body (message + status)."""
    assert isinstance(parsed, dict)
    assert all(k in parsed for k in ('message', 'status'))
    assert 'stack' not in parsed.get('details', {})


def _sse_error_event(chunks: list[bytes]) -> dict:
    """Return the first SSE ``error`` payload, or fail."""
    text = b''.join(chunks).decode()
    for line in text.splitlines():
        if line.startswith('data: '):
            payload = json.loads(line[6:])
            if 'error' in payload:
                return payload['error']
    raise AssertionError(f'no SSE error event in {text!r}')


def _build_views() -> dict[str, Any]:
    """Build the Django views used by the integration tests."""
    ai = Genkit()

    async def my_context_provider(request_data: RequestData) -> dict[str, Any]:
        """Provide a context for the flow."""
        return {'username': request_data.headers.get('authorization')}

    @genkit_django_handler(ai, context_provider=my_context_provider)
    @ai.flow()
    async def say_hi(name: str, ctx: ActionRunContext) -> dict[str, str]:
        ctx.send_chunk(1)
        ctx.send_chunk({'username': ctx.context.get('username')})
        ctx.send_chunk({'foo': 'bar'})
        return {'bar': 'baz'}

    @genkit_django_handler(ai)
    @ai.flow()
    async def raise_error(_: str) -> None:
        raise ValueError('Intentional test error')

    @genkit_django_handler(ai)
    @ai.flow()
    async def raise_invalid(_: str) -> None:
        raise GenkitError(status='INVALID_ARGUMENT', message='bad id 12345')

    @genkit_django_handler(ai)
    @ai.flow()
    async def raise_public(_: str) -> None:
        raise PublicError('NOT_FOUND', 'no order 99')

    async def echo_context(request_data: RequestData) -> dict[str, Any]:
        return {
            'method': request_data.method,
            'authorization': request_data.headers['authorization'],
            'input': request_data.input,
        }

    @genkit_django_handler(ai, context_provider=echo_context)
    @ai.flow()
    async def echo_request(_: str, ctx: ActionRunContext) -> dict[str, Any]:
        return {
            'method': ctx.context['method'],
            'authorization': ctx.context['authorization'],
            'input': ctx.context['input'],
        }

    return {
        'say_hi': say_hi,
        'raise_error': raise_error,
        'raise_invalid': raise_invalid,
        'raise_public': raise_public,
        'echo_request': echo_request,
    }


@pytest.fixture
def urlconf(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install a temporary URLconf module that mounts the test views."""
    views = _build_views()

    module = types.ModuleType('genkit_django_tests_urls')
    # Django reads urlpatterns off the module object; type checkers can't see
    # this attribute on `types.ModuleType`, hence the suppressions.
    module.urlpatterns = [  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]
        path('chat', views['say_hi']),
        path('error_flow', views['raise_error']),
        path('invalid_flow', views['raise_invalid']),
        path('public_flow', views['raise_public']),
        path('echo_request', views['echo_request']),
    ]
    monkeypatch.setitem(sys.modules, 'genkit_django_tests_urls', module)

    with override_settings(ROOT_URLCONF='genkit_django_tests_urls'):
        yield


@pytest.mark.asyncio
async def test_simple_post(urlconf: None) -> None:  # noqa: ARG001
    """A POST with the {data: ...} envelope returns {result: ...}."""
    client = AsyncClient()
    response = await client.post(
        '/chat',
        data=json.dumps({'data': 'banana'}),
        content_type='application/json',
        headers={'authorization': 'Pavel'},
    )

    assert response.status_code == 200
    assert json.loads(response.content) == {'result': {'bar': 'baz'}}


@pytest.mark.asyncio
async def test_streaming(urlconf: None) -> None:  # noqa: ARG001
    """A POST with Accept: text/event-stream streams chunks then result."""
    client = AsyncClient()
    response = await client.post(
        '/chat',
        data=json.dumps({'data': 'banana'}),
        content_type='application/json',
        headers={
            'authorization': 'Pavel',
            'accept': 'text/event-stream',
        },
    )

    assert response.status_code == 200
    assert response['Content-Type'].startswith('text/event-stream')

    chunks = [chunk async for chunk in response.streaming_content]

    assert chunks == [
        b'data: {"message":1}\n\n',
        b'data: {"message":{"username":"Pavel"}}\n\n',
        b'data: {"message":{"foo":"bar"}}\n\n',
        b'data: {"result":{"bar":"baz"}}\n\n',
    ]


@pytest.mark.asyncio
async def test_400_missing_data_returns_valid_json(urlconf: None) -> None:  # noqa: ARG001
    """400 (missing data) must return valid JSON."""
    client = AsyncClient()
    response = await client.post(
        '/chat',
        data=json.dumps({}),  # no 'data' key
        content_type='application/json',
    )
    assert response.status_code == 400
    _assert_is_error_response(json.loads(response.content))


@pytest.mark.asyncio
async def test_400_invalid_json_returns_valid_json(urlconf: None) -> None:  # noqa: ARG001
    """400 (malformed body) must return valid JSON, not crash."""
    client = AsyncClient()
    response = await client.post(
        '/chat',
        data='not json',
        content_type='application/json',
    )
    assert response.status_code == 400
    _assert_is_error_response(json.loads(response.content))


@pytest.mark.asyncio
async def test_405_non_post_returns_valid_json(urlconf: None) -> None:  # noqa: ARG001
    """GET (or any non-POST) must return 405 with valid JSON, not a Django default."""
    client = AsyncClient()
    response = await client.get('/chat')
    assert response.status_code == 405
    _assert_is_error_response(json.loads(response.content))


@pytest.mark.asyncio
async def test_500_flow_exception_returns_valid_json(urlconf: None) -> None:  # noqa: ARG001
    """500 (flow exception) must return valid JSON in HttpErrorWireFormat shape."""
    client = AsyncClient()
    code_snippet = 'query = f"SELECT * FROM users WHERE id={user_input}"'
    response = await client.post(
        '/error_flow',
        data=json.dumps({'data': code_snippet}),
        content_type='application/json',
    )
    assert response.status_code == 500
    parsed = json.loads(response.content)
    _assert_is_error_response(parsed)
    assert parsed == {'message': 'Internal Error', 'status': 'INTERNAL'}


@pytest.mark.asyncio
async def test_django_flow_raising_invalid_argument_returns_400_with_generic_message(
    urlconf: None,
) -> None:  # noqa: ARG001
    """Django POST to a flow that raises GenkitError INVALID_ARGUMENT returns 400 'Invalid argument', not its text."""
    client = AsyncClient()
    response = await client.post(
        '/invalid_flow',
        data=json.dumps({'data': 'x'}),
        content_type='application/json',
    )
    assert response.status_code == 400
    assert json.loads(response.content) == {'message': 'Invalid argument', 'status': 'INVALID_ARGUMENT'}
    assert b'12345' not in response.content


@pytest.mark.asyncio
async def test_django_flow_raising_public_error_returns_its_status_and_message(
    urlconf: None,
) -> None:  # noqa: ARG001
    """Django POST to a flow that raises PublicError NOT_FOUND returns 404 with that message."""
    client = AsyncClient()
    response = await client.post(
        '/public_flow',
        data=json.dumps({'data': '99'}),
        content_type='application/json',
    )
    assert response.status_code == 404
    assert json.loads(response.content) == {'message': 'no order 99', 'status': 'NOT_FOUND'}


@pytest.mark.asyncio
async def test_django_flow_raising_value_error_returns_500_internal_error_without_stack(
    urlconf: None,
) -> None:  # noqa: ARG001
    """Django POST to a flow that raises ValueError returns a generic 500."""
    client = AsyncClient()
    response = await client.post(
        '/error_flow',
        data=json.dumps({'data': 'secret'}),
        content_type='application/json',
    )
    assert response.status_code == 500
    body = json.loads(response.content)
    assert body == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'secret' not in response.content
    assert 'stack' not in body


@pytest.mark.asyncio
async def test_django_stream_flow_raising_value_error_sends_sse_internal_error_without_stack(
    urlconf: None,
) -> None:  # noqa: ARG001
    """Django SSE to a flow that raises ValueError sends a generic Internal Error."""
    client = AsyncClient()
    response = await client.post(
        '/error_flow',
        data=json.dumps({'data': 'secret'}),
        content_type='application/json',
        headers={'accept': 'text/event-stream'},
    )
    chunks = [chunk async for chunk in response.streaming_content]
    error = _sse_error_event(chunks)
    assert error == {'message': 'Internal Error', 'status': 'INTERNAL'}
    assert b'secret' not in b''.join(chunks)
    assert 'stack' not in error


@pytest.mark.asyncio
async def test_django_context_provider_sees_method_lowercase_headers_and_input(
    urlconf: None,
) -> None:  # noqa: ARG001
    """Django context_provider sees method, lowercase headers, and input."""
    client = AsyncClient()
    response = await client.post(
        '/echo_request',
        data=json.dumps({'data': 'hello'}),
        content_type='application/json',
        headers={'Authorization': 'Bearer tok'},
    )
    assert response.status_code == 200
    assert json.loads(response.content) == {
        'result': {
            'method': 'POST',
            'authorization': 'Bearer tok',
            'input': 'hello',
        }
    }
