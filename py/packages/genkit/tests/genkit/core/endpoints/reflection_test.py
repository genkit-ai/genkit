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

"""Tests for the reflection API server.

This module contains unit tests for the ASGI-based reflection API server
which provides endpoints for inspecting and interacting with Genkit during
development.

Test coverage includes:
- Health check endpoint (/api/__health)
- Listing registered actions (/api/actions)
- Notification endpoint (/api/notify)
- Action execution with various scenarios (/api/runAction):
  - Standard action execution
  - Streaming action execution
  - Error handling when action not found
  - Context passing to actions

The tests use an ASGI client with mocked Registry to isolate and verify
each endpoint's behavior.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, Field

from genkit import Genkit
from genkit._core._action import ActionKind
from genkit._core._middleware import BaseMiddleware
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._registry import Registry
from genkit._core._typing import ActionMetadata


@pytest.fixture
def mock_registry() -> MagicMock:
    """Create a mock Registry for testing."""
    return MagicMock(spec=Registry)


@pytest_asyncio.fixture
async def asgi_client(mock_registry: MagicMock) -> AsyncIterator[AsyncClient]:
    """Create an ASGI test client with a mock registry.

    Args:
        mock_registry: A mock Registry object.

    Returns:
        An AsyncClient configured to make requests to the test ASGI app.
    """
    mock_registry.initialize_all_plugins = AsyncMock(return_value=None)
    mock_registry.list_actions = AsyncMock(return_value={})
    app = create_reflection_asgi_app(mock_registry)
    transport = ASGITransport(app=app)
    client = AsyncClient(transport=transport, base_url='http://test')
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_health_check(asgi_client: AsyncClient) -> None:
    """Test that the health check endpoint returns 200 OK."""
    response = await asgi_client.get('/api/__health')
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_actions(asgi_client: AsyncClient, mock_registry: MagicMock) -> None:
    """Test that the actions list endpoint returns registered actions."""

    async def mock_list_actions() -> dict[str, ActionMetadata]:
        return {
            '/custom/action1': ActionMetadata(
                key='/custom/action1',
                action_type=ActionKind.CUSTOM,
                name='action1',
            )
        }

    mock_registry.list_actions = mock_list_actions
    response = await asgi_client.get('/api/actions')
    assert response.status_code == 200
    result = response.json()
    assert '/custom/action1' in result
    assert result['/custom/action1']['name'] == 'action1'
    assert result['/custom/action1']['key'] == '/custom/action1'
    assert 'type' not in result['/custom/action1']
    assert 'actionType' not in result['/custom/action1']


@pytest.mark.asyncio
async def test_notify_endpoint(asgi_client: AsyncClient) -> None:
    """Test that the notify endpoint returns 200 OK."""
    response = await asgi_client.post('/api/notify')
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_run_action_not_found(asgi_client: AsyncClient, mock_registry: MagicMock) -> None:
    """Test that requesting a non-existent action returns a 404 error."""

    async def mock_resolve_action_by_key(key: str) -> None:
        return None

    mock_registry.resolve_action_by_key = mock_resolve_action_by_key
    response = await asgi_client.post(
        '/api/runAction',
        json={'key': 'non_existent_action', 'input': {'data': 'test'}},
    )
    assert response.status_code == 404
    assert 'error' in response.json()


@pytest.mark.asyncio
async def test_run_action_standard(asgi_client: AsyncClient, mock_registry: MagicMock) -> None:
    """Test that a standard (non-streaming) action works correctly."""
    mock_action = AsyncMock()
    mock_output = MagicMock()
    mock_output.response = {'result': 'success'}
    mock_output.trace_id = 'test_trace_id'
    mock_output.span_id = 'test_span_id'

    async def side_effect(
        input: object = None,
        on_chunk: object | None = None,
        context: object | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> MagicMock:
        if on_trace_start:
            await on_trace_start('test_trace_id', 'test_span_id')
        return mock_output

    mock_action.run.side_effect = side_effect

    async def mock_resolve_action_by_key(key: str) -> AsyncMock:
        return mock_action

    mock_registry.resolve_action_by_key = mock_resolve_action_by_key

    response = await asgi_client.post('/api/runAction', json={'key': 'test_action', 'input': {'data': 'test'}})

    assert response.status_code == 200
    response_data = response.json()
    assert 'result' in response_data
    assert 'telemetry' in response_data
    assert response_data['telemetry']['traceId'] == 'test_trace_id'
    assert response_data['telemetry']['spanId'] == 'test_span_id'
    assert response.headers['X-Genkit-Trace-Id'] == 'test_trace_id'
    assert response.headers['X-Genkit-Span-Id'] == 'test_span_id'
    mock_action.run.assert_called_once_with(
        input={'data': 'test'},
        context={},
        on_trace_start=ANY,
        on_chunk=None,
        telemetry_labels=None,
    )


@pytest.mark.asyncio
async def test_run_action_with_context(asgi_client: AsyncClient, mock_registry: MagicMock) -> None:
    """Test that an action with context works correctly."""
    mock_action = AsyncMock()
    mock_output = MagicMock()
    mock_output.response = {'result': 'success'}
    mock_output.trace_id = 'test_trace_id'
    mock_output.span_id = 'test_span_id'
    mock_action.run.return_value = mock_output

    async def mock_resolve_action_by_key(key: str) -> AsyncMock:
        return mock_action

    mock_registry.resolve_action_by_key = mock_resolve_action_by_key

    response = await asgi_client.post(
        '/api/runAction',
        json={
            'key': 'test_action',
            'input': {'data': 'test'},
            'context': {'user': 'test_user'},
        },
    )

    assert response.status_code == 200
    mock_action.run.assert_called_once_with(
        input={'data': 'test'},
        context={'user': 'test_user'},
        on_trace_start=ANY,
        on_chunk=None,
        telemetry_labels=None,
    )


@pytest.mark.asyncio
async def test_run_action_streaming(
    asgi_client: AsyncClient,
    mock_registry: MagicMock,
) -> None:
    """Test that streaming actions work correctly."""
    mock_action = AsyncMock()

    async def mock_streaming(
        input: object = None,
        on_chunk: object | None = None,
        context: object | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> MagicMock:
        if on_trace_start:
            await on_trace_start('stream_trace_id', 'stream_span_id')
        if on_chunk:
            on_chunk_fn = cast(Callable[[object], Awaitable[None]], on_chunk)
            await on_chunk_fn({'chunk': 1})
            await on_chunk_fn({'chunk': 2})
        mock_output = MagicMock()
        mock_output.response = {'final': 'result'}
        mock_output.trace_id = 'stream_trace_id'
        mock_output.span_id = 'stream_span_id'
        return mock_output

    mock_action.run.side_effect = mock_streaming
    mock_registry.resolve_action_by_key.return_value = mock_action

    response = await asgi_client.post(
        '/api/runAction?stream=true',
        json={'key': 'test_action', 'input': {'data': 'test'}},
    )

    assert response.status_code == 200
    assert response.headers['X-Genkit-Trace-Id'] == 'stream_trace_id'
    assert response.headers['X-Genkit-Span-Id'] == 'stream_span_id'


@pytest.mark.parametrize(
    'chunks, expected_lines',
    [
        (['string chunk 1', 'string chunk 2'], ['"string chunk 1"', '"string chunk 2"']),
        ([123, 456], ['123', '456']),
        ([12.3, 45.6], ['12.3', '45.6']),
        ([True, False], ['true', 'false']),
        ([None], ['null']),
        ([{'key': 'value'}], ['{"key": "value"}']),
    ],
)
@pytest.mark.asyncio
async def test_run_action_streaming_primitive_types(
    asgi_client: AsyncClient,
    mock_registry: MagicMock,
    chunks: list[Any],
    expected_lines: list[str],
) -> None:
    """Test that streaming actions with primitive type chunks work correctly."""
    mock_action = AsyncMock()

    async def mock_streaming(
        input: object = None,
        on_chunk: object | None = None,
        context: object | None = None,
        on_trace_start: Callable[[str, str], Awaitable[None]] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> MagicMock:
        if on_trace_start:
            await on_trace_start('stream_trace_id', 'stream_span_id')
        if on_chunk:
            on_chunk_fn = cast(Callable[[object], None], on_chunk)
            for chunk in chunks:
                on_chunk_fn(chunk)
        mock_output = MagicMock()
        mock_output.response = {'final': 'result'}
        mock_output.trace_id = 'stream_trace_id'
        mock_output.span_id = 'stream_span_id'
        return mock_output

    mock_action.run.side_effect = mock_streaming
    mock_registry.resolve_action_by_key.return_value = mock_action

    response = await asgi_client.post(
        '/api/runAction?stream=true',
        json={'key': 'test_action', 'input': {'data': 'test'}},
    )

    assert response.status_code == 200
    assert response.headers['X-Genkit-Trace-Id'] == 'stream_trace_id'
    assert response.headers['X-Genkit-Span-Id'] == 'stream_span_id'

    lines = response.text.strip().split('\n')
    assert lines[:-1] == expected_lines

    final_result = json.loads(lines[-1])
    assert final_result['result'] == {'final': 'result'}


# Real-registry tests for the /api/values?type=middleware endpoint. The other
# endpoint tests above use a MagicMock registry, but here we want to exercise
# the actual GenerateMiddleware serialization path the Dev UI consumes — mocking
# would defeat the point.


async def _registry_asgi_client(registry: Registry) -> AsyncClient:
    """Build an ASGI client wired to a real Registry instance."""
    app = create_reflection_asgi_app(registry)
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url='http://test')


@pytest.mark.asyncio
async def test_values_middleware_includes_derived_config_schema() -> None:
    """The Dev UI's /api/values?type=middleware response carries each middleware's configSchema.

    The schema is derived from the middleware class's pydantic fields by ``GenerateMiddleware(cls=...)``.
    """

    ai = Genkit()

    class _FallbackConfig(BaseModel):
        models: list[str] = Field(default_factory=list)
        statuses: list[str] = Field(default_factory=list)
        isolate_config: bool = False

    @ai.middleware(name='fallback', description='Falls back to alternative models on failure')
    class _Fallback(BaseMiddleware[_FallbackConfig]):
        pass

    client = await _registry_asgi_client(ai.registry)
    try:
        response = await client.get('/api/values?type=middleware')
        assert response.status_code == 200
        body = response.json()
        entry = body['fallback']
        assert entry['name'] == 'fallback'
        assert entry['description'] == 'Falls back to alternative models on failure'
        config_schema = entry['configSchema']
        assert config_schema['type'] == 'object'
        # Author-defined fields show up; framework-injected ones (registry,
        # custom_context / on_chunk) must not leak into the form.
        assert set(config_schema['properties'].keys()) == {'models', 'statuses', 'isolate_config'}
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_values_middleware_uses_class_docstring_as_description_fallback() -> None:
    """When no explicit description is passed, the class docstring is the fallback.

    Mirrors the action/tool convention: authors get a Dev-UI-visible description
    for free from a well-written docstring, with leading indentation cleaned up.
    """

    ai = Genkit()

    @ai.middleware(name='docstring_mw')
    class _DocMw(BaseMiddleware):
        """Logs every model call with a configurable prefix.

        Extra paragraphs end up in the description verbatim.
        """

    client = await _registry_asgi_client(ai.registry)
    try:
        response = await client.get('/api/values?type=middleware')
        assert response.status_code == 200
        entry = response.json()['docstring_mw']
        assert entry['description'] == (
            'Logs every model call with a configurable prefix.\n\nExtra paragraphs end up in the description verbatim.'
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_values_middleware_empty_config_schema_for_no_op() -> None:
    """A middleware with no config knobs still gets an (empty) object schema.

    The Dev UI renders an empty config form, signalling registered.
    """

    ai = Genkit()

    @ai.middleware(name='no_op')
    class _NoOp(BaseMiddleware):
        pass

    client = await _registry_asgi_client(ai.registry)
    try:
        response = await client.get('/api/values?type=middleware')
        assert response.status_code == 200
        entry = response.json()['no_op']
        assert entry['configSchema'] == {
            'type': 'object',
            'properties': {},
            'additionalProperties': True,
        }
    finally:
        await client.aclose()
