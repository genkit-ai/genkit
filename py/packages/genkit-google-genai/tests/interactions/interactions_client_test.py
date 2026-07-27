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

"""Tests for the raw HTTP Interactions client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from genkit_google_genai._interactions import client as interactions_client
from genkit_google_genai._interactions.client import (
    API_REVISION,
    cancel_interaction,
    create_interaction,
    get_interaction,
    google_ai_url,
    headers,
    timeout_seconds,
)
from genkit_google_genai._interactions.options import ClientOptions
from google.genai.interactions import Interaction

from genkit import GenkitError
from genkit.plugin_api import GENKIT_CLIENT_HEADER


def test_google_ai_url_defaults() -> None:
    assert google_ai_url('interactions') == ('https://generativelanguage.googleapis.com/v1beta/interactions')


def test_google_ai_url_respects_overrides() -> None:
    url = google_ai_url(
        'interactions/abc',
        client_options=ClientOptions(base_url='https://example.test/', api_version='v1alpha'),
    )
    assert url == 'https://example.test/v1alpha/interactions/abc'


def test_headers_set_api_revision_and_strip_reserved_custom() -> None:
    result = headers(
        api_key='k',
        client_options=ClientOptions(
            custom_headers={
                'x-custom': '1',
                'x-goog-api-key': 'stolen',
                'x-goog-api-client': 'wrapper',
            }
        ),
    )
    assert result['Api-Revision'] == API_REVISION
    assert result['x-goog-api-key'] == 'k'
    assert result['x-goog-api-client'] == GENKIT_CLIENT_HEADER
    assert result['x-custom'] == '1'


def test_timeout_seconds_converts_milliseconds() -> None:
    assert timeout_seconds(ClientOptions(timeout=1500)) == 1.5
    assert timeout_seconds(ClientOptions()) is None


def mock_response(
    *,
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
    text: str = '',
    headers_map: dict[str, str] | None = None,
    content: bytes | None = None,
) -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, headers=headers_map or {})
    if content is not None:
        return httpx.Response(status_code, content=content, headers=headers_map or {})
    return httpx.Response(status_code, text=text, headers=headers_map or {})


@pytest.fixture
def http_client() -> MagicMock:
    client = MagicMock()
    client.request = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_create_interaction_posts_body(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        json_body={'id': 'ix-1', 'status': 'in_progress'},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        result = await create_interaction(
            'key',
            {'agent': 'deep-research', 'background': True},
            ClientOptions(base_url='https://example.test'),
        )

    assert isinstance(result, Interaction)
    assert result.id == 'ix-1'
    method, url = http_client.request.call_args.args[:2]
    assert method == 'POST'
    assert url == 'https://example.test/v1beta/interactions'
    kwargs = http_client.request.call_args.kwargs
    assert kwargs['json'] == {'agent': 'deep-research', 'background': True}
    assert kwargs['headers']['Api-Revision'] == API_REVISION
    assert kwargs['headers']['x-goog-api-key'] == 'key'


@pytest.mark.asyncio
async def test_get_interaction_gets_by_id(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        json_body={'id': 'ix-9', 'status': 'completed', 'steps': []},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        result = await get_interaction('key', 'ix-9')

    assert result.id == 'ix-9'
    method, url = http_client.request.call_args.args[:2]
    assert method == 'GET'
    assert url.endswith('/interactions/ix-9')


@pytest.mark.asyncio
async def test_cancel_interaction_normalizes_http_cancelled(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        status_code=499,
        json_body={'error': {'message': 'cancelled'}},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        result = await cancel_interaction('key', 'ix-cancel')

    assert result.id == 'ix-cancel'
    assert result.status == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_interaction_normalizes_success_as_cancelled(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        json_body={'id': 'ix-cancel', 'status': 'completed'},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        result = await cancel_interaction('key', 'ix-cancel')

    assert result.status == 'cancelled'


@pytest.mark.asyncio
async def test_cancel_interaction_rethrows_non_cancelled_errors(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        status_code=404,
        json_body={'error': {'message': 'missing'}},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        with pytest.raises(GenkitError) as exc_info:
            await cancel_interaction('key', 'ix-missing')
    assert exc_info.value.status == 'NOT_FOUND'


@pytest.mark.asyncio
async def test_rate_limit_includes_retry_after_ms(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        status_code=429,
        json_body={'error': {'message': 'slow down'}},
        headers_map={'retry-after': '1.5'},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        with pytest.raises(GenkitError) as exc_info:
            await create_interaction('key', {'model': 'lyria'})
    assert exc_info.value.status == 'RESOURCE_EXHAUSTED'
    assert exc_info.value.response_metadata is not None
    assert exc_info.value.response_metadata.get('retry_after_ms') == 1500.0


@pytest.mark.asyncio
async def test_empty_success_body_is_internal_error(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(content=b'')
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        with pytest.raises(GenkitError, match='empty response') as exc_info:
            await get_interaction('key', 'ix-1')
    assert exc_info.value.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_timeout_maps_to_deadline_exceeded(http_client: MagicMock) -> None:
    http_client.request.side_effect = httpx.TimeoutException('late')
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client):
        with pytest.raises(GenkitError) as exc_info:
            await create_interaction('key', {'model': 'lyria'}, ClientOptions(timeout=1000))
    assert exc_info.value.status == 'DEADLINE_EXCEEDED'


@pytest.mark.asyncio
async def test_request_timeout_converted_from_ms(http_client: MagicMock) -> None:
    http_client.request.return_value = mock_response(
        json_body={'id': 'ix-1', 'status': 'completed', 'steps': []},
    )
    with patch.object(interactions_client, 'get_cached_client', return_value=http_client) as cached:
        await create_interaction('key', {'model': 'lyria'}, ClientOptions(timeout=2500))

    assert cached.call_args.kwargs['timeout'] == interactions_client.NO_TIMEOUT
    assert http_client.request.call_args.kwargs['timeout'] == 2.5
