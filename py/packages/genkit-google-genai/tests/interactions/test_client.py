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

"""Tests for the Interactions HTTP client."""

from __future__ import annotations

import json

import httpx
import pytest
from genkit_google_genai._interactions.client import (
    InteractionsClient,
    get_google_ai_url,
)
from genkit_google_genai._interactions.types import API_REVISION

from genkit import GenkitError
from genkit.plugin_api import GENKIT_CLIENT_HEADER


@pytest.mark.asyncio
async def test_create_interaction_posts_with_headers() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured['method'] = request.method
        captured['url'] = str(request.url)
        captured['headers'] = dict(request.headers)
        captured['body'] = json.loads(request.content.decode())
        return httpx.Response(200, json={'id': 'ix-1', 'status': 'in_progress'})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = InteractionsClient(api_key='test-key', http_client=http_client)
        result = await client.create_interaction({'input': [{'type': 'user_input', 'content': []}]})

    assert result == {'id': 'ix-1', 'status': 'in_progress'}
    assert captured['method'] == 'POST'
    assert captured['url'] == 'https://generativelanguage.googleapis.com/v1beta/interactions'
    headers = captured['headers']
    assert isinstance(headers, dict)
    assert headers.get('x-goog-api-key') == 'test-key'
    assert headers.get('api-revision') == API_REVISION
    assert GENKIT_CLIENT_HEADER in str(headers.get('x-goog-api-client', ''))


@pytest.mark.asyncio
async def test_get_interaction_uses_resource_url() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == 'GET'
        assert str(request.url).endswith('/v1beta/interactions/ix-42')
        return httpx.Response(200, json={'id': 'ix-42', 'status': 'completed'})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = InteractionsClient(api_key='test-key', http_client=http_client)
        result = await client.get_interaction('ix-42')

    assert result['status'] == 'completed'


@pytest.mark.asyncio
async def test_cancel_interaction_maps_success_to_cancelled() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == 'POST'
        assert str(request.url).endswith('/interactions/ix-9/cancel')
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = InteractionsClient(api_key='test-key', http_client=http_client)
        result = await client.cancel_interaction('ix-9')

    assert result == {'id': 'ix-9', 'status': 'cancelled'}


@pytest.mark.asyncio
async def test_error_mapping_includes_status_and_retry_after() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={'retry-after': '2'},
            json={'error': {'message': 'rate limited'}},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = InteractionsClient(api_key='test-key', http_client=http_client)
        with pytest.raises(GenkitError) as exc_info:
            await client.get_interaction('ix-1')

    error = exc_info.value
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert 'rate limited' in error.original_message
    assert error.response_metadata == {'retry_after_ms': 2000.0}


def test_get_google_ai_url_honors_client_options() -> None:
    url = get_google_ai_url(
        resource_path='interactions/foo',
        client_options={'api_version': 'v1', 'base_url': 'https://example.test'},
    )
    assert url == 'https://example.test/v1/interactions/foo'
