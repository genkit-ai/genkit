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

"""Tests for reflection API authentication and dev-only endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._reflection_config import REFLECTION_SECRET_HEADER
from genkit._core._registry import Registry

SECRET = 'test-secret'


def _registry() -> MagicMock:
    registry = MagicMock(spec=Registry)
    registry.initialize_all_plugins = AsyncMock(return_value=None)
    registry.list_actions = AsyncMock(return_value={})
    return registry


async def _client(secret: str | None) -> AsyncClient:
    app = create_reflection_asgi_app(_registry(), secret=secret)
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://test')


@pytest_asyncio.fixture
async def secured() -> AsyncIterator[AsyncClient]:
    client = await _client(SECRET)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def open_client() -> AsyncIterator[AsyncClient]:
    client = await _client(None)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_rejects_request_with_no_secret(secured: AsyncClient) -> None:
    response = await secured.get('/api/actions')
    assert response.status_code == 401
    assert response.content == b''


@pytest.mark.asyncio
async def test_rejects_request_with_wrong_secret(secured: AsyncClient) -> None:
    response = await secured.get('/api/actions', headers={REFLECTION_SECRET_HEADER: 'nope'})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_accepts_request_with_right_secret(secured: AsyncClient) -> None:
    response = await secured.get('/api/actions', headers={REFLECTION_SECRET_HEADER: SECRET})
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_is_exempt(secured: AsyncClient) -> None:
    response = await secured.get('/api/__health')
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_no_secret_configured_requires_nothing(open_client: AsyncClient) -> None:
    response = await open_client.get('/api/actions')
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_quitquitquit_is_absent_outside_dev(open_client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('GENKIT_ENV', raising=False)
    client = await _client(None)
    try:
        response = await client.get('/api/__quitquitquit')
        assert response.status_code == 404
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_quitquitquit_is_present_in_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GENKIT_ENV', 'dev')
    # Only the route's presence matters; the handler kills the process, so it
    # is never actually invoked here.
    app = create_reflection_asgi_app(_registry())
    paths = {route.path for route in app.routes if hasattr(route, 'path')}
    assert '/api/__quitquitquit' in paths
