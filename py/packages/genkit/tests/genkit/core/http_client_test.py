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

"""Tests for genkit.core.http_client module.

This module tests the per-event-loop HTTP client caching functionality,
which is critical for avoiding "bound to different event loop" errors
when using httpx.AsyncClient in async code.

Test Categories:
    - Client caching: Verify clients are cached and reused per event loop
    - Cache key isolation: Different cache keys get different clients
    - Closed client handling: Closed clients are replaced
    - Cleanup utilities: close_cached_clients and clear_client_cache work
    - Configuration: Headers and timeouts are applied correctly
"""

import asyncio

import httpx
import pytest

from genkit.core.http_client import (
    _loop_clients,
    clear_client_cache,
    close_cached_clients,
    get_cached_client,
)


class TestGetCachedClient:
    """Tests for get_cached_client function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear the client cache before each test."""
        clear_client_cache()

    @pytest.mark.asyncio
    async def test_returns_httpx_async_client(self) -> None:
        """Verify get_cached_client returns an httpx.AsyncClient instance."""
        client = get_cached_client(cache_key='test')

        assert isinstance(client, httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_client_cached_per_event_loop(self) -> None:
        """Verify the same client is returned for the same cache key in same loop."""
        client1 = get_cached_client(cache_key='test')
        client2 = get_cached_client(cache_key='test')

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_different_cache_keys_get_different_clients(self) -> None:
        """Verify different cache keys return different client instances."""
        client1 = get_cached_client(cache_key='plugin-a')
        client2 = get_cached_client(cache_key='plugin-b')

        assert client1 is not client2

    @pytest.mark.asyncio
    async def test_client_has_correct_headers(self) -> None:
        """Verify client is configured with provided headers."""
        headers = {
            'Authorization': 'Bearer test-token',
            'X-Custom-Header': 'custom-value',
        }
        client = get_cached_client(cache_key='test', headers=headers)

        # Check headers are set on the client
        assert client.headers.get('Authorization') == 'Bearer test-token'
        assert client.headers.get('X-Custom-Header') == 'custom-value'

    @pytest.mark.asyncio
    async def test_client_has_correct_timeout_float(self) -> None:
        """Verify client is configured with float timeout."""
        client = get_cached_client(cache_key='test', timeout=30.0)

        assert client.timeout.read == 30.0
        assert client.timeout.connect == 30.0

    @pytest.mark.asyncio
    async def test_client_has_correct_timeout_object(self) -> None:
        """Verify client is configured with httpx.Timeout object."""
        timeout = httpx.Timeout(60.0, connect=10.0)
        client = get_cached_client(cache_key='test', timeout=timeout)

        assert client.timeout.read == 60.0
        assert client.timeout.connect == 10.0

    @pytest.mark.asyncio
    async def test_default_timeout_applied(self) -> None:
        """Verify default timeout is applied when not specified."""
        client = get_cached_client(cache_key='test')

        # Default is 60s total, 10s connect
        assert client.timeout.read == 60.0
        assert client.timeout.connect == 10.0

    @pytest.mark.asyncio
    async def test_closed_client_gets_replaced(self) -> None:
        """Verify a closed client is replaced with a new one."""
        client1 = get_cached_client(cache_key='test')
        await client1.aclose()

        assert client1.is_closed

        client2 = get_cached_client(cache_key='test')

        assert client2 is not client1
        assert not client2.is_closed

    @pytest.mark.asyncio
    async def test_client_stored_in_cache(self) -> None:
        """Verify client is stored in the module-level cache."""
        clear_client_cache()

        _ = get_cached_client(cache_key='test')

        loop = asyncio.get_running_loop()
        assert loop in _loop_clients
        assert 'test' in _loop_clients[loop]

    def test_raises_without_running_event_loop(self) -> None:
        """Verify RuntimeError is raised when called outside async context."""
        with pytest.raises(RuntimeError, match='must be called from within an async context'):
            get_cached_client(cache_key='test')


class TestCloseCachedClients:
    """Tests for close_cached_clients function."""

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear the client cache before each test."""
        clear_client_cache()

    @pytest.mark.asyncio
    async def test_close_specific_client(self) -> None:
        """Verify closing a specific client by cache key."""
        _ = get_cached_client(cache_key='to-close')
        _ = get_cached_client(cache_key='keep')

        await close_cached_clients('to-close')

        # The closed client should be removed from cache
        loop = asyncio.get_running_loop()
        assert 'to-close' not in _loop_clients[loop]
        assert 'keep' in _loop_clients[loop]

    @pytest.mark.asyncio
    async def test_close_all_clients_in_loop(self) -> None:
        """Verify closing all clients in current event loop."""
        _ = get_cached_client(cache_key='client-a')
        _ = get_cached_client(cache_key='client-b')

        await close_cached_clients()

        # All clients should be removed from this loop's cache
        loop = asyncio.get_running_loop()
        assert loop not in _loop_clients or len(_loop_clients[loop]) == 0

    @pytest.mark.asyncio
    async def test_close_nonexistent_key_is_noop(self) -> None:
        """Verify closing a non-existent key doesn't raise."""
        _ = get_cached_client(cache_key='exists')

        # Should not raise
        await close_cached_clients('does-not-exist')

    @pytest.mark.asyncio
    async def test_close_when_no_clients_is_noop(self) -> None:
        """Verify closing when no clients exist doesn't raise."""
        # Should not raise
        await close_cached_clients()


class TestClearClientCache:
    """Tests for clear_client_cache function."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_cached_clients(self) -> None:
        """Verify clear_client_cache removes all clients from cache."""
        _ = get_cached_client(cache_key='client-a')
        _ = get_cached_client(cache_key='client-b')

        clear_client_cache()

        assert len(_loop_clients) == 0

    def test_clear_when_empty_is_noop(self) -> None:
        """Verify clearing an empty cache doesn't raise."""
        clear_client_cache()
        clear_client_cache()  # Should not raise


class TestMultipleEventLoops:
    """Tests for behavior across multiple event loops.

    Note: These tests verify the cache key isolation but cannot fully test
    different event loops in the same test due to pytest-asyncio limitations.
    The WeakKeyDictionary behavior ensures proper isolation in production.
    """

    @pytest.fixture(autouse=True)
    def clear_cache(self) -> None:
        """Clear the client cache before each test."""
        clear_client_cache()

    @pytest.mark.asyncio
    async def test_cache_uses_current_event_loop_as_key(self) -> None:
        """Verify the cache is keyed by the current event loop."""
        _ = get_cached_client(cache_key='test')

        loop = asyncio.get_running_loop()
        assert loop in _loop_clients

    @pytest.mark.asyncio
    async def test_multiple_cache_keys_same_loop(self) -> None:
        """Verify multiple cache keys can coexist in the same loop."""
        client_a = get_cached_client(cache_key='plugin-a')
        client_b = get_cached_client(cache_key='plugin-b')
        client_c = get_cached_client(cache_key='plugin-c')

        loop = asyncio.get_running_loop()
        assert len(_loop_clients[loop]) == 3
        assert _loop_clients[loop]['plugin-a'] is client_a
        assert _loop_clients[loop]['plugin-b'] is client_b
        assert _loop_clients[loop]['plugin-c'] is client_c
