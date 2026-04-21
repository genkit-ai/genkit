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

"""Tests for HTTP client caching."""

import httpx
import pytest

from genkit._core._http_client import (
    clear_client_cache,
    close_cached_clients,
    get_cached_client,
)


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    clear_client_cache()


@pytest.mark.asyncio
async def test_returns_httpx_async_client() -> None:
    client = get_cached_client(cache_key='test')
    assert isinstance(client, httpx.AsyncClient)


@pytest.mark.asyncio
async def test_client_cached_per_event_loop() -> None:
    client1 = get_cached_client(cache_key='test')
    client2 = get_cached_client(cache_key='test')
    assert client1 is client2


@pytest.mark.asyncio
async def test_different_cache_keys_get_different_clients() -> None:
    client1 = get_cached_client(cache_key='plugin-a')
    client2 = get_cached_client(cache_key='plugin-b')
    assert client1 is not client2


@pytest.mark.asyncio
async def test_client_has_correct_headers() -> None:
    headers = {'Authorization': 'Bearer test-token', 'X-Custom': 'value'}
    client = get_cached_client(cache_key='test', headers=headers)
    assert client.headers.get('Authorization') == 'Bearer test-token'
    assert client.headers.get('X-Custom') == 'value'


@pytest.mark.asyncio
async def test_client_has_correct_timeout_float() -> None:
    client = get_cached_client(cache_key='test', timeout=30.0)
    assert client.timeout.read == 30.0
    assert client.timeout.connect == 30.0


@pytest.mark.asyncio
async def test_client_has_correct_timeout_object() -> None:
    timeout = httpx.Timeout(60.0, connect=10.0)
    client = get_cached_client(cache_key='test', timeout=timeout)
    assert client.timeout.read == 60.0
    assert client.timeout.connect == 10.0


@pytest.mark.asyncio
async def test_default_timeout_applied() -> None:
    client = get_cached_client(cache_key='test')
    assert client.timeout.read == 60.0
    assert client.timeout.connect == 10.0


@pytest.mark.asyncio
async def test_closed_client_gets_replaced() -> None:
    client1 = get_cached_client(cache_key='test')
    await client1.aclose()
    assert client1.is_closed

    client2 = get_cached_client(cache_key='test')
    assert client2 is not client1
    assert not client2.is_closed


@pytest.mark.asyncio
async def test_client_stored_in_cache() -> None:
    clear_client_cache()
    client = get_cached_client(cache_key='test')
    client2 = get_cached_client(cache_key='test')
    assert client is client2


def test_raises_without_running_event_loop() -> None:
    with pytest.raises(RuntimeError, match='no running event loop'):
        get_cached_client(cache_key='test')


@pytest.mark.asyncio
async def test_close_specific_client() -> None:
    client_to_close = get_cached_client(cache_key='to-close')
    client_keep = get_cached_client(cache_key='keep')

    await close_cached_clients('to-close')

    # 'to-close' should be gone; new fetch returns new client
    client_after = get_cached_client(cache_key='to-close')
    assert client_after is not client_to_close
    # 'keep' should still be cached
    assert get_cached_client(cache_key='keep') is client_keep


@pytest.mark.asyncio
async def test_close_all_clients_in_loop() -> None:
    client_a = get_cached_client(cache_key='client-a')
    client_b = get_cached_client(cache_key='client-b')

    await close_cached_clients()

    # Cache should be empty; new fetches return new clients
    new_a = get_cached_client(cache_key='client-a')
    new_b = get_cached_client(cache_key='client-b')
    assert new_a is not client_a
    assert new_b is not client_b


@pytest.mark.asyncio
async def test_close_nonexistent_key_is_noop() -> None:
    _ = get_cached_client(cache_key='exists')
    await close_cached_clients('does-not-exist')


@pytest.mark.asyncio
async def test_close_when_no_clients_is_noop() -> None:
    await close_cached_clients()


@pytest.mark.asyncio
async def test_clear_removes_all_cached_clients() -> None:
    client_a = get_cached_client(cache_key='client-a')
    client_b = get_cached_client(cache_key='client-b')
    clear_client_cache()
    # Cache cleared; new fetches return new clients
    new_a = get_cached_client(cache_key='client-a')
    new_b = get_cached_client(cache_key='client-b')
    assert new_a is not client_a
    assert new_b is not client_b


def test_clear_when_empty_is_noop() -> None:
    clear_client_cache()
    clear_client_cache()


@pytest.mark.asyncio
async def test_cache_uses_current_event_loop_as_key() -> None:
    client1 = get_cached_client(cache_key='test')
    client2 = get_cached_client(cache_key='test')
    assert client1 is client2


@pytest.mark.asyncio
async def test_multiple_cache_keys_same_loop() -> None:
    client_a = get_cached_client(cache_key='plugin-a')
    client_b = get_cached_client(cache_key='plugin-b')
    client_c = get_cached_client(cache_key='plugin-c')

    assert client_a is not client_b
    assert client_b is not client_c
    assert client_a is not client_c
    assert get_cached_client(cache_key='plugin-a') is client_a
    assert get_cached_client(cache_key='plugin-b') is client_b
    assert get_cached_client(cache_key='plugin-c') is client_c
