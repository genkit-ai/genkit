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

"""Tests for the per-event-loop client cache."""

import asyncio
import gc
import http.server
import os
import threading
import weakref
from collections.abc import Callable
from unittest import mock
from unittest.mock import MagicMock

import httpx
import pytest

from genkit.plugin_api import loop_local_client


@pytest.mark.asyncio
async def test_same_loop_returns_cached_client() -> None:
    get = loop_local_client(httpx.AsyncClient)
    first = get()
    assert get() is first
    await first.aclose()


def test_each_loop_gets_its_own_client() -> None:
    get = loop_local_client(httpx.AsyncClient)

    async def grab() -> httpx.AsyncClient:
        return get()

    a = asyncio.run(grab())
    b = asyncio.run(grab())
    assert a is not b


@pytest.mark.asyncio
async def test_decorator_form_caches_the_factory_result() -> None:
    @loop_local_client
    def http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=60.0)

    first = http_client()
    assert http_client() is first
    await first.aclose()


class _Bistro:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.builds = 0

    @loop_local_client
    def client(self) -> httpx.AsyncClient:
        """Per-loop HTTP client for the bistro."""
        self.builds += 1
        return httpx.AsyncClient(headers={'x-api-key': self.api_key})


@pytest.mark.asyncio
async def test_method_form_caches_per_instance() -> None:
    bistro = _Bistro('k1')
    assert bistro.client.__name__ == 'client'  # ty: ignore[unresolved-attribute]
    assert bistro.client.__doc__ == 'Per-loop HTTP client for the bistro.'
    first = bistro.client()
    assert bistro.client() is first
    assert bistro.builds == 1
    assert first.headers['x-api-key'] == 'k1'
    await first.aclose()


@pytest.mark.asyncio
async def test_method_form_keeps_instances_apart() -> None:
    a, b = _Bistro('k1'), _Bistro('k2')
    assert a.client() is not b.client()
    assert b.client().headers['x-api-key'] == 'k2'
    await a.client().aclose()
    await b.client().aclose()


def test_method_form_gives_each_loop_its_own_client() -> None:
    bistro = _Bistro('k1')

    async def grab() -> httpx.AsyncClient:
        return bistro.client()

    assert asyncio.run(grab()) is not asyncio.run(grab())
    assert bistro.builds == 2


@pytest.mark.asyncio
async def test_instance_attribute_overrides_method_form() -> None:
    """Tests stub the client by assigning the attribute, as with cached_property."""
    bistro = _Bistro('k1')
    stub = MagicMock()
    bistro.client = lambda: stub
    assert bistro.client() is stub
    assert bistro.builds == 0


def test_method_form_stays_loop_local_after_first_read() -> None:
    """The bound getter cached in __dict__ is itself per-loop."""
    bistro = _Bistro('k1')
    getter = bistro.client

    async def grab() -> httpx.AsyncClient:
        assert bistro.client is getter
        return bistro.client()

    first, second = asyncio.run(grab()), asyncio.run(grab())
    assert first is not second
    assert bistro.builds == 2


@pytest.mark.asyncio
async def test_del_drops_bound_getter_and_rebuilds() -> None:
    bistro = _Bistro('k1')
    first = bistro.client()
    del bistro.client
    second = bistro.client()
    assert second is not first
    assert bistro.builds == 2
    await first.aclose()
    await second.aclose()


@pytest.mark.asyncio
async def test_patch_object_on_unread_attribute_restores() -> None:
    bistro = _Bistro('k1')
    stub = MagicMock()
    with mock.patch.object(bistro, 'client', lambda: stub):
        assert bistro.client() is stub
    real = bistro.client()
    assert real is not stub
    assert bistro.builds == 1
    await real.aclose()


class _KitchenClient:
    """Stands in for an SDK client whose transport points back at its loop."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.closed_on: asyncio.AbstractEventLoop | None = None

    async def aclose(self) -> None:
        self.closed_on = asyncio.get_running_loop()


def test_asyncio_run_closes_the_client_and_frees_the_loop() -> None:
    """asyncio.run() shutdown awaits the cached client's aclose() and lets the loop be collected.

    Regression: a weak key on the loop never died, because the client holds the
    loop, so every asyncio.run() pinned its loop and keep-alive sockets.
    """
    get = loop_local_client(_KitchenClient)
    clients: list[_KitchenClient] = []
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def order() -> None:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        clients.append(get())

    for _ in range(8):
        asyncio.run(order())
    gc.collect()

    assert all(c.closed_on is c.loop for c in clients)
    clients.clear()
    gc.collect()
    assert [r() for r in loops] == [None] * 8


def test_loop_closed_without_shutdown_is_dropped_on_next_call() -> None:
    """A loop closed with plain close() is evicted the next time any loop asks for a client."""
    get = loop_local_client(_KitchenClient)
    loop = asyncio.new_event_loop()
    first = loop.run_until_complete(_call(get))
    loop.close()
    loop_ref = weakref.ref(loop)
    del loop

    async def again() -> _KitchenClient:
        return get()

    second = asyncio.run(again())
    assert second is not first
    assert first.closed_on is None
    del first
    gc.collect()
    assert loop_ref() is None


async def _call(get: Callable[[], _KitchenClient]) -> _KitchenClient:
    return get()


def test_clients_without_async_close_are_left_alone() -> None:
    """Values with no async close (e.g. asyncio.Lock) are cached and dropped without error."""
    get = loop_local_client(asyncio.Lock)

    async def take() -> bool:
        lock = get()
        assert get() is lock
        async with lock:
            return lock.locked()

    assert asyncio.run(take()) is True
    assert asyncio.run(take()) is True


def test_sequential_asyncio_run_keeps_fd_count_flat() -> None:
    """Eight asyncio.run() calls over a keep-alive httpx client leave no sockets open."""

    class Menu(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self) -> None:  # noqa: N802
            body = b'{"dish":"miso"}'
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Menu)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{server.server_port}/menu'

    @loop_local_client
    def http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=5)

    async def fetch() -> int:
        return (await http_client().get(url)).status_code

    def open_fds() -> int:
        return len(os.listdir('/dev/fd'))

    try:
        assert asyncio.run(fetch()) == 200
        gc.collect()
        before = open_fds()
        for _ in range(8):
            assert asyncio.run(fetch()) == 200
        gc.collect()
        after = open_fds()
    finally:
        server.shutdown()
        server.server_close()

    assert after <= before
