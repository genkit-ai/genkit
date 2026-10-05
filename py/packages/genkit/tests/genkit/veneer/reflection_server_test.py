#!/usr/bin/env python3
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the automatic background Dev UI reflection server.

Covers the key invariants of the background-thread approach:
- Server starts on Genkit() construction in dev mode, no extra wiring needed
- Works alongside FastAPI with no lifespan hooks
- Multiple Genkit instances can coexist in the same process
- Flows registered after construction are immediately visible
- No server starts in production mode
"""

import asyncio
import json
import os
import socket
import threading
import time
from typing import Any
from unittest import mock

import httpx
import pytest
from websockets.asyncio.server import serve

from genkit import Genkit
from genkit._core._environment import GENKIT_ENV, GenkitEnvironment
from genkit._core._reflection import ServerSpec


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _wait_and_get(ai: Genkit, path: str) -> httpx.Response:
    assert ai._reflection_ready.wait(timeout=5), 'Reflection server never became ready'  # pyright: ignore[reportPrivateUsage]
    spec = ai._reflection_server_spec  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    return httpx.get(f'{spec.url}{path}', timeout=1.0)


def test_server_starts_on_construction() -> None:
    """Core invariant: Genkit() in dev mode brings up Dev UI automatically.

    No run_main(), no lifespan hooks — construction is sufficient.
    """
    port = _find_free_port()
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))
        resp = _wait_and_get(ai, '/api/__health')
    assert resp.status_code == 200


def test_flow_registered_after_construction_is_visible() -> None:
    """Flows defined after Genkit() are visible in /api/actions.

    Note: this is a sequential test (flow registered before the HTTP request),
    so it proves the plumbing works but NOT concurrent thread-safety.
    See test_registry_reads_concurrent_with_writes for that.
    """
    port = _find_free_port()
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))

        @ai.flow()
        async def greet(name: str) -> str:
            return f'Hello, {name}!'

        resp = _wait_and_get(ai, '/api/actions')

    assert resp.status_code == 200
    assert '/flow/greet' in resp.json()


def test_registry_reads_concurrent_with_writes() -> None:
    """The reflection thread reads the registry while the main thread writes to it.

    Spams /api/actions from a background thread while registering flows via
    @ai.flow() on the main thread simultaneously. The registry uses
    threading.RLock — responses must always be valid JSON dicts, never empty
    or corrupted.
    """
    port = _find_free_port()
    errors: list[Exception] = []

    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))
        assert ai._reflection_ready.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]

        stop = threading.Event()

        def spam_reads() -> None:
            url = f'http://127.0.0.1:{port}/api/actions'
            while not stop.is_set():
                try:
                    data = httpx.get(url, timeout=1.0).json()
                    assert isinstance(data, dict), f'Got non-dict: {data!r}'
                except Exception as e:
                    errors.append(e)

        reader = threading.Thread(target=spam_reads, daemon=True)
        reader.start()

        # Register flows while the reader is active; sufficient to exercise concurrent writes
        def _make_flow(i: int) -> None:
            @ai.flow()
            async def _f(x: str) -> str:
                return f'flow_{i}: {x}'

        for i in range(20):
            _make_flow(i)

        stop.set()
        reader.join(timeout=2)
        assert not reader.is_alive(), 'reader thread did not stop'

    assert not errors, f'Concurrent read/write errors: {errors}'


def test_two_instances_serve_concurrently() -> None:
    """Two Genkit() instances in the same process don't interfere with each other."""
    port1, port2 = _find_free_port(), _find_free_port()
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai1 = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port1))
        ai2 = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port2))

        assert ai1._reflection_ready.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]
        assert ai2._reflection_ready.wait(timeout=5)  # pyright: ignore[reportPrivateUsage]

    assert httpx.get(f'http://127.0.0.1:{port1}/api/__health', timeout=1.0).status_code == 200
    assert httpx.get(f'http://127.0.0.1:{port2}/api/__health', timeout=1.0).status_code == 200


def test_busy_pinned_port_fails_the_constructor() -> None:
    """A pinned port that is taken raises from Genkit(), not from the background thread."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        port = taken.getsockname()[1]
        env = {'GENKIT_REFLECTION_ENABLED': 'true', 'GENKIT_REFLECTION_PORT': str(port)}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(OSError):
                Genkit()


def test_serves_on_ipv6_loopback() -> None:
    """GENKIT_REFLECTION_HOST=::1 binds an IPv6 socket instead of failing to resolve."""
    if not socket.has_ipv6:
        pytest.skip('IPv6 not available')
    env = {'GENKIT_REFLECTION_ENABLED': 'true', 'GENKIT_REFLECTION_HOST': '::1'}
    with mock.patch.dict(os.environ, env, clear=True):
        ai = Genkit()
        resp = _wait_and_get(ai, '/api/__health')
    assert resp.status_code == 200
    spec = ai._reflection_server_spec  # pyright: ignore[reportPrivateUsage]
    assert spec is not None
    assert spec.host == '[::1]'


def test_programmatic_port_is_bound_exactly() -> None:
    """ServerSpec(port=N) binds N, and a taken N fails instead of shifting."""
    port = _find_free_port()
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}, clear=True):
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))
        assert _wait_and_get(ai, '/api/__health').status_code == 200
        spec = ai._reflection_server_spec  # pyright: ignore[reportPrivateUsage]
        assert spec is not None
        assert spec.port == port
        with pytest.raises(OSError):
            Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))


@pytest.mark.parametrize('manager_closes', [True, False], ids=['manager-closes', 'manager-keeps-open'])
def test_run_main_returns_when_the_cli_rejects_the_runtime(manager_closes: bool) -> None:
    """A -32001 register rejection stops reflection, and run_main returns instead of hanging.

    The real CLI closes the socket after rejecting, but the runtime must not
    rely on that: with the socket left open, it has to hang up itself.
    """
    loop = asyncio.new_event_loop()
    started = threading.Event()
    stop = asyncio.Event()
    port_box: list[int] = []

    async def _reject(ws: Any) -> None:  # noqa: ANN401 - websockets connection
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get('method') == 'register':
                await ws.send(
                    json.dumps({
                        'jsonrpc': '2.0',
                        'id': msg['id'],
                        'error': {'code': -32001, 'message': 'Invalid reflection secret.'},
                    })
                )
                if manager_closes:
                    await ws.close(1008, 'unauthorized')
                    return

    async def _serve() -> None:
        async with serve(_reject, '127.0.0.1', 0) as server:
            port_box.append(next(iter(server.sockets)).getsockname()[1])
            started.set()
            await stop.wait()

    manager = threading.Thread(target=lambda: loop.run_until_complete(_serve()), daemon=True)
    manager.start()
    try:
        assert started.wait(timeout=5)
        env = {
            'GENKIT_REFLECTION_ENABLED': 'true',
            'GENKIT_REFLECTION_V2_SERVER': f'ws://127.0.0.1:{port_box[0]}',
            'GENKIT_REFLECTION_SECRET_TOKEN': 'wrong',
        }
        with mock.patch.dict(os.environ, env, clear=True):
            ai = Genkit()

            async def _main() -> str:
                return 'done'

            # run_main installs a SIGTERM receiver, so it must run on the main
            # thread. A watchdog turns a hang into a failure instead of a stuck
            # test run: it stops reflection itself, which would also unblock
            # run_main, so the elapsed time is what the assertion checks.
            watchdog = threading.Timer(10, ai._reflection_stopped.set)  # pyright: ignore[reportPrivateUsage]
            watchdog.start()
            started_at = time.monotonic()
            try:
                result = ai.run_main(_main())
            finally:
                watchdog.cancel()
        assert time.monotonic() - started_at < 10, 'run_main kept blocking after the CLI rejected the runtime'
        assert result == 'done'
    finally:
        loop.call_soon_threadsafe(stop.set)
        manager.join(timeout=5)


def test_ready_message_outside_dev_names_the_bound_address() -> None:
    """Outside dev there is no runtime file, so run_main must not claim the Dev UI is ready."""
    port = _find_free_port()
    env = {'GENKIT_REFLECTION_ENABLED': 'true', 'GENKIT_REFLECTION_PORT': str(port)}
    with mock.patch.dict(os.environ, env, clear=True):
        ai = Genkit()
        message = ai._reflection_ready_message()  # pyright: ignore[reportPrivateUsage]
    assert message.startswith(f'Reflection API listening on 127.0.0.1:{port}.')
    assert 'Dev UI' not in message


def test_ready_message_in_dev_mentions_the_dev_ui() -> None:
    """Under dev the runtime file is written, so the Dev UI can find the runtime."""
    port = _find_free_port()
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai = Genkit(reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port))
        message = ai._reflection_ready_message()  # pyright: ignore[reportPrivateUsage]
    assert message.startswith('Dev UI ready.')


def test_no_server_in_prod_mode() -> None:
    """Genkit() with no GENKIT_ENV must NOT start a background server."""
    with mock.patch.dict(os.environ, {}, clear=True):
        ai = Genkit()

    assert not ai._reflection_ready.is_set()  # pyright: ignore[reportPrivateUsage]


def test_no_server_from_host_or_port_alone() -> None:
    """Outside dev, host/port are settings, not on-switches (same as before this change)."""
    port = _find_free_port()
    env = {'GENKIT_REFLECTION_HOST': '127.0.0.1', 'GENKIT_REFLECTION_PORT': str(port)}
    with mock.patch.dict(os.environ, env, clear=True):
        ai = Genkit()

    assert not ai._reflection_config.enabled  # pyright: ignore[reportPrivateUsage]
    assert not ai._reflection_ready.is_set()  # pyright: ignore[reportPrivateUsage]
