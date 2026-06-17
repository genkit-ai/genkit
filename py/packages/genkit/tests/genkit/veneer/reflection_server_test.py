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

import os
import socket
import threading
from unittest import mock

import httpx

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


def test_no_server_in_prod_mode() -> None:
    """Genkit() with no GENKIT_ENV must NOT start a background server."""
    with mock.patch.dict(os.environ, {}, clear=True):
        ai = Genkit()

    assert not ai._reflection_ready.is_set()  # pyright: ignore[reportPrivateUsage]
