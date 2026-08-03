#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-event-loop client caching (#4925)."""

import asyncio

from genkit._core._loop_cache import _loop_local_client


def test_loop_local_client_caches_within_same_loop() -> None:
    getter = _loop_local_client(object)
    created: list[object] = []

    async def once() -> tuple[object, object]:
        a = getter()
        b = getter()
        created.extend([a, b])
        return a, b

    a, b = asyncio.run(once())
    assert a is b
    assert len(created) == 2


def test_loop_local_client_new_client_after_closed_loop() -> None:
    """Cloud Run warm-instance sync bridge: new loop per request must not reuse a closed-loop client."""
    created: list[object] = []
    getter = _loop_local_client(lambda: created.append(object()) or created[-1])

    def call_once() -> object:
        loop = asyncio.new_event_loop()
        try:

            async def _get() -> object:
                return getter()

            return loop.run_until_complete(_get())
        finally:
            loop.close()

    first = call_once()
    second = call_once()
    assert first is not second
    assert len(created) == 2


def test_loop_local_client_prunes_closed_loop_entries() -> None:
    created: list[object] = []
    getter = _loop_local_client(lambda: created.append(object()) or created[-1])

    loop1 = asyncio.new_event_loop()
    try:

        async def _get() -> object:
            return getter()

        first = loop1.run_until_complete(_get())
    finally:
        loop1.close()

    # Access from a live loop should prune the closed loop1 entry and create fresh.
    loop2 = asyncio.new_event_loop()
    try:

        async def _get2() -> object:
            return getter()

        second = loop2.run_until_complete(_get2())
    finally:
        loop2.close()

    assert first is not second
    assert len(created) == 2
