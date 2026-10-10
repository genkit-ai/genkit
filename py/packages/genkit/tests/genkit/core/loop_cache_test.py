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
        self.builds += 1
        return httpx.AsyncClient(headers={'x-api-key': self.api_key})


@pytest.mark.asyncio
async def test_method_form_caches_per_instance() -> None:
    bistro = _Bistro('k1')
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
    bistro.client = lambda: stub  # type: ignore[method-assign]
    assert bistro.client() is stub
    assert bistro.builds == 0
