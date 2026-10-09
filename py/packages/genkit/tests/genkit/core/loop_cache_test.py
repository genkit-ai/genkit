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
