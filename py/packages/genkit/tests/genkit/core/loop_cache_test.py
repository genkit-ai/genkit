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
async def test_closed_httpx_client_is_rebuilt() -> None:
    get = loop_local_client(httpx.AsyncClient)
    first = get()
    await first.aclose()

    second = get()
    assert second is not first
    assert not second.is_closed
    assert get() is second
    await second.aclose()


@pytest.mark.asyncio
async def test_closed_method_style_client_is_rebuilt() -> None:
    class SdkClient:
        """OpenAI and Anthropic SDK clients expose is_closed() as a method."""

        def __init__(self) -> None:
            self.closed = False

        def is_closed(self) -> bool:
            return self.closed

    get = loop_local_client(SdkClient)
    first = get()
    first.closed = True
    assert get() is not first


@pytest.mark.asyncio
async def test_mock_is_closed_does_not_force_rebuild() -> None:
    get = loop_local_client(MagicMock)
    first = get()
    assert get() is first
