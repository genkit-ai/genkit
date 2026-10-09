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

"""Per-event-loop caching for long-lived async clients."""

import asyncio
import threading
import weakref
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')


def loop_local_client(factory: Callable[[], T]) -> Callable[[], T]:
    """Cache a long-lived client per event loop. Use it as a decorator on a factory.

    Use it for a client you reuse across calls that code on different event
    loops may call. Async clients (httpx, the OpenAI and Anthropic SDKs) are
    bound to the loop they first run on, so one shared client breaks when Genkit
    runs on a second loop. Each loop gets its own client, built on first use
    there and returned on every later call there.

    The client is shared, so don't close it; it lives as long as its loop. For a
    client scoped to one call, skip the cache and use
    ``async with httpx.AsyncClient() as client:`` inside the call.

    ```python
    import httpx
    from genkit.plugin_api import loop_local_client


    # 1. Decorate a factory
    @loop_local_client
    def http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=60.0)


    # 2. Call it wherever you need the client
    response = await http_client().get('https://example.com/menu.json')

    # 3. Same loop, same client
    print(http_client() is http_client())
    # => True
    ```

    Where the client lives depends on whether it carries plugin settings:

    - **Module level** (the decorator above): the client is a plain transport
      and the API key, URL, headers, and timeout go on each request. Every
      plugin instance shares one connection pool per loop.
    - **On the plugin instance**: the client is built from plugin settings, as
      with SDK clients that take ``api_key`` or ``base_url``. Wrap a factory in
      ``__init__`` and call the getter where you need the client:

    ```python
    from openai import AsyncOpenAI


    class Bistro(Plugin):
        def __init__(self, api_key: str) -> None:
            self._client = loop_local_client(lambda: AsyncOpenAI(api_key=api_key))

        async def _generate(self, request: ModelRequest) -> ModelResponse:
            completion = await self._client().chat.completions.create(...)
    ```

    Plain callables work too, e.g. ``loop_local_client(asyncio.Lock)``.
    """
    by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = weakref.WeakKeyDictionary()
    lock = threading.Lock()

    def _get() -> T:
        loop = asyncio.get_running_loop()
        with lock:
            existing = by_loop.get(loop)
            if existing is not None:
                return existing
            created = factory()
            by_loop[loop] = created
            return created

    return _get
