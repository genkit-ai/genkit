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

"""Per-event-loop caching for async clients and other loop-bound objects."""

import asyncio
import threading
import weakref
from collections.abc import Callable
from typing import TypeVar

T = TypeVar('T')


def _is_closed(obj: object) -> bool:
    closed = getattr(obj, 'is_closed', None)
    # httpx exposes a property; the OpenAI and Anthropic SDKs a method. Only a
    # real True counts, so mocks and unrelated attributes never force a rebuild.
    if callable(closed):
        closed = closed()
    return closed is True


def loop_local_client(factory: Callable[[], T]) -> Callable[[], T]:
    """Cache one instance per event loop. Use it as a decorator on a factory.

    Async clients (httpx, the OpenAI and Anthropic SDKs) are bound to the event
    loop that created them, so one module-level client breaks when Genkit runs
    on a second loop. The decorated function builds the client on first call
    in each loop and returns that same client on every later call there.

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

    A cached object whose ``is_closed`` is true (httpx property, SDK method) is
    rebuilt, so closing a client by hand doesn't leave a dead one in the cache.
    Plain callables work too, e.g. ``loop_local_client(asyncio.Lock)``.
    """
    by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = weakref.WeakKeyDictionary()
    lock = threading.Lock()

    def _get() -> T:
        loop = asyncio.get_running_loop()
        with lock:
            existing = by_loop.get(loop)
            if existing is not None and not _is_closed(existing):
                return existing
            created = factory()
            by_loop[loop] = created
            return created

    return _get
