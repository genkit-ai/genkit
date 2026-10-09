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
import functools
import threading
import weakref
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast, overload

from typing_extensions import Self

T = TypeVar('T')
S = TypeVar('S')


class _LoopLocal(Generic[S, T]):
    """One cached instance per event loop; on a class, one per (object, loop).

    Called directly it is the module-level getter. Read as a class attribute it
    binds to the instance like ``functools.cached_property``: the bound getter
    is stored in the instance ``__dict__`` under the method name. Assigning the
    attribute (e.g. a test stub) replaces it for that instance.
    """

    def __init__(self, factory: Callable[..., T]) -> None:
        self._factory = factory
        self._by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, T] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        self._name: str | None = None
        functools.update_wrapper(self, factory, updated=())  # pyright: ignore[reportArgumentType]

    def __call__(self) -> T:
        loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._by_loop.get(loop)
            if existing is not None:
                return existing
            created = self._factory()
            self._by_loop[loop] = created
            return created

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(self, instance: S, owner: type | None = None) -> Callable[[], T]: ...
    def __get__(self, instance: object, owner: type | None = None) -> Any:  # noqa: ANN401
        if instance is None or self._name is None:
            return self
        cached = instance.__dict__.get(self._name)
        if cached is not None:
            return cached
        bound: _LoopLocal[Any, T] = _LoopLocal(functools.partial(self._factory, instance))
        with self._lock:
            return instance.__dict__.setdefault(self._name, bound)

    def __set__(self, instance: S, value: Callable[[], T]) -> None:
        # Lets tests stub the client: plugin._client = lambda: fake_client
        if self._name is None:
            raise AttributeError('loop_local_client method has no name')
        instance.__dict__[self._name] = value  # pyright: ignore[reportAttributeAccessIssue]


@overload
def loop_local_client(factory: Callable[[], T]) -> Callable[[], T]: ...
@overload
def loop_local_client(factory: Callable[[S], T]) -> _LoopLocal[S, T]: ...
def loop_local_client(factory: Callable[..., T]) -> Any:  # noqa: ANN401
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
      with SDK clients that take ``api_key`` or ``base_url``. Decorate a method;
      each plugin instance gets its own client per loop:

    ```python
    from openai import AsyncOpenAI


    class Bistro(Plugin):
        def __init__(self, api_key: str) -> None:
            self._api_key = api_key

        @loop_local_client
        def _client(self) -> AsyncOpenAI:
            return AsyncOpenAI(api_key=self._api_key)

        async def _generate(self, request: ModelRequest) -> ModelResponse:
            completion = await self._client().chat.completions.create(...)
    ```

    Plain callables work too, e.g. ``loop_local_client(asyncio.Lock)``.
    """
    return cast(Any, _LoopLocal(factory))
