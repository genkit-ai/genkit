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
import contextlib
import functools
import inspect
import threading
import weakref
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar, cast, overload

from typing_extensions import Self

from genkit._core._logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
S = TypeVar('S')


async def _aclose_client(client: object) -> None:
    """Close an async client if it exposes an async close; ignore anything else."""
    # httpx.AsyncClient.aclose, AsyncOpenAI/AsyncAnthropic.close, genai.Client.aio.aclose.
    for owner, name in ((client, 'aclose'), (client, 'close'), (getattr(client, 'aio', None), 'aclose')):
        close = getattr(owner, name, None)
        if close is not None and inspect.iscoroutinefunction(close):
            try:
                await close()
            except Exception as e:  # noqa: BLE001 - best-effort cleanup at loop shutdown
                logger.debug('loop_local_client: closing cached client failed', error=e)
            return


class _Entry(NamedTuple):
    loop: weakref.ref[asyncio.AbstractEventLoop]
    value: object
    finalizer: AsyncGenerator[None, None]


class _LoopLocal(Generic[S, T]):
    """One cached instance per event loop; on a class, one per (object, loop).

    Called directly it is the module-level getter. Read as a class attribute it
    binds to the instance like ``functools.cached_property``: the bound getter,
    itself a per-loop cache, is stored in the instance ``__dict__`` under the
    method name, so later reads skip ``__get__``. It is a non-data descriptor:
    assigning the attribute (e.g. a test stub) shadows it for that instance, and
    ``del`` drops the bound getter and its per-loop clients.
    """

    def __init__(self, factory: Callable[..., T]) -> None:
        self._factory = factory
        # Keyed by id(loop), not weakly by the loop: a cached client's
        # keep-alive transport points back at its loop, so a weak key never
        # dies and every closed loop stayed pinned with its sockets.
        self._by_loop: dict[int, _Entry] = {}
        self._lock = threading.Lock()
        self._name: str | None = None
        functools.update_wrapper(self, factory, updated=())  # pyright: ignore[reportArgumentType]

    def __call__(self) -> T:
        loop = asyncio.get_running_loop()
        key = id(loop)
        with self._lock:
            entry = self._by_loop.get(key)
            if entry is not None and entry.loop() is loop:
                return cast(T, entry.value)
            stale = self._pop_dead_locked()
            created = self._factory()
            finalizer = self._finalize_on_shutdown(key, loop, created)
            self._by_loop[key] = _Entry(weakref.ref(loop), created, finalizer)
        for old in stale:
            _close_finalizer_now(old.finalizer)
        # Run the finalizer to its first yield on this loop's thread. That
        # registers it with the loop's async-generator hooks, so
        # loop.shutdown_asyncgens() (asyncio.run, uvloop.run, asgiref) closes
        # it while the loop can still await the client's close.
        with contextlib.suppress(StopIteration):
            finalizer.asend(None).send(None)
        return created

    def _pop_dead_locked(self) -> list[_Entry]:
        """Drop entries whose loop closed without running shutdown_asyncgens()."""
        dead = [k for k, e in self._by_loop.items() if (lp := e.loop()) is None or lp.is_closed()]
        return [self._by_loop.pop(k) for k in dead]

    def _forget(self, key: int, value: object) -> None:
        with self._lock:
            entry = self._by_loop.get(key)
            if entry is not None and entry.value is value:
                del self._by_loop[key]

    async def _finalize_on_shutdown(
        self, key: int, loop: asyncio.AbstractEventLoop, value: object
    ) -> AsyncGenerator[None, None]:
        loop_ref = weakref.ref(loop)
        del loop
        try:
            yield
        finally:
            self._forget(key, value)
            # Only await on the client's own loop. A finalizer closed from
            # another loop (its own loop already closed) just drops the client.
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is not None and running is loop_ref():
                await _aclose_client(value)

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    @overload
    def __get__(self, instance: None, owner: type | None = None) -> Self: ...
    @overload
    def __get__(self, instance: S, owner: type | None = None) -> Callable[[], T]: ...
    def __get__(self, instance: object, owner: type | None = None) -> Any:  # noqa: ANN401
        if instance is None or self._name is None:
            return self
        # Only reached until the bound getter lands in instance.__dict__, which
        # then shadows this non-data descriptor. setdefault keeps concurrent
        # first reads on one bound getter.
        bound: _LoopLocal[Any, T] = _LoopLocal(functools.partial(self._factory, instance))
        functools.update_wrapper(bound, self, updated=())  # pyright: ignore[reportArgumentType]
        with self._lock:
            return instance.__dict__.setdefault(self._name, bound)

    if TYPE_CHECKING:
        # Declared for type checkers only, so `plugin._client = lambda: stub`
        # and `del plugin._client` type-check. At runtime the descriptor stays
        # non-data and both act on the instance __dict__.
        def __set__(self, instance: S, value: Callable[[], T]) -> None: ...
        def __delete__(self, instance: S) -> None: ...


@overload
def loop_local_client(factory: Callable[[], T]) -> Callable[[], T]: ...
@overload
def loop_local_client(factory: Callable[[S], T]) -> _LoopLocal[S, T]: ...
def _close_finalizer_now(finalizer: AsyncGenerator[None, None]) -> None:
    """Finish a finalizer whose loop is gone, so GC doesn't hand it to a closed loop."""
    with contextlib.suppress(StopIteration, RuntimeError):
        finalizer.aclose().send(None)


def loop_local_client(factory: Callable[..., T]) -> Any:  # noqa: ANN401
    """Cache a long-lived client per event loop. Use it as a decorator on a factory.

    Use it for a client you reuse across calls that code on different event
    loops may call. Async clients (httpx, the OpenAI and Anthropic SDKs) are
    bound to the loop they first run on, so one shared client breaks when Genkit
    runs on a second loop. Each loop gets its own client, built on first use
    there and returned on every later call there.

    The client is shared, so don't close it; it lives as long as its loop.
    When the loop shuts down through ``asyncio.run`` (or anything else that
    calls ``loop.shutdown_asyncgens()``), Genkit awaits the client's
    ``aclose()`` or async ``close()`` and drops it, so a closed loop doesn't
    pin its sockets. A loop closed without that step is dropped the next time
    the getter runs. For a client scoped to one call, skip the cache and use
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
