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

"""Shared HTTP client utilities for Genkit plugins.

This module provides utilities for managing httpx.AsyncClient instances
across different event loops. The key feature is per-event-loop caching
that ensures:

1. Clients are reused within the same event loop (avoiding connection
   setup overhead)
2. Each event loop gets its own client (avoiding "bound to different
   event loop" errors)
3. Automatic cleanup when event loops are garbage collected

Example Usage
-------------

Basic usage with default settings::

    from genkit.core.http_client import get_cached_client


    async def my_async_function():
        client = get_cached_client(
            cache_key='my-plugin',
            headers={'Authorization': 'Bearer token'},
        )
        response = await client.get('https://api.example.com')

With custom timeout::

    client = get_cached_client(
        cache_key='my-plugin',
        headers={'Authorization': 'Bearer token'},
        timeout=httpx.Timeout(120.0, connect=30.0),
    )

Multiple cache keys for different configurations::

    # One client for API calls
    api_client = get_cached_client(
        cache_key='my-plugin/api',
        headers={'Authorization': 'Bearer api-token'},
    )

    # Another client for media fetching (different headers)
    media_client = get_cached_client(
        cache_key='my-plugin/media',
        headers={'User-Agent': 'MyApp/1.0'},
    )

Implementation Notes
--------------------

The cache uses a two-level structure:

1. Outer level: WeakKeyDictionary keyed by event loop
   - Automatically cleans up when event loop is garbage collected
   - Prevents memory leaks from abandoned event loops

2. Inner level: Regular dict keyed by cache_key string
   - Allows multiple clients per event loop with different configurations
   - Cache key should include any configuration that affects the client

The client configuration (headers, timeout, etc.) is only used when
creating a new client. If a cached client exists, the provided
configuration is ignored. This is intentional - changing configuration
requires a new cache key.

Thread Safety
-------------

This module is thread-safe. A threading.Lock protects all modifications
and multi-step read operations on the cache. In multi-threaded applications
where different threads manage different event loops, this prevents race
conditions between concurrent cache operations.
"""

import asyncio
import threading
import weakref
from collections.abc import MutableMapping
from typing import Any

import httpx

from genkit.core.logging import get_logger

logger = get_logger(__name__)

# Two-level cache: event_loop -> (cache_key -> client)
# Using WeakKeyDictionary for event loops ensures automatic cleanup
_loop_clients: MutableMapping[asyncio.AbstractEventLoop, dict[str, httpx.AsyncClient]] = weakref.WeakKeyDictionary()

# Lock for thread-safe cache operations
_cache_lock = threading.Lock()


def get_cached_client(
    cache_key: str,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | float | None = None,
    **httpx_kwargs: Any,  # noqa: ANN401
) -> httpx.AsyncClient:
    """Get or create a cached httpx.AsyncClient for the current event loop.

    This function provides per-event-loop client caching, which:
    - Reuses clients within the same event loop (reduces connection overhead)
    - Creates separate clients for different event loops (avoids binding errors)
    - Automatically cleans up when event loops are garbage collected

    Args:
        cache_key: Unique identifier for this client configuration.
            Use a consistent key for the same configuration to benefit from
            caching. Include any distinguishing factors in the key (e.g.,
            'my-plugin/api' vs 'my-plugin/media' for different use cases).
        headers: HTTP headers to include in all requests.
        timeout: Request timeout. Can be a float (total timeout in seconds)
            or an httpx.Timeout object for fine-grained control.
            Defaults to 60s total with 10s connect timeout.
        **httpx_kwargs: Additional arguments passed to httpx.AsyncClient().

    Returns:
        A cached or newly created httpx.AsyncClient instance.

    Raises:
        RuntimeError: If called outside of an async context (no running loop).

    Note:
        The client configuration is only used when creating a new client.
        If a cached client exists, the provided configuration is ignored.
        To use different configurations, use different cache_key values.

    Example::

        # In an async function
        client = get_cached_client(
            cache_key='vertex-ai-evaluator',
            headers={'Authorization': f'Bearer {token}'},
            timeout=60.0,
        )
        response = await client.post(url, json=data)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as e:
        raise RuntimeError(
            'get_cached_client() must be called from within an async context '
            '(inside an async function with a running event loop)'
        ) from e

    with _cache_lock:
        # Get or create the cache dict for this event loop
        loop_cache = _loop_clients.setdefault(loop, {})

        # Check if we have a cached client for this key
        if cache_key in loop_cache:
            client = loop_cache[cache_key]
            # Verify the client is still usable
            if not client.is_closed:
                return client
            # Client was closed, remove from cache
            logger.debug('Cached client was closed, creating new one', cache_key=cache_key)
            del loop_cache[cache_key]

        # Create a new client
        if timeout is None:
            timeout = httpx.Timeout(60.0, connect=10.0)
        elif isinstance(timeout, (int, float)):
            timeout = httpx.Timeout(float(timeout))

        client = httpx.AsyncClient(
            headers=headers or {},
            timeout=timeout,
            **httpx_kwargs,
        )

        loop_cache[cache_key] = client
        logger.debug('Created new httpx client', cache_key=cache_key, loop_id=id(loop))

        return client


async def close_cached_clients(cache_key: str | None = None) -> None:
    """Close and remove cached clients.

    This is useful for cleanup in tests or when reconfiguring clients.
    In normal usage, clients are automatically cleaned up when their
    event loop is garbage collected.

    Args:
        cache_key: If provided, only close clients with this key.
            If None, close all clients in the current event loop's cache.

    Example::

        # Close specific client
        await close_cached_clients('my-plugin')

        # Close all clients in current event loop
        await close_cached_clients()
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # No running loop, nothing to close

    # Collect clients to close under the lock, then close outside the lock
    # to avoid blocking other threads during async I/O
    clients_to_close: dict[str, httpx.AsyncClient] = {}

    with _cache_lock:
        if loop not in _loop_clients:
            return

        loop_cache = _loop_clients[loop]

        if cache_key is not None:
            # Close specific client
            if cache_key in loop_cache:
                clients_to_close[cache_key] = loop_cache.pop(cache_key)
        else:
            # Close all clients in this loop's cache
            clients_to_close.update(loop_cache)
            loop_cache.clear()

    # Close clients outside the lock to avoid blocking
    for key, client in clients_to_close.items():
        try:
            await client.aclose()
            logger.debug('Closed cached client', cache_key=key)
        except Exception as e:
            logger.warning('Failed to close cached client', cache_key=key, error=e)


def clear_client_cache() -> None:
    """Clear all cached clients across all event loops.

    This is primarily for testing purposes. Clients are NOT closed,
    just removed from the cache. Use close_cached_clients() to properly
    close clients before clearing.

    Warning:
        This will cause any existing client references to become orphaned.
        Only use this in tests or when you're sure no clients are in use.
    """
    with _cache_lock:
        _loop_clients.clear()
    logger.debug('Cleared all client caches')
