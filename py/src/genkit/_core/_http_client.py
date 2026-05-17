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

"""Shared HTTP client utilities for Genkit plugins."""

from typing import Any

import httpx

from genkit._core._logger import get_logger
from genkit._core._loop_cache import _loop_local_client

logger = get_logger(__name__)

_get_store = _loop_local_client(dict)


def get_cached_client(
    cache_key: str,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | float | None = None,
    **httpx_kwargs: Any,
) -> httpx.AsyncClient:
    """Get or create a cached httpx.AsyncClient for the current event loop."""
    d = _get_store()
    if cache_key not in d or d[cache_key].is_closed:
        if timeout is None:
            timeout = httpx.Timeout(60.0, connect=10.0)
        elif isinstance(timeout, (int, float)):
            timeout = httpx.Timeout(float(timeout))
        d[cache_key] = httpx.AsyncClient(headers=headers or {}, timeout=timeout, **httpx_kwargs)
    return d[cache_key]


async def close_cached_clients(cache_key: str | None = None) -> None:
    """Close and remove cached clients for the current event loop."""
    try:
        d = _get_store()
    except RuntimeError:
        return

    clients_to_close: dict[str, httpx.AsyncClient] = {}

    if cache_key is not None:
        if cache_key in d:
            clients_to_close[cache_key] = d.pop(cache_key)
    else:
        clients_to_close.update(d)
        d.clear()

    for key, client in clients_to_close.items():
        try:
            await client.aclose()
        except Exception as e:
            logger.warning('Failed to close cached client', cache_key=key, error=e)


def clear_client_cache() -> None:
    """Clear all cached clients (for testing). Does NOT close clients."""
    try:
        d = _get_store()
        d.clear()
    except RuntimeError:
        pass
