# Copyright 2025 Google LLC
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

"""Connection error helpers for the Ollama plugin."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from genkit.plugin_api import provider_error


@asynccontextmanager
async def wrap_connection_errors(server_address: str) -> AsyncIterator[None]:
    """Turn an unreachable or slow Ollama server into a GenkitError that says how to fix it.

    Catches two flavours of unreachable-server failure:

    - The ``ollama`` SDK intercepts ``httpx.ConnectError`` and re-raises a plain
      :class:`ConnectionError`, so that is the error most paths actually surface.
    - Timeouts the SDK does not intercept (``ReadTimeout``/``PoolTimeout`` and
      friends) bubble up as ``httpx.TransportError``.

    Genuine server responses are left untouched: the SDK turns
    ``httpx.HTTPStatusError`` into ``ollama.ResponseError`` (not caught here), and
    a raw ``HTTPStatusError`` is not a ``TransportError`` either.

    Args:
        server_address: The Ollama server URL, surfaced in the error message.

    Yields:
        None. Wraps the enclosed ``async with`` block.

    Raises:
        GenkitError: DEADLINE_EXCEEDED if the request timed out, UNAVAILABLE
            if the server could not be reached.
    """
    try:
        yield
    except httpx.TimeoutException as exc:
        raise provider_error(
            exc,
            status='DEADLINE_EXCEEDED',
            message=f'Request to Ollama server at {server_address} timed out.',
        ) from exc
    except (httpx.TransportError, ConnectionError) as exc:
        raise provider_error(
            exc,
            status='UNAVAILABLE',
            message=(
                f'Cannot reach the Ollama server at {server_address}. '
                f'Start it with `ollama serve` (or set server_address to a reachable host).'
            ),
        ) from exc
