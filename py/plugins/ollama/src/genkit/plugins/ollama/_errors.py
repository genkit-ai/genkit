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


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama server is unreachable.

    Subclasses ``ConnectionError`` so callers catching the standard exception
    still work.
    """


@asynccontextmanager
async def wrap_connection_errors(server_address: str) -> AsyncIterator[None]:
    """Translate transport failures into an actionable OllamaConnectionError.

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
        OllamaConnectionError: If the enclosed block fails to reach the server.
    """
    try:
        yield
    except OllamaConnectionError:
        # Already actionable (e.g. nested wrap); don't re-wrap.
        raise
    except httpx.TimeoutException as exc:
        raise OllamaConnectionError(f'Request to Ollama server at {server_address} timed out.') from exc
    except (httpx.TransportError, ConnectionError) as exc:
        raise OllamaConnectionError(
            f'Cannot reach the Ollama server at {server_address}. '
            f'Start it with `ollama serve` (or set server_address to a reachable host).'
        ) from exc
