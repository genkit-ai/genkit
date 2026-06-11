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

"""Error helpers for the Ollama plugin."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx


class OllamaConnectionError(ConnectionError):
    """Raised when the Ollama server is unreachable.

    Subclasses :class:`ConnectionError` so callers that catch the standard
    exception still work.
    """


@asynccontextmanager
async def wrap_connection_errors(server_address: str) -> AsyncIterator[None]:
    """Translate raw httpx connection failures into actionable errors.

    Yields control to the wrapped block. On :class:`httpx.TransportError`
    (connect failures, timeouts, network and pool errors), re-raises an
    :class:`OllamaConnectionError` whose message tells the user to start
    the Ollama server. HTTP status errors are left untouched so genuine
    server responses are not masked.
    """
    try:
        yield
    except httpx.TransportError as exc:
        raise OllamaConnectionError(
            f'Cannot reach the Ollama server at {server_address}. '
            f'Start it with `ollama serve` (or set server_address to a reachable host).'
        ) from exc
