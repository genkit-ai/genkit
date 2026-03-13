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

"""ASGI-compatible type definitions for Genkit web framework integration.

This module provides Protocol-based type definitions that are compatible with
any ASGI framework (litestar, starlette, fastapi, Django, etc.) without
requiring framework-specific type unions.

The ASGI specification defines interfaces using structural typing, so we use
typing.Protocol to match this approach. Any framework that follows the ASGI
spec will be compatible with these types.

Supported frameworks:
    - asgiref (Django)
    - FastAPI
    - Litestar
    - Starlette
    - Any other ASGI-compliant framework

Example:
    ```python
    from genkit.web.typing import Scope, Receive, Send


    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        # This signature works with any ASGI framework
        ...
    ```

See Also:
    - ASGI Spec: https://asgi.readthedocs.io/
    - PEP 544 (Protocols): https://peps.python.org/pep-0544/
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, Protocol, runtime_checkable

# These Protocol-based types follow the ASGI specification and are compatible
# with any ASGI framework. Using Protocols instead of Union types allows:
#
# 1. Any framework implementing the ASGI interface to work
# 2. No import-time dependencies on specific frameworks
# 3. Proper structural subtyping for type checkers

# ASGI Scope - a dict-like object with connection info
# Using MutableMapping[str, Any] as the base provides structural compatibility
Scope = MutableMapping[str, Any]
"""ASGI scope - connection metadata dict.

Contains at minimum:
    - type: 'http', 'websocket', or 'lifespan'
    - asgi: dict with 'version' key

For HTTP connections, also includes:
    - method: HTTP method (GET, POST, etc.)
    - path: URL path
    - headers: list of (name, value) byte tuples
"""

# ASGI Receive Callable
# Async function that returns the next message from the client
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
"""ASGI receive callable - async function to get next message from client."""

# ASGI Send Callable
# Async function that sends a message to the client
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
"""ASGI send callable - async function to send message to client."""


@runtime_checkable
class ASGIApp(Protocol):
    """Protocol for ASGI applications.

    Any async callable with the signature (scope, receive, send) -> None
    is considered an ASGI application.

    Example:
        ```python
        async def my_app(scope: Scope, receive: Receive, send: Send) -> None:
            await send({
                'type': 'http.response.start',
                'status': 200,
                'headers': [(b'content-type', b'text/plain')],
            })
            await send({
                'type': 'http.response.body',
                'body': b'Hello, World!',
            })


        # my_app is an ASGIApp
        assert isinstance(my_app, ASGIApp)
        ```
    """

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Handle an ASGI connection."""
        ...


# Type alias for ASGI applications
# Using Any because each framework (Litestar, Starlette, FastAPI) has its own
# type definitions that aren't structurally compatible with our Protocol.
# At runtime, they all work correctly - this is purely a type checker limitation.
Application = Any
"""Type alias for ASGI application objects.

Note: Uses Any because external frameworks (Litestar, Starlette, etc.) define
their own ASGI types that aren't structurally compatible with our Protocol.
At runtime, all ASGI-compliant apps work correctly.
"""


# Specialized Scope Types (for documentation, not enforced at runtime)
# These are aliases for Scope with documentation about expected keys.
# Type checkers treat them as Scope, but the docstrings help developers.

HTTPScope = Scope
"""HTTP connection scope.

Expected keys beyond base Scope:
    - method: str (GET, POST, etc.)
    - path: str
    - query_string: bytes
    - headers: list[tuple[bytes, bytes]]
"""

WebSocketScope = Scope
"""WebSocket connection scope.

Expected keys beyond base Scope:
    - path: str
    - query_string: bytes
    - headers: list[tuple[bytes, bytes]]
"""

LifespanScope = Scope
"""Lifespan scope for startup/shutdown.

Expected keys beyond base Scope:
    - type: 'lifespan'
"""


# Handler Type Aliases

LifespanHandler = Callable[[LifespanScope, Receive, Send], Awaitable[None]]
"""ASGI lifespan handler - manages app startup and shutdown."""

StartupHandler = Callable[[], Awaitable[None]]
"""Simple startup/shutdown handler (0-argument async function).

Used by Starlette/Litestar for on_startup/on_shutdown callbacks.
This is distinct from LifespanHandler which is the full ASGI protocol.
"""
