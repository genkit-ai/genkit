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

"""Leaf module of structural interfaces (Protocols) for core Genkit types.

Keeping interfaces here instead of in their implementation modules breaks
circular-import cycles.  A realistic example in Genkit is the Registry/Plugin/Middleware cycle:

    1. Registry (_registry.py) imports Plugin (_plugin.py) to manage plugins.
    2. Plugin (_plugin.py) imports GenerateMiddleware (_middleware.py) to type-hint list_middleware.
    3. BaseMiddleware/GenerateMiddleware (_middleware.py) need to annotate their request-scoped
       registry attribute, which refers back to Registry.

    If _middleware.py imported Registry from _registry.py directly, it would complete the
    import cycle: _registry -> _plugin -> _middleware -> _registry.

    Solution: BaseMiddleware type-hints with RegistryLike from this leaf module, breaking the cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from genkit._core._action import Action, ActionKind
from genkit._core._typing import Artifact, MessageData


@runtime_checkable
class RegistryLike(Protocol):
    """Structural interface for the subset of Registry used by middleware and the generate engine.

    Add methods as needed.
    """

    def new_child(self) -> RegistryLike:
        """Return a scoped child registry that delegates misses to this one."""
        ...

    def lookup_value(self, kind: str, name: str) -> Any:  # noqa: ANN401
        """Look up a registered value by kind and name."""
        ...

    def register_value(self, kind: str, name: str, value: object) -> None:
        """Register an arbitrary value under kind/name."""
        ...

    def list_values(self, kind: str) -> dict[str, object]:
        """List all values registered under a kind, merged with the parent registry."""
        ...

    def register_action_from_instance(self, action: Action) -> None:
        """Register a pre-built Action instance."""
        ...

    async def resolve_action(self, kind: ActionKind, name: str) -> Action | None:
        """Resolve an action by kind and name, initialising plugins as needed."""
        ...


@runtime_checkable
class SessionLike(Protocol):
    """Structural interface for agent session state peekable from generate middleware.

    The concrete ``Session`` in ``_ai._session`` satisfies this protocol. Middleware
    should treat ``GenerateMiddlewareContext.session`` as optional and only call
    methods when a bind is present.
    """

    async def get_artifacts(self) -> list[Artifact]:
        """Return a copy of artifacts currently stored on the session."""
        ...

    async def add_artifacts(self, *artifacts: Artifact) -> None:
        """Append artifacts, replacing any existing entry with the same name."""
        ...

    async def get_messages(self) -> list[MessageData]:
        """Return a copy of messages currently stored on the session."""
        ...

    async def add_messages(self, *messages: MessageData) -> None:
        """Append messages to the session history."""
        ...

    async def get_custom(self) -> Any:  # noqa: ANN401
        """Return the session's custom state blob, if any."""
        ...

    async def update_custom(self, fn: Callable[[Any], Any]) -> None:  # noqa: ANN401
        """Replace custom state via ``fn(old) -> new``."""
        ...


@runtime_checkable
class GenkitLike(Protocol):
    """Structural interface for the Genkit instance exposed on middleware context."""

    def registry(self) -> RegistryLike:
        """Return the call-scoped registry for this generate invocation."""
        ...

    def current_session(self) -> SessionLike | None:
        """Return the bound agent session, if running inside one."""
        ...
