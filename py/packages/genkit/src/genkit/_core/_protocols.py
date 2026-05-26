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

from typing import Any, Protocol, runtime_checkable

from genkit._core._action import Action, ActionKind


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
