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
circular-import cycles.  The pattern is:

    Module A needs B's type in annotations, but B depends on A → cycle.
    Solution: extract B's interface here; A imports the interface, not B.

Currently defined:

- ``RegistryLike`` — structural Protocol covering the registry methods that
                   middleware and the generate engine actually call.  Use instead
                   of the concrete ``Registry`` whenever a cycle would result.
                   The real ``Registry`` satisfies it structurally; no
                   ``register`` call or inheritance is needed.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Protocol, runtime_checkable

from genkit._core._action import Action, ActionKind


@runtime_checkable
class RegistryLike(Protocol):
    """Structural interface for the subset of Registry used by middleware and the generate engine.

    Middleware plugins (e.g. ``Fallback``) and ``generate_action`` depend only
    on this interface, not the full concrete ``Registry``.  This avoids pulling
    in ``_registry.py`` from modules that ``_registry.py`` itself depends on.

    The concrete ``Registry`` satisfies this protocol structurally — no
    subclassing or registration is required.
    """

    def new_child(self) -> RegistryLike:
        """Return a scoped child registry that delegates misses to this one."""
        ...

    def lookup_value(self, kind: str, name: str) -> Any:
        """Look up a registered value by kind and name."""
        ...

    def register_value(self, kind: str, name: str, value: object) -> None:
        """Register an arbitrary value under kind/name."""
        ...

    def register_action_from_instance(self, action: Action) -> None:
        """Register a pre-built Action instance."""
        ...

    async def resolve_action(self, kind: ActionKind, name: str) -> Action | None:
        """Resolve an action by kind and name, initialising plugins as needed."""
        ...
