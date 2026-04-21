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

"""Abstract base class for Genkit plugins."""

import abc

from genkit._core._action import Action, ActionKind, ActionMetadata


class Plugin(abc.ABC):
    """Abstract base class for Genkit plugins."""

    name: str  # plugin namespace

    @abc.abstractmethod
    async def init(self) -> list[Action]:
        """Lazy warm-up called once per plugin; return actions to pre-register."""
        ...

    @abc.abstractmethod
    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve a single action by kind and namespaced name."""
        ...

    @abc.abstractmethod
    async def list_actions(self) -> list[ActionMetadata]:
        """Return advertised actions for dev UI/reflection listing."""
        ...

    async def model(self, name: str) -> Action | None:
        """Resolve a model action by name (local or namespaced)."""
        target = name if '/' in name else f'{self.name}/{name}'
        return await self.resolve(ActionKind.MODEL, target)

    async def embedder(self, name: str) -> Action | None:
        """Resolve an embedder action by name (local or namespaced)."""
        target = name if '/' in name else f'{self.name}/{name}'
        return await self.resolve(ActionKind.EMBEDDER, target)
