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

from __future__ import annotations

import abc
from typing import ClassVar

from genkit._core._action import Action, ActionKind
from genkit._core._middleware import GenerateMiddleware
from genkit._core._typing import ActionMetadata


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
        """Return advertised actions for dev UI/reflection listing.

        ``ActionMetadata.action_type`` must be set (typically ``ActionKind.*``) and
        ``ActionMetadata.name`` must match resolution keys (typically
        ``{plugin.name}/localName`` for plugin-backed actions).
        """
        ...

    def list_middleware(self) -> list[GenerateMiddleware]:
        """Return middleware descriptors for this plugin to register on the app.

        This runs while :class:`Genkit` is being constructed, after
        built-in middleware is registered. Use unique flat names without
        slash characters so they do not collide with built-ins or other
        plugins.

        Returns:
            Descriptors to list in the Dev UI and to resolve by name from
            ``generate(use=...)``.
        """
        return []

    async def model(self, name: str) -> Action | None:
        """Resolve a model action by name (local or namespaced)."""
        target = name if '/' in name else f'{self.name}/{name}'
        return await self.resolve(ActionKind.MODEL, target)

    async def embedder(self, name: str) -> Action | None:
        """Resolve an embedder action by name (local or namespaced)."""
        target = name if '/' in name else f'{self.name}/{name}'
        return await self.resolve(ActionKind.EMBEDDER, target)


class MiddlewarePlugin(Plugin):
    """Plugin that contributes middleware descriptors only.

    Example:
        from genkit import Genkit
        from genkit.middleware import BaseMiddleware
        from genkit.plugin_api import MiddlewarePlugin, new_middleware

        class PrefixPromptMiddleware(BaseMiddleware):
            ...

        class MyMiddlewarePlugin(MiddlewarePlugin):
            name = 'my-middleware'
            middleware = [
                new_middleware(
                    PrefixPromptMiddleware,
                    name='prefix_prompt',
                    description='Prepends a fixed prompt',
                ),
            ]

        ai = Genkit(plugins=[MyMiddlewarePlugin()])
    """

    name: str = ''
    middleware: ClassVar[list[GenerateMiddleware]] = []

    def __init__(self) -> None:
        if not type(self).name:
            raise ValueError(f'{type(self).__name__} must set `name` to the plugin namespace string.')
        if not self.list_middleware():
            raise ValueError(
                f'{type(self).__name__} must provide middleware via the `middleware` class '
                'attribute or a `list_middleware` override. Each entry should come from '
                'new_middleware(YourMiddleware, name=..., description=...).'
            )

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []

    def list_middleware(self) -> list[GenerateMiddleware]:
        return list(type(self).middleware)
