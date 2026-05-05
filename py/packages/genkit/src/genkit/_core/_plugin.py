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
from collections.abc import Sequence

from genkit._core._action import Action, ActionKind
from genkit._core._middleware._base import MiddlewareDesc, _validate_middleware_key_segment
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

    def list_middleware(self) -> list[MiddlewareDesc]:
        """Return middleware descriptors for this plugin to register on the app.

        This runs while Genkit is being constructed, after built-in middleware is
        registered. Use unique flat names without slash characters so they do not
        collide with built-ins or other plugins.

        Returns:
            Descriptors to list in the developer UI and to resolve by name from
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


class _MiddlewareDescsPlugin(Plugin):
    """Plugin implementation that contributes only middleware descriptors."""

    def __init__(self, plugin_name: str, descs: list[MiddlewareDesc]) -> None:
        self.name = plugin_name
        self._descs = descs

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []

    def list_middleware(self) -> list[MiddlewareDesc]:
        return list(self._descs)


def _middleware_registry_name(namespace: str | None, desc_name: str) -> str:
    """Registry key for a descriptor under an optional namespace prefix."""
    if not namespace:
        return desc_name
    return f'{namespace}_{desc_name}'


def middleware_plugin(
    descs: Sequence[MiddlewareDesc],
    *,
    namespace: str | None = None,
) -> Plugin:
    """Wrap a list of middleware descriptors as a single plugin (for ``plugins=[...]``).

    Pass all descriptors for this plugin in one list so one plugin can register several
    middlewares together. Example:

        Genkit(plugins=[
            middleware_plugin([
                new_middleware(PrefixPromptMiddleware),
                new_middleware(OtherMiddleware),
            ], namespace='myapp'),
        ])

    Build each item with ``new_middleware`` in ``genkit.middleware`` or the same API on
    your ``Genkit`` instance; neither registers by itself. Registration happens when this
    plugin is passed in ``plugins=[...]``.

    Args:
        descs: Non-empty sequence of middleware descriptors.
        namespace: If set, becomes the plugin name and each descriptor is registered as
            namespace + underscore + descriptor name (e.g. acme + logging → acme_logging).
            If omitted, the plugin name is ``extension-middleware`` and registry keys
            stay the descriptors' own names. Same flat-segment rules as middleware
            descriptor names (no ``/``, whitespace, ``:``, backslashes, or control
            characters).

    Returns:
        A plugin object whose ``list_middleware`` returns the descriptors (renamed when
        namespace is set).
    """
    built = list(descs)
    if not built:
        raise ValueError(
            'middleware_plugin() needs a non-empty list of MiddlewareDesc instances. '
            + 'Build each with new_middleware(...) from genkit.middleware or ai.new_middleware(...).'
        )
    if not namespace:
        ns = None
    else:
        ns = namespace.strip() or None
    if ns is not None:
        _validate_middleware_key_segment(ns, label='middleware_plugin namespace')

    if ns is None:
        registered = built
    else:
        registered = [d.with_name(_middleware_registry_name(ns, d.name)) for d in built]

    plugin_name = ns if ns is not None else 'extension-middleware'

    return _MiddlewareDescsPlugin(plugin_name, registered)
