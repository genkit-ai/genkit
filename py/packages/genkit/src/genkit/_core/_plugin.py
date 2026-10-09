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

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit._core._action import Action, ActionKind, set_action_name
from genkit._core._middleware import GenerateMiddleware
from genkit._core._typing import ActionMetadata


class PluginConfig(BaseModel):
    """Base configuration model for plugins, middleware, and nested options.

    Accepts snake_case kwargs in Python code and camelCase from serialized JSON/wire.
    Rejects unrecognized extra kwargs to prevent typos in configuration.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        alias_generator=to_camel,
        extra='forbid',
        validate_by_name=True,
        validate_by_alias=True,
    )


def resolved_action_name(*, plugin: str, requested_id: str) -> str:
    """The registry key for an action reached through ``resolve``.

    The caller asked for ``{plugin}/{requested_id}``; that's the id they look
    up later, so the action is stored and named under it even when the plugin
    returned a different name. See the naming rule on :meth:`Plugin.resolve`.
    """
    return f'{plugin}/{requested_id}'


class Plugin(abc.ABC):
    """Abstract base class for Genkit plugins."""

    name: str  # plugin namespace

    @abc.abstractmethod
    async def init(self) -> list[Action]:
        """Lazy warm-up called once per plugin; return actions to pre-register.

        Return names with or without the ``{plugin.name}/`` prefix. A name that
        doesn't start with it is registered under ``{plugin.name}/{name}`` with
        the whole name kept: ``endpoints/123`` becomes ``myplug/endpoints/123``,
        and ``other/x`` becomes ``myplug/other/x``. See :meth:`resolve`.
        """
        ...

    @abc.abstractmethod
    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve a single action by kind and its id inside this plugin.

        Naming rule. When an app asks for ``{plugin.name}/{rest}``:

        1. Genkit removes exactly one ``{plugin.name}/`` from the front and
           passes ``rest`` as ``name``. That's the provider's id, verbatim:
           send it upstream unchanged and don't strip anything from it.
           Provider ids may contain slashes (publisher paths, tuned
           endpoints, ARNs, OpenRouter ids).
        2. The returned action is renamed to exactly what the app asked for
           and stored there, whatever name the plugin gave it, so the same
           string always looks up the same action.

        Return ``None`` to decline; the caller then gets NOT_FOUND.

        Example:
            ```python
            from genkit import Genkit
            from genkit.plugin_api import ActionKind
            from genkit_openai import OpenAI

            # 1. Point the OpenAI plugin at OpenRouter, whose ids carry a vendor
            ai = Genkit(plugins=[OpenAI(base_url='https://openrouter.ai/api/v1')])

            # 2. One `openai/` is the plugin; the rest is the OpenRouter id
            response = await ai.generate(model='openai/openai/gpt-4o', prompt='Suggest a dish.')
            # => resolve(ActionKind.MODEL, 'openai/gpt-4o')
            #    request sent with model='openai/gpt-4o'

            # 3. The action lives under the id the app typed
            action = await ai.registry.resolve_action(ActionKind.MODEL, 'openai/openai/gpt-4o')
            print(action.name)
            # => openai/openai/gpt-4o
            ```
        """
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
        """Resolve a model action by id, with or without this plugin's prefix.

        Follows the naming rule on :meth:`resolve`: one leading
        ``{plugin.name}/`` is removed, the rest goes to ``resolve`` unchanged,
        and the action is named ``{plugin.name}/{rest}``. Any other prefix is
        part of the id, so ``Bedrock().model('googleai/gemini')`` asks Bedrock
        for ``googleai/gemini``.
        """
        return await self._named_resolve(ActionKind.MODEL, name)

    async def embedder(self, name: str) -> Action | None:
        """Resolve an embedder action by id, with or without this plugin's prefix.

        Same naming rule as :meth:`model`.
        """
        return await self._named_resolve(ActionKind.EMBEDDER, name)

    async def _named_resolve(self, kind: ActionKind, name: str) -> Action | None:
        requested_id = name.removeprefix(f'{self.name}/')
        action = await self.resolve(kind, requested_id)
        if action is not None:
            set_action_name(action, resolved_action_name(plugin=self.name, requested_id=requested_id))
        return action


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
