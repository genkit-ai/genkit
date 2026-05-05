# Copyright 2026 Google LLC
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

# Note: ty type checker has a known limitation with StrEnum where it sees
# enum members as Literal values instead of the enum type. We use ty: ignore
# comments to suppress these false positives. See: https://github.com/python/typing/issues/1367

"""Tests for AI plugin functionality."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from genkit import Genkit, GenkitError, Message, ModelResponse, Part, Plugin, Role, TextPart, middleware_plugin
from genkit._core._action import Action, ActionRunContext
from genkit._core._middleware._base import BaseMiddleware
from genkit._core._model import ModelHookParams, ModelRequest
from genkit._core._registry import ActionKind
from genkit._core._typing import ActionMetadata, FinishReason, MiddlewareRef
from genkit.middleware import MiddlewareDesc, new_middleware


class AsyncResolveOnlyPlugin(Plugin):
    """Plugin that only implements async resolve."""

    name = 'async-resolve-only'

    async def init(self) -> list[Action]:
        """Initialize the plugin."""
        # Intentionally register nothing eagerly.
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action."""
        if action_type != ActionKind.MODEL:
            return None
        if name != f'{self.name}/lazy-model':
            return None

        async def _generate(req: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            return ModelResponse(
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='OK: lazy'))]),
                finish_reason=FinishReason.STOP,
            )

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_generate,
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List available actions."""
        return [
            ActionMetadata(
                action_type=ActionKind.MODEL,
                name=f'{self.name}/lazy-model',
            )
        ]


class AsyncInitPlugin(Plugin):
    """Plugin that implements async init."""

    name = 'async-init-plugin'

    async def init(self) -> list[Action]:
        """Initialize the plugin."""
        action = await self.resolve(ActionKind.MODEL, f'{self.name}/init-model')
        return [action] if action else []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action."""
        if action_type != ActionKind.MODEL:
            return None
        if name != f'{self.name}/init-model':
            return None

        async def _generate(req: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            return ModelResponse(
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='OK: resolve'))]),
                finish_reason=FinishReason.STOP,
            )

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_generate,
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List available actions."""
        return [
            ActionMetadata(
                action_type=ActionKind.MODEL,
                name=f'{self.name}/init-model',
            )
        ]


class _NoopPluginMiddleware(BaseMiddleware):
    """Minimal middleware for plugin registration tests."""

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next_fn(params)


class _ClassMetaMiddleware(BaseMiddleware):
    """Registered via ``new_middleware(MyClass)`` using class attributes."""

    name = 'class_meta_mw'
    description = 'from class attrs'

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next_fn(params)


class PluginWithMiddleware(Plugin):
    """Plugin that contributes descriptors via ``list_middleware``."""

    name = 'plugin-with-mw'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []

    def list_middleware(self) -> list[MiddlewareDesc]:
        def _factory(_config: dict[str, Any] | None) -> BaseMiddleware:
            return _NoopPluginMiddleware()

        return [
            MiddlewareDesc(
                name=f'{self.name}_noop',
                factory=_factory,
                description='noop plugin middleware for tests',
            )
        ]


def test_plugin_list_middleware_registers_on_registry() -> None:
    """Descriptors returned from ``Plugin.list_middleware`` register after built-ins."""
    ai = Genkit(plugins=[PluginWithMiddleware()])
    desc = ai.registry.lookup_value('middleware', 'plugin-with-mw_noop')
    assert isinstance(desc, MiddlewareDesc)
    assert desc.name == 'plugin-with-mw_noop'
    assert desc.model_dump(by_alias=True, exclude_none=True)['description'] == 'noop plugin middleware for tests'


class _PluginPackMw(BaseMiddleware):
    name = 'my_pack_custom_mw'
    description = 'from plugin() helper'

    async def wrap_model(
        self,
        params: ModelHookParams,
        next_fn: Callable[[ModelHookParams], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await next_fn(params)


def test_middleware_plugin_registers_on_registry() -> None:
    """``middleware_plugin`` wraps descriptors in a minimal ``Plugin``."""
    desc = new_middleware(_PluginPackMw)
    plugin = middleware_plugin([desc])
    assert plugin.name == 'extension-middleware'
    ai = Genkit(plugins=[plugin])
    looked = ai.registry.lookup_value('middleware', 'my_pack_custom_mw')
    assert isinstance(looked, MiddlewareDesc)
    assert looked.name == 'my_pack_custom_mw'
    assert looked.model_dump(by_alias=True, exclude_none=True)['description'] == 'from plugin() helper'


class _MwOne(BaseMiddleware):
    name = 'mw_one'


class _MwTwo(BaseMiddleware):
    name = 'mw_two'


def test_middleware_plugin_registers_multiple_descs() -> None:
    """One plugin can bundle several ``MiddlewareDesc`` instances."""
    d1 = new_middleware(_MwOne)
    d2 = new_middleware(_MwTwo)
    ai = Genkit(plugins=[middleware_plugin([d1, d2])])
    assert ai.registry.lookup_value('middleware', 'mw_one') is not None
    assert ai.registry.lookup_value('middleware', 'mw_two') is not None


class _LoggingMw(BaseMiddleware):
    name = 'logging'
    description = 'd'


def test_middleware_plugin_namespace_prefixes_registry_key() -> None:
    """With ``namespace``, keys are ``{namespace}_{desc.name}``; ``Plugin.name`` defaults to namespace."""
    g = new_middleware(_LoggingMw)
    plugin = middleware_plugin([g], namespace='acme')
    assert plugin.name == 'acme'
    ai = Genkit(plugins=[plugin])
    assert ai.registry.lookup_value('middleware', 'acme_logging') is not None
    assert ai.registry.lookup_value('middleware', 'logging') is None


class _XMw(BaseMiddleware):
    name = 'x'


def test_middleware_plugin_namespace_must_not_contain_slash() -> None:
    with pytest.raises(ValueError, match='namespace'):
        g = new_middleware(_XMw)
        middleware_plugin([g], namespace='bad/ns')


class _BadSlashNameMw(BaseMiddleware):
    name = 'bad/name'


def test_new_middleware_name_must_not_contain_slash() -> None:
    """``/`` is reserved for model/action style keys; middleware ids are flat."""
    with pytest.raises(ValueError, match='path-free'):
        new_middleware(_BadSlashNameMw)


class _BadSpaceNameMw(BaseMiddleware):
    name = 'bad name'


def test_new_middleware_name_rejects_whitespace_in_segment() -> None:
    with pytest.raises(ValueError, match='path-free'):
        new_middleware(_BadSpaceNameMw)


class _BadColonNameMw(BaseMiddleware):
    name = 'bad:name'


def test_new_middleware_name_rejects_colon_in_segment() -> None:
    with pytest.raises(ValueError, match='path-free'):
        new_middleware(_BadColonNameMw)


def test_middleware_plugin_namespace_rejects_colon() -> None:
    g = new_middleware(_XMw)
    with pytest.raises(ValueError, match='middleware_plugin namespace'):
        middleware_plugin([g], namespace='acme:ns')


def test_new_middleware_from_class_reads_name_and_description() -> None:
    """``new_middleware(MyMiddleware)`` uses ``name`` / ``description`` class attributes."""
    desc = new_middleware(_ClassMetaMiddleware)
    assert desc.name == 'class_meta_mw'
    assert desc.description == 'from class attrs'
    ai = Genkit(plugins=[middleware_plugin([desc])])
    assert ai.registry.lookup_value('middleware', 'class_meta_mw') is not None


def test_new_middleware_from_class_requires_name() -> None:
    class EmptyNameMiddleware(BaseMiddleware):
        pass

    with pytest.raises(ValueError, match='name must be set'):
        new_middleware(EmptyNameMiddleware)


def test_genkit_new_middleware_accepts_class_form() -> None:
    """``Genkit.new_middleware`` builds a descriptor but does not register it."""
    ai = Genkit()
    gm = ai.new_middleware(_ClassMetaMiddleware)
    assert gm.name == 'class_meta_mw'
    assert ai.registry.lookup_value('middleware', 'class_meta_mw') is None

    ai2 = Genkit(plugins=[middleware_plugin([gm])])
    assert ai2.registry.lookup_value('middleware', 'class_meta_mw') is gm


def test_middleware_plugin_requires_at_least_one_desc() -> None:
    """Calling ``middleware_plugin`` with an empty list raises."""
    with pytest.raises(ValueError, match='non-empty'):
        middleware_plugin([])


@pytest.mark.asyncio
async def test_unknown_middleware_ref_raises_genkit_error() -> None:
    """A ``MiddlewareRef`` name that is not registered surfaces a clear ``GenkitError``."""
    ai = Genkit(plugins=[AsyncResolveOnlyPlugin()])
    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(
            model='async-resolve-only/lazy-model',
            prompt='hello',
            use=[MiddlewareRef(name='not-registered-middleware', config=None)],
        )
    assert exc_info.value.status == 'NOT_FOUND'
    assert 'not-registered-middleware' in str(exc_info.value)
    assert 'middleware_plugin' in str(exc_info.value)


@pytest.mark.asyncio
async def test_async_resolve_is_awaited_via_generate() -> None:
    """Test that async resolve is awaited when calling generate."""
    ai = Genkit(plugins=[AsyncResolveOnlyPlugin()])
    resp = await ai.generate(model='async-resolve-only/lazy-model', prompt='hello')
    assert resp.text == 'OK: lazy'


@pytest.mark.asyncio
async def test_async_init_is_awaited_via_generate() -> None:
    """Test that async init is awaited when calling generate."""
    ai = Genkit(plugins=[AsyncInitPlugin()])
    resp = await ai.generate(model='async-init-plugin/init-model', prompt='hello')
    assert resp.text == 'OK: resolve'
