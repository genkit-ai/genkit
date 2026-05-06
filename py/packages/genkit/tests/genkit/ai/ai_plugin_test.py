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

from typing import ClassVar

import pytest

from genkit import Genkit, Message, ModelResponse, Part, Plugin, Role, TextPart
from genkit._core._action import Action, ActionRunContext
from genkit._core._model import ModelRequest
from genkit._core._registry import ActionKind
from genkit._core._typing import ActionMetadata, FinishReason
from genkit.middleware import BaseMiddleware, MiddlewareDesc, new_middleware


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


class _RegistryMw(BaseMiddleware):
    name: ClassVar[str] = 'ai_plugin_test_mw'


class MiddlewareListingPlugin(Plugin):
    """Plugin that contributes middleware via list_middleware."""

    name = 'mw-list-plugin'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []

    def list_middleware(self) -> list[MiddlewareDesc]:
        return [new_middleware(_RegistryMw)]


@pytest.mark.asyncio
async def test_plugin_list_middleware_registers_on_registry() -> None:
    """Descriptors from Plugin.list_middleware appear under list_values('middleware')."""
    ai = Genkit(plugins=[MiddlewareListingPlugin()])
    names = ai.registry.list_values('middleware')
    assert 'ai_plugin_test_mw' in names
    desc = ai.registry.lookup_value('middleware', 'ai_plugin_test_mw')
    assert desc is not None
    assert isinstance(desc, MiddlewareDesc)


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
