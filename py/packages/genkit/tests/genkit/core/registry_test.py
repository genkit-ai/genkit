#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registry module.

This module contains unit tests for the Registry class and its associated
functionality, ensuring proper registration and management of Genkit resources.
"""

import pytest

from genkit import Genkit, Plugin
from genkit._core._action import Action, ActionKind, ActionRunContext, create_action_key
from genkit._core._dap import DapValue, define_dynamic_action_provider
from genkit._core._model import ModelRequest, ModelResponse
from genkit._core._registry import Registry
from genkit._core._typing import ActionMetadata, Operation


async def _identity(x: object) -> object:
    return x


@pytest.mark.asyncio
async def test_register_action_with_name_and_kind() -> None:
    """Ensure we can register an action with a name and kind."""
    registry = Registry()
    action = registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=_identity)
    got = await registry.resolve_action(ActionKind.CUSTOM, 'test_action')

    assert got == action
    assert got is not None
    assert got.name == 'test_action'
    assert got.kind == ActionKind.CUSTOM


@pytest.mark.asyncio
async def test_resolve_action_by_key() -> None:
    """Ensure we can resolve an action by its key."""
    registry = Registry()
    action = registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=_identity)
    got = await registry.resolve_action_by_key('/custom/test_action')

    assert got == action
    assert got is not None
    assert got.name == 'test_action'
    assert got.kind == ActionKind.CUSTOM


@pytest.mark.asyncio
async def test_resolve_action_by_key_invalid_format() -> None:
    """Ensure resolve_action_by_key handles invalid key format."""
    registry = Registry()
    with pytest.raises(ValueError, match='Invalid action key format'):
        await registry.resolve_action_by_key('invalid_key')


@pytest.mark.asyncio
async def test_resolve_action_via_dynamic_action_provider() -> None:
    """Registry resolves DAP tools only for DAP-qualified names (host:kind/name)."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    inner = Action(
        name='inner-tool',
        kind=ActionKind.TOOL,
        fn=tool_fn,
        metadata={'name': 'inner-tool'},
    )

    async def dap_fn() -> DapValue:
        return {'tool': [inner]}

    define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    got = await registry.resolve_action(ActionKind.TOOL, 'my-dap:tool/inner-tool')
    assert got is inner


@pytest.mark.asyncio
async def test_resolve_action_by_key_dap_qualified() -> None:
    """DAP-qualified keys resolve nested actions."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    inner = Action(
        name='inner-tool',
        kind=ActionKind.TOOL,
        fn=tool_fn,
        metadata={'name': 'inner-tool'},
    )

    async def dap_fn() -> DapValue:
        return {'tool': [inner]}

    define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    got = await registry.resolve_action_by_key('/dynamic-action-provider/my-dap:tool/inner-tool')
    assert got is inner


@pytest.mark.asyncio
async def test_resolve_action_from_plugin() -> None:
    """Resolve action from plugin test."""
    resolver_calls = []

    class MyPlugin(Plugin):
        name = 'myplugin'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action:
            nonlocal resolver_calls
            resolver_calls.append([action_type, name])

            async def model_fn() -> None:
                pass

            return Action(name=name, fn=model_fn, kind=action_type)

        async def list_actions(self) -> list[ActionMetadata]:
            return [ActionMetadata(action_type=ActionKind.MODEL, name='myplugin/foo')]

    ai = Genkit(plugins=[MyPlugin()])

    catalog = await ai.registry.list_actions()
    assert catalog['/model/myplugin/foo'].name == 'myplugin/foo'

    action = await ai.registry.resolve_action(ActionKind.MODEL, 'myplugin/foo')

    assert action is not None
    assert len(resolver_calls) == 1

    assert resolver_calls == [[ActionKind.MODEL, 'myplugin/foo']]

    # should be idempotent
    await ai.registry.resolve_action(ActionKind.MODEL, 'myplugin/foo')
    assert len(resolver_calls) == 1


def test_register_value() -> None:
    """Register a value and lookup test."""
    registry = Registry()

    registry.register_value('format', 'json', [1, 2, 3])

    assert registry.lookup_value('format', 'json') == [1, 2, 3]


@pytest.mark.asyncio
async def test_trigger_lazy_loading_reentrant_guard() -> None:
    """Regression: _trigger_lazy_loading must not recurse infinitely.

    When a lazy factory resolves its own action key, the re-entrancy guard
    must skip the nested invocation instead of recursing until
    RecursionError.  See https://github.com/genkit-ai/genkit/issues/4491.
    """
    registry = Registry()

    call_count = 0

    async def self_resolving_factory() -> None:
        nonlocal call_count
        call_count += 1
        # This attempts to resolve the same action, which would trigger
        # _trigger_lazy_loading again.  Without the guard, infinite recursion.
        await registry.resolve_action(ActionKind.CUSTOM, 'self_ref')

    async def noop() -> None:
        pass

    action = registry.register_action(
        kind=ActionKind.CUSTOM,
        name='self_ref',
        fn=noop,
        metadata={'lazy': True},
    )
    setattr(action, '_async_factory', self_resolving_factory)  # noqa: B010

    # Should complete without RecursionError
    resolved = await registry.resolve_action(ActionKind.CUSTOM, 'self_ref')
    assert resolved is not None
    assert resolved.name == 'self_ref'
    # Factory should have been called exactly once (re-entrant call skipped)
    assert call_count == 1


# =============================================================================
# Child registry tests
# =============================================================================


@pytest.mark.asyncio
async def test_new_child_is_child() -> None:
    """new_child() returns a child whose is_child is True."""
    parent = Registry()
    child = parent.new_child()
    assert child.is_child
    assert not parent.is_child
    assert child.parent is parent


@pytest.mark.asyncio
async def test_child_resolves_parent_action() -> None:
    """Child registry falls back to parent for resolve_action."""
    parent = Registry()
    action = parent.register_action(name='shared', kind=ActionKind.CUSTOM, fn=_identity)

    child = parent.new_child()
    got = await child.resolve_action(ActionKind.CUSTOM, 'shared')
    assert got is action


@pytest.mark.asyncio
async def test_child_action_does_not_pollute_parent() -> None:
    """Actions registered on child are invisible to parent."""
    parent = Registry()
    child = parent.new_child()
    child.register_action(name='child_only', kind=ActionKind.CUSTOM, fn=_identity)

    assert await parent.resolve_action(ActionKind.CUSTOM, 'child_only') is None
    assert await child.resolve_action(ActionKind.CUSTOM, 'child_only') is not None


@pytest.mark.asyncio
async def test_child_shadows_parent_action() -> None:
    """Child action with the same name takes precedence over parent."""
    parent = Registry()
    parent_action = parent.register_action(name='shared', kind=ActionKind.CUSTOM, fn=_identity)

    child = parent.new_child()

    async def child_fn(x: object) -> object:
        return x

    child_action = child.register_action(name='shared', kind=ActionKind.CUSTOM, fn=child_fn)

    assert await child.resolve_action(ActionKind.CUSTOM, 'shared') is child_action
    assert await parent.resolve_action(ActionKind.CUSTOM, 'shared') is parent_action


def test_child_inherits_default_model() -> None:
    """Child falls back to parent for the default model singleton entry."""
    parent = Registry()
    parent.register_value('defaultModel', 'defaultModel', 'gemini-pro')

    child = parent.new_child()
    assert child.lookup_value('defaultModel', 'defaultModel') == 'gemini-pro'

    child.register_value('defaultModel', 'defaultModel', 'gemini-flash')
    assert child.lookup_value('defaultModel', 'defaultModel') == 'gemini-flash'
    assert parent.lookup_value('defaultModel', 'defaultModel') == 'gemini-pro'


def test_child_inherits_lookup_value() -> None:
    """Child falls back to parent for lookup_value."""
    parent = Registry()
    parent.register_value('format', 'json', {'json': True})

    child = parent.new_child()
    assert child.lookup_value('format', 'json') == {'json': True}

    # Local override shadows parent
    child.register_value('format', 'json', {'json': False})
    assert child.lookup_value('format', 'json') == {'json': False}
    assert parent.lookup_value('format', 'json') == {'json': True}


@pytest.mark.asyncio
async def test_child_resolvable_includes_parent_plugin() -> None:
    """list_actions on child includes parent plugin rows not shadowed locally."""

    class ParentPlugin(Plugin):
        name = 'parentplugin'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            return None

        async def list_actions(self) -> list[ActionMetadata]:
            return [ActionMetadata(action_type=ActionKind.MODEL, name='parentplugin/my-model')]

    parent = Registry()
    parent.register_plugin(ParentPlugin())

    child = parent.new_child()
    catalog = await child.list_actions()
    assert '/model/parentplugin/my-model' in catalog
    assert catalog['/model/parentplugin/my-model'].name == 'parentplugin/my-model'


@pytest.mark.asyncio
async def test_child_resolvable_local_tool_shadows_parent_plugin_metadata() -> None:
    """A tool registered on the child must not inherit parent plugin metadata for the same name."""

    class ParentPlugin(Plugin):
        name = 'parentplugin'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            return None

        async def list_actions(self) -> list[ActionMetadata]:
            return [
                ActionMetadata(
                    action_type=ActionKind.TOOL,
                    name='parentplugin/shared-name',
                    description='from parent plugin',
                )
            ]

    async def local_tool(_: str) -> str:
        return 'local'

    parent = Registry()
    parent.register_plugin(ParentPlugin())
    child = parent.new_child()
    child.register_action(
        kind=ActionKind.TOOL,
        name='parentplugin/shared-name',
        fn=local_tool,
        description='from child registry',
    )

    catalog = await child.list_actions()
    entry = catalog['/tool/parentplugin/shared-name']
    assert entry.description == 'from child registry'
    assert entry.description != 'from parent plugin'


@pytest.mark.asyncio
async def test_child_resolvable_dap_tool_shadows_parent_plugin_metadata() -> None:
    """DAP-exposed nested actions must shadow parent plugin metadata for the same (kind, name)."""

    class ParentPlugin(Plugin):
        name = 'parentplugin'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            return None

        async def list_actions(self) -> list[ActionMetadata]:
            return [
                ActionMetadata(
                    action_type=ActionKind.TOOL,
                    name='parentplugin/mcp-tool',
                    description='stale parent schema',
                )
            ]

    async def mcp_tool_fn(_: str) -> str:
        return 'mcp'

    mcp_tool = Action(
        kind=ActionKind.TOOL,
        name='parentplugin/mcp-tool',
        fn=mcp_tool_fn,
        description='from mcp',
    )

    parent = Registry()
    parent.register_plugin(ParentPlugin())
    child = parent.new_child()

    async def dap_fn() -> DapValue:
        return {'tool': [mcp_tool]}

    define_dynamic_action_provider(child, 'mcp', dap_fn)

    catalog = await child.list_actions()
    qualified = create_action_key(ActionKind.DYNAMIC_ACTION_PROVIDER, 'mcp:tool/parentplugin/mcp-tool')
    assert catalog[qualified].description == 'from mcp'
    assert catalog['/tool/parentplugin/mcp-tool'].description == 'stale parent schema'


@pytest.mark.asyncio
async def test_list_actions_registered_canonical_coexists_with_qualified_dap_rows() -> None:
    """Registered ``/tool/...`` row coexists with DAP ``/dynamic-action-provider/...`` rows when shortnames collide."""
    tool_name = 'suite/same-canonical'

    async def registered_fn(_: str) -> str:
        return 'registered'

    async def dap_nested_fn(_: str) -> str:
        return 'dap'

    dap_nested = Action(
        kind=ActionKind.TOOL,
        name=tool_name,
        fn=dap_nested_fn,
        description='from dap nested',
    )

    registry = Registry()
    registry.register_action(
        kind=ActionKind.TOOL,
        name=tool_name,
        fn=registered_fn,
        description='from registry registration',
    )

    async def dap_fn() -> DapValue:
        return {'tool': [dap_nested]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    catalog = await registry.list_actions()

    canonical = create_action_key(ActionKind.TOOL, tool_name)
    record_key = f'mcp:tool/{tool_name}'
    qualified = create_action_key(ActionKind.DYNAMIC_ACTION_PROVIDER, record_key)
    provider_key = create_action_key(ActionKind.DYNAMIC_ACTION_PROVIDER, 'mcp')

    assert canonical in catalog
    assert catalog[canonical].description == 'from registry registration'

    assert qualified in catalog
    assert catalog[qualified].key == qualified

    assert provider_key in catalog


def test_registry_satisfies_registry_like() -> None:
    """Registry must structurally satisfy RegistryLike so middleware can use it as such."""
    from genkit._core._protocols import RegistryLike
    from genkit._core._registry import Registry

    assert isinstance(Registry(None), RegistryLike)


async def _bg_start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
    return Operation(id='bg-op', done=False)


async def _bg_check(op: Operation) -> Operation:
    return op


@pytest.mark.asyncio
async def test_resolve_model_finds_background_model() -> None:
    """A name registered only as a background model is still findable."""
    ai = Genkit()
    action = ai.define_background_model(name='bg-model', start=_bg_start, check=_bg_check)

    got = await ai.registry.resolve_model('bg-model')

    assert got is not None
    assert got is action.start_action
    assert got.kind == ActionKind.BACKGROUND_MODEL


@pytest.mark.asyncio
async def test_resolve_model_prefers_foreground_when_both_exist() -> None:
    """A normal model of the same name wins. The fallback is only for names that have no MODEL."""

    async def fg(_request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return ModelResponse()

    ai = Genkit()
    foreground = ai.define_model(name='same-name', fn=fg)
    ai.define_background_model(name='same-name', start=_bg_start, check=_bg_check)

    got = await ai.registry.resolve_model('same-name')

    assert got is not None
    assert got is foreground
    assert got.kind == ActionKind.MODEL


@pytest.mark.asyncio
async def test_resolve_model_missing_is_none() -> None:
    """Unknown names stay None. This is not NOT_FOUND — callers decide the error."""
    ai = Genkit()
    assert await ai.registry.resolve_model('no-such-model') is None


@pytest.mark.asyncio
async def test_resolve_model_finds_plugin_background_model() -> None:
    """A plugin MODEL miss still lets the BACKGROUND_MODEL start action through."""

    class VeoPlugin(Plugin):
        name = 'plug'

        async def init(self) -> list[Action]:
            return []

        async def list_actions(self) -> list[ActionMetadata]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            if action_type != ActionKind.BACKGROUND_MODEL:
                return None
            return Action(name=name, kind=ActionKind.BACKGROUND_MODEL, fn=_bg_start)

    ai = Genkit(plugins=[VeoPlugin()])
    got = await ai.registry.resolve_model('plug/veo-2.0-generate-001')

    assert got is not None
    assert got.kind == ActionKind.BACKGROUND_MODEL
    assert got.name == 'plug/veo-2.0-generate-001'
