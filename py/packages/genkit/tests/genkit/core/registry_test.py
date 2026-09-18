#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registry module.

This module contains unit tests for the Registry class and its associated
functionality, ensuring proper registration and management of Genkit resources.
"""

import asyncio

import pytest
from structlog.testing import capture_logs

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
async def test_lookup_mcp_tool_echo_does_not_register_tool_v2_echo() -> None:
    """resolve_action(TOOL, 'mcp:tool/echo') returns the Action. It does not write /tool.v2/echo."""
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
    assert 'inner-tool' not in registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_child_resolve_action_dap_tool_returns_action_without_catalog() -> None:
    """A generate child can peek the parent's DAP Action. Bind still happens in expand."""
    parent = Registry()

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

    define_dynamic_action_provider(parent, 'my-dap', dap_fn)
    child = parent.new_child()

    got = await child.resolve_action(ActionKind.TOOL, 'my-dap:tool/inner-tool')
    assert got is inner
    assert 'inner-tool' not in child._entries.get(ActionKind.TOOL, {})
    assert 'inner-tool' not in parent._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_resolve_action_by_key_dap_qualified() -> None:
    """Qualified DAP key returns the child Action without binding it as a canonical tool."""
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
    assert 'inner-tool' not in registry._entries.get(ActionKind.TOOL, {})

    catalog = await registry.list_actions()
    assert '/dynamic-action-provider/my-dap' in catalog
    assert '/dynamic-action-provider/my-dap:tool/inner-tool' in catalog
    assert '/tool.v2/inner-tool' not in catalog


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
    entry = catalog['/tool.v2/parentplugin/shared-name']
    assert entry.description == 'from child registry'
    assert entry.description != 'from parent plugin'


@pytest.mark.asyncio
async def test_child_dap_child_row_does_not_shadow_parent_plugin_metadata() -> None:
    """A DAP child is its own catalog row; the plugin-advertised ``/tool.v2/`` row is untouched."""

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
    assert catalog['/tool.v2/parentplugin/mcp-tool'].description == 'stale parent schema'


@pytest.mark.asyncio
async def test_list_actions_registered_canonical_coexists_with_qualified_dap_rows() -> None:
    """A DAP child sharing a registered tool's name gets its own row, it does not replace it."""
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

    assert catalog[qualified].description == 'from dap nested'
    assert provider_key in catalog


def test_registry_satisfies_registry_like() -> None:
    """Registry must structurally satisfy RegistryLike so middleware can use it as such."""
    from genkit._core._protocols import RegistryLike
    from genkit._core._registry import Registry

    assert isinstance(Registry(None), RegistryLike)


async def _bg_start(_request: ModelRequest, _ctx: ActionRunContext) -> Operation:
    return Operation(id='bg-op', done=False)


async def _bg_check(op: Operation, _ctx: ActionRunContext) -> Operation:
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


def _dap_child(name: str, description: str) -> Action:
    async def child_fn(x: str) -> str:
        return x

    return Action(kind=ActionKind.TOOL, name=name, fn=child_fn, description=description)


@pytest.mark.asyncio
async def test_list_actions_expands_dap_children() -> None:
    """Every child gets a catalog row under the key resolve_action_by_key accepts."""
    registry = Registry()
    registry.register_action(kind=ActionKind.TOOL, name='local-tool', fn=_identity)

    async def dap_fn() -> DapValue:
        return {'tool': [_dap_child('echo', 'echoes'), _dap_child('add', 'adds')]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    catalog = await registry.list_actions()

    assert '/tool.v2/local-tool' in catalog
    assert '/dynamic-action-provider/mcp' in catalog
    echo = catalog['/dynamic-action-provider/mcp:tool/echo']
    assert echo.key == '/dynamic-action-provider/mcp:tool/echo'
    assert echo.name == 'echo'
    assert echo.action_type == 'tool'
    assert echo.description == 'echoes'
    assert '/dynamic-action-provider/mcp:tool/add' in catalog
    assert await registry.resolve_action_by_key(echo.key) is not None


@pytest.mark.asyncio
async def test_list_actions_without_dap_is_unaffected() -> None:
    """A registry with no provider lists exactly its registered actions."""
    registry = Registry()
    registry.register_action(kind=ActionKind.TOOL, name='local-tool', fn=_identity)
    registry.register_action(kind=ActionKind.CUSTOM, name='local-custom', fn=_identity)

    catalog = await registry.list_actions()

    assert sorted(catalog) == ['/custom/local-custom', '/tool.v2/local-tool']


@pytest.mark.asyncio
async def test_list_actions_survives_a_failing_dap() -> None:
    """One unreachable provider costs its own rows, not the catalog."""
    registry = Registry()
    registry.register_action(kind=ActionKind.TOOL, name='local-tool', fn=_identity)

    async def broken_fn() -> DapValue:
        raise RuntimeError('mcp server is down')

    async def healthy_fn() -> DapValue:
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(registry, 'broken', broken_fn)
    define_dynamic_action_provider(registry, 'healthy', healthy_fn)

    with capture_logs() as logs:
        catalog = await registry.list_actions()

    assert any('Error listing actions' in entry['event'] and 'broken' in entry['event'] for entry in logs)
    assert '/tool.v2/local-tool' in catalog
    assert '/dynamic-action-provider/broken' in catalog
    assert '/dynamic-action-provider/healthy:tool/echo' in catalog
    assert not [key for key in catalog if key.startswith('/dynamic-action-provider/broken:')]


@pytest.mark.asyncio
async def test_list_actions_survives_a_dap_child_without_a_name() -> None:
    """A nameless child is rejected by the provider, and that rejection stays local."""
    registry = Registry()
    registry.register_action(kind=ActionKind.TOOL, name='local-tool', fn=_identity)

    async def nameless_fn() -> DapValue:
        return {'tool': [_dap_child('', 'no name')]}

    define_dynamic_action_provider(registry, 'nameless', nameless_fn)

    catalog = await registry.list_actions()

    assert '/tool.v2/local-tool' in catalog
    assert '/dynamic-action-provider/nameless' in catalog
    assert not [key for key in catalog if key.startswith('/dynamic-action-provider/nameless:')]


@pytest.mark.asyncio
async def test_list_actions_drops_children_of_a_timed_out_dap() -> None:
    """A stalled provider yields no rows, and its fetch keeps running to warm the cache."""
    registry = Registry()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def slow_fn() -> DapValue:
        await release.wait()
        finished.set()
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(registry, 'slow', slow_fn)

    with capture_logs() as logs:
        catalog = await asyncio.wait_for(registry.list_actions(dap_timeout_seconds=0.05), timeout=5)

    assert any('Timed out listing actions' in entry['event'] and 'slow' in entry['event'] for entry in logs)
    assert '/dynamic-action-provider/slow' in catalog
    assert '/dynamic-action-provider/slow:tool/echo' not in catalog

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=5)

    assert '/dynamic-action-provider/slow:tool/echo' in await registry.list_actions()


@pytest.mark.asyncio
async def test_list_actions_holds_a_timed_out_listing_until_it_finishes() -> None:
    """The in-flight fetch is referenced until it completes, then released."""
    from genkit._core._registry import _dap_listing_tasks

    registry = Registry()
    release = asyncio.Event()

    async def slow_fn() -> DapValue:
        await release.wait()
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(registry, 'slow', slow_fn)

    before = set(_dap_listing_tasks)
    await asyncio.wait_for(registry.list_actions(dap_timeout_seconds=0.05), timeout=5)
    abandoned = _dap_listing_tasks - before

    assert len(abandoned) == 1
    (task,) = abandoned

    release.set()
    await task

    assert task not in _dap_listing_tasks


@pytest.mark.asyncio
async def test_list_actions_lists_daps_concurrently() -> None:
    """Listing is concurrent: a provider that waits on a later one still returns rows."""
    registry = Registry()
    second_started = asyncio.Event()

    async def first_fn() -> DapValue:
        await second_started.wait()
        return {'tool': [_dap_child('first', 'first')]}

    async def second_fn() -> DapValue:
        second_started.set()
        return {'tool': [_dap_child('second', 'second')]}

    define_dynamic_action_provider(registry, 'first', first_fn)
    define_dynamic_action_provider(registry, 'second', second_fn)

    catalog = await registry.list_actions(dap_timeout_seconds=5)

    assert '/dynamic-action-provider/first:tool/first' in catalog
    assert '/dynamic-action-provider/second:tool/second' in catalog


@pytest.mark.asyncio
async def test_list_actions_only_lists_children_that_can_be_run() -> None:
    """Every child row must resolve, so the Dev UI never offers a row that fails on run."""
    registry = Registry()

    async def dap_fn() -> DapValue:
        return {'tool': [_dap_child('echo', 'echoes')], ActionKind.TOOL: [_dap_child('shadow', 'wrong bucket')]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    catalog = await registry.list_actions()
    children = [key for key in catalog if key.startswith('/dynamic-action-provider/mcp:')]

    assert children == ['/dynamic-action-provider/mcp:tool/echo']
    for key in children:
        assert await registry.resolve_action_by_key(key) is not None


@pytest.mark.asyncio
async def test_list_actions_skips_children_of_a_namespaced_provider() -> None:
    """A provider name holding a slash cannot host a resolvable child key."""
    registry = Registry()

    async def dap_fn() -> DapValue:
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(registry, 'myplugin/mcp', dap_fn)

    with capture_logs() as logs:
        catalog = await registry.list_actions()

    assert any('their keys cannot be resolved' in entry['event'] for entry in logs)
    assert '/dynamic-action-provider/myplugin/mcp' in catalog
    assert not [key for key in catalog if ':tool/echo' in key]


@pytest.mark.asyncio
async def test_list_actions_releases_a_cancelled_listing() -> None:
    """Cancelling a listing leaves its fetch running and its outcome taken."""
    from genkit._core._registry import _dap_listing_tasks

    registry = Registry()
    started = asyncio.Event()
    release = asyncio.Event()

    async def failing_fn() -> DapValue:
        started.set()
        await release.wait()
        raise RuntimeError('mcp server died')

    define_dynamic_action_provider(registry, 'slow', failing_fn)

    before = set(_dap_listing_tasks)
    listing = asyncio.ensure_future(registry.list_actions())
    await started.wait()
    in_flight = _dap_listing_tasks - before

    assert len(in_flight) == 1
    (task,) = in_flight

    listing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await listing

    release.set()
    with pytest.raises(RuntimeError):
        await task

    assert task not in _dap_listing_tasks


@pytest.mark.asyncio
async def test_list_actions_forwards_the_timeout_to_the_parent_registry() -> None:
    """A child listing must not wait on a parent's stalled provider for the default timeout."""
    parent = Registry()
    release = asyncio.Event()

    async def slow_fn() -> DapValue:
        await release.wait()
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(parent, 'slow', slow_fn)
    child = parent.new_child()

    catalog = await asyncio.wait_for(child.list_actions(dap_timeout_seconds=0.05), timeout=5)

    assert '/dynamic-action-provider/slow' in catalog
    assert '/dynamic-action-provider/slow:tool/echo' not in catalog

    release.set()


@pytest.mark.asyncio
async def test_list_actions_survives_a_third_party_cancelling_a_listing() -> None:
    """A fetch cancelled from outside costs its provider's rows, not the catalog."""
    from genkit._core._registry import _dap_listing_tasks

    registry = Registry()
    registry.register_action(kind=ActionKind.TOOL, name='local-tool', fn=_identity)
    started = asyncio.Event()

    async def stalled_fn() -> DapValue:
        started.set()
        await asyncio.Event().wait()
        return {'tool': [_dap_child('echo', 'echoes')]}

    define_dynamic_action_provider(registry, 'shared', stalled_fn)

    before = set(_dap_listing_tasks)
    listing = asyncio.ensure_future(registry.list_actions())
    await started.wait()
    (task,) = _dap_listing_tasks - before

    task.cancel()
    with capture_logs() as logs:
        catalog = await asyncio.wait_for(listing, timeout=5)

    assert any('was cancelled' in entry['event'] and 'shared' in entry['event'] for entry in logs)
    assert '/tool.v2/local-tool' in catalog
    assert '/dynamic-action-provider/shared' in catalog
    assert not [key for key in catalog if key.startswith('/dynamic-action-provider/shared:')]
