#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the registry module.

This module contains unit tests for the Registry class and its associated
functionality, ensuring proper registration and management of Genkit resources.
"""

import pytest

from genkit.ai import Genkit, Plugin
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry


@pytest.mark.asyncio
async def test_register_action_with_name_and_kind() -> None:
    """Ensure we can register an action with a name and kind."""
    registry = Registry()
    action = registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=lambda x: x)
    got = await registry.resolve_action(ActionKind.CUSTOM, 'test_action')

    assert got == action
    assert got is not None
    assert got.name == 'test_action'
    assert got.kind == ActionKind.CUSTOM


@pytest.mark.asyncio
async def test_resolve_action_by_key() -> None:
    """Ensure we can resolve an action by its key."""
    registry = Registry()
    action = registry.register_action(name='test_action', kind=ActionKind.CUSTOM, fn=lambda x: x)
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

            def model_fn() -> None:
                pass

            return Action(name=name, fn=model_fn, kind=action_type)

        async def list_actions(self) -> list[ActionMetadata]:
            return [ActionMetadata(kind=ActionKind.MODEL, name='foo')]

    ai = Genkit(plugins=[MyPlugin()])

    metas = await ai.registry.list_actions()
    assert metas == [ActionMetadata(kind=ActionKind.MODEL, name='myplugin/foo')]

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
    RecursionError.  See https://github.com/firebase/genkit/issues/4491.
    """
    registry = Registry()

    call_count = 0

    async def self_resolving_factory() -> None:
        nonlocal call_count
        call_count += 1
        # This attempts to resolve the same action, which would trigger
        # _trigger_lazy_loading again.  Without the guard, infinite recursion.
        await registry.resolve_action(ActionKind.CUSTOM, 'self_ref')

    action = registry.register_action(
        kind=ActionKind.CUSTOM,
        name='self_ref',
        fn=lambda: None,
        metadata={'lazy': True},
    )
    setattr(action, '_async_factory', self_resolving_factory)  # noqa: B010

    # Should complete without RecursionError
    resolved = await registry.resolve_action(ActionKind.CUSTOM, 'self_ref')
    assert resolved is not None
    assert resolved.name == 'self_ref'
    # Factory should have been called exactly once (re-entrant call skipped)
    assert call_count == 1
