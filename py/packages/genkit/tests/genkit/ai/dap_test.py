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

"""Tests for the Dynamic Action Provider (DAP) module."""

import asyncio

import pytest

from genkit._core._action import Action, ActionKind
from genkit._core._dap import (
    DapValue,
    DynamicActionProvider,
    define_dynamic_action_provider,
    is_dynamic_action_provider,
)
from genkit._core._registry import Registry


@pytest.fixture
def registry() -> Registry:
    return Registry()


@pytest.fixture
def tool1(registry: Registry) -> Action:
    async def tool1_fn(input: str) -> str:
        return 'tool1'

    return registry.register_action(
        name='tool1',
        kind=ActionKind.TOOL,
        fn=tool1_fn,
        metadata={'name': 'tool1'},
    )


@pytest.fixture
def tool2(registry: Registry) -> Action:
    async def tool2_fn(input: str) -> str:
        return 'tool2'

    return registry.register_action(
        name='tool2',
        kind=ActionKind.TOOL,
        fn=tool2_fn,
        metadata={'name': 'tool2'},
    )


@pytest.fixture
def other_tool(registry: Registry) -> Action:
    async def other_tool_fn(input: str) -> str:
        return 'other'

    return registry.register_action(
        name='other-tool',
        kind=ActionKind.TOOL,
        fn=other_tool_fn,
        metadata={'name': 'other-tool'},
    )


@pytest.mark.asyncio
async def test_gets_specific_action(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    action = await dap.get_action('tool', 'tool1')
    assert action is tool1
    assert call_count == 1


@pytest.mark.asyncio
async def test_lists_action_metadata(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    metadata = await dap.list_action_metadata('tool', '*')
    assert len(metadata) == 2
    assert metadata[0] == tool1.metadata
    assert metadata[1] == tool2.metadata
    assert call_count == 1


@pytest.mark.asyncio
async def test_caches_actions(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    action = await dap.get_action('tool', 'tool1')
    assert action is tool1
    assert call_count == 1

    # This should be cached
    action = await dap.get_action('tool', 'tool2')
    assert action is tool2
    assert call_count == 1

    metadata = await dap.list_action_metadata('tool', '*')
    assert len(metadata) == 2
    assert call_count == 1


@pytest.mark.asyncio
async def test_invalidates_cache(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    await dap.get_action('tool', 'tool1')
    assert call_count == 1

    dap.invalidate_cache()

    await dap.get_action('tool', 'tool2')
    assert call_count == 2


@pytest.mark.asyncio
async def test_respects_cache_ttl(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn, cache_ttl_millis=10)

    await dap.get_action('tool', 'tool1')
    assert call_count == 1

    # Wait for TTL to expire
    await asyncio.sleep(0.025)  # 25ms > 10ms TTL

    await dap.get_action('tool', 'tool2')
    assert call_count == 2


@pytest.mark.asyncio
async def test_lists_actions_with_prefix(registry: Registry, tool1: Action, tool2: Action, other_tool: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2, other_tool]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    metadata = await dap.list_action_metadata('tool', 'tool*')
    assert len(metadata) == 2
    assert metadata[0] == tool1.metadata
    assert metadata[1] == tool2.metadata
    assert call_count == 1


@pytest.mark.asyncio
async def test_lists_actions_exact_match(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    metadata = await dap.list_action_metadata('tool', 'tool1')
    assert len(metadata) == 1
    assert metadata[0] == tool1.metadata
    assert call_count == 1


@pytest.mark.asyncio
async def test_gets_action_metadata_record(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {
            'tool': [tool1, tool2],
            'flow': [tool1],
        }

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    record = await dap.get_action_metadata_record('dap/my-dap')
    assert 'dap/my-dap:tool/tool1' in record
    assert 'dap/my-dap:tool/tool2' in record
    assert 'dap/my-dap:flow/tool1' in record
    assert record['dap/my-dap:tool/tool1'] == tool1.metadata
    assert record['dap/my-dap:tool/tool2'] == tool2.metadata
    assert record['dap/my-dap:flow/tool1'] == tool1.metadata
    assert call_count == 1


@pytest.mark.asyncio
async def test_handles_concurrent_requests(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    results = await asyncio.gather(
        dap.list_action_metadata('tool', '*'),
        dap.list_action_metadata('tool', '*'),
    )

    metadata1, metadata2 = results
    assert len(metadata1) == 2
    assert len(metadata2) == 2
    assert metadata1[0] == tool1.metadata
    assert metadata2[0] == tool1.metadata
    assert call_count == 1


@pytest.mark.asyncio
async def test_handles_fetch_errors(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError('Fetch failed')
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    with pytest.raises(RuntimeError, match='Fetch failed'):
        await dap.list_action_metadata('tool', '*')
    assert call_count == 1

    metadata = await dap.list_action_metadata('tool', '*')
    assert len(metadata) == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_identifies_dap(registry: Registry, tool1: Action) -> None:
    async def dap_fn() -> DapValue:
        return {}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)
    assert is_dynamic_action_provider(dap) is True
    assert is_dynamic_action_provider(tool1) is False


@pytest.mark.asyncio
async def test_get_action_returns_none_for_unknown_type(registry: Registry, tool1: Action) -> None:
    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    action = await dap.get_action('unknown-type', 'tool1')
    assert action is None


@pytest.mark.asyncio
async def test_get_action_returns_none_for_unknown_name(registry: Registry, tool1: Action) -> None:
    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    action = await dap.get_action('tool', 'unknown-name')
    assert action is None


@pytest.mark.asyncio
async def test_list_action_metadata_returns_empty_for_unknown_type(registry: Registry, tool1: Action) -> None:
    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    metadata = await dap.list_action_metadata('unknown-type', '*')
    assert metadata == []


@pytest.mark.asyncio
async def test_negative_ttl_disables_caching(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn, cache_ttl_millis=-1)

    await dap.get_action('tool', 'tool1')
    assert call_count == 1

    # With negative TTL, this should trigger another fetch
    await dap.get_action('tool', 'tool2')
    assert call_count == 2


@pytest.mark.asyncio
async def test_zero_ttl_uses_default(registry: Registry, tool1: Action, tool2: Action) -> None:
    call_count = 0

    async def dap_fn() -> DapValue:
        nonlocal call_count
        call_count += 1
        return {'tool': [tool1, tool2]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn, cache_ttl_millis=0)

    await dap.get_action('tool', 'tool1')
    assert call_count == 1

    # With default TTL (3s), this should still be cached
    await dap.get_action('tool', 'tool2')
    assert call_count == 1


@pytest.mark.asyncio
async def test_get_action_metadata_record_raises_on_missing_name(registry: Registry) -> None:
    async def nameless_fn(input: str) -> str:
        return 'nameless'

    nameless_action = registry.register_action(
        name='nameless',
        kind=ActionKind.TOOL,
        fn=nameless_fn,
        metadata={},
    )
    nameless_action._name = ''

    async def dap_fn() -> DapValue:
        return {'tool': [nameless_action]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    with pytest.raises(ValueError, match='name required'):
        await dap.get_action_metadata_record('dap/my-dap')


def test_define_dap_with_full_options(registry: Registry) -> None:
    async def dap_fn() -> DapValue:
        return {}

    dap = define_dynamic_action_provider(
        registry,
        'full-config-dap',
        dap_fn,
        description='A DAP with all options',
        cache_ttl_millis=5000,
        metadata={'custom': 'value'},
    )
    assert isinstance(dap, DynamicActionProvider)
