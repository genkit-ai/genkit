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
import concurrent.futures
import contextlib
import sys
import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

from genkit._core import _dap
from genkit._core._action import Action, ActionKind
from genkit._core._dap import (
    DapMetadata,
    DapValue,
    DynamicActionProvider,
    define_dynamic_action_provider,
    is_dynamic_action_provider,
)
from genkit._core._registry import Registry
from genkit._core._typing import ActionMetadata


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
async def test_dap_authors_write_tool_not_tool_v2(registry: Registry, tool1: Action) -> None:
    """Authors write {tool: [...]}. That is what mcp:tool/echo reads."""

    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    assert await dap.get_action('tool', 'tool1') is tool1
    listed = await dap.list_action_metadata_by_key('my-dap')
    assert list(listed) == ['/dynamic-action-provider/my-dap:tool/tool1']
    assert listed['/dynamic-action-provider/my-dap:tool/tool1'].action_type == 'tool'


@pytest.mark.asyncio
async def test_dap_tool_v2_bucket_is_not_found_by_mcp_tool_echo(registry: Registry, tool1: Action) -> None:
    """{tool.v2: [...]} is missed by mcp:tool/echo."""

    async def dap_fn() -> DapValue:
        return {ActionKind.TOOL: [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    assert await dap.get_action('tool', 'tool1') is None
    assert await dap.list_action_metadata('tool', '*') == []
    listed = await dap.list_action_metadata_by_key('my-dap')
    assert list(listed) == ['/dynamic-action-provider/my-dap:tool.v2/tool1']
    assert '/dynamic-action-provider/my-dap:tool/tool1' not in listed


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

    record = await dap.list_action_metadata_by_key('my-dap')
    tool1_key = '/dynamic-action-provider/my-dap:tool/tool1'
    tool2_key = '/dynamic-action-provider/my-dap:tool/tool2'
    flow1_key = '/dynamic-action-provider/my-dap:flow/tool1'
    assert tool1_key in record
    assert tool2_key in record
    assert flow1_key in record
    tool1_meta = record[tool1_key]
    assert tool1_meta.key == tool1_key
    assert tool1_meta.name == 'tool1'
    assert tool1_meta.action_type == 'tool'
    assert tool1_meta.description == tool1.description
    assert tool1_meta.input_schema == tool1.input_schema
    assert tool1_meta.output_schema == tool1.output_schema
    assert tool1_meta.metadata == tool1.metadata
    assert record[tool2_key].name == 'tool2'
    assert record[tool2_key].action_type == 'tool'
    assert record[tool2_key].metadata == tool2.metadata
    assert record[flow1_key].name == 'tool1'
    assert record[flow1_key].action_type == 'flow'
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
async def test_list_action_metadata_by_key_raises_on_missing_name(registry: Registry) -> None:
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
        await dap.list_action_metadata_by_key('my-dap')


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


@contextlib.contextmanager
def _background_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """Run an event loop on its own thread for the duration of the block."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def test_concurrent_listing_from_two_loops(registry: Registry, tool1: Action) -> None:
    """A second loop listing during an in-flight fetch must not await the first loop's task.

    Under ``genkit start`` the reflection server owns a loop on its own thread, so it
    lists a provider while the app loop may be mid-fetch.
    """
    entered = threading.Semaphore(0)
    gate: concurrent.futures.Future[None] = concurrent.futures.Future()

    async def dap_fn() -> DapValue:
        entered.release()
        await asyncio.wrap_future(gate)
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    with _background_loop() as app_loop, _background_loop() as reflection_loop:
        app_listing = asyncio.run_coroutine_threadsafe(dap.list_action_metadata_by_key('my-dap'), app_loop)
        assert entered.acquire(timeout=5), 'app loop never started its fetch'

        reflection_listing = asyncio.run_coroutine_threadsafe(
            dap.list_action_metadata_by_key('my-dap'), reflection_loop
        )
        joined_in_flight = entered.acquire(timeout=5)

        gate.set_result(None)
        app_rows = app_listing.result(timeout=5)
        reflection_rows = reflection_listing.result(timeout=5)

    assert joined_in_flight, 'reflection loop never reached the fetch'
    assert list(app_rows) == ['/dynamic-action-provider/my-dap:tool/tool1']
    assert list(reflection_rows) == ['/dynamic-action-provider/my-dap:tool/tool1']


def test_listing_is_consistent_while_another_thread_invalidates(registry: Registry, tool1: Action) -> None:
    """A cross-thread invalidate must never surface as a missing or half-read cache."""
    fetches = 0

    async def dap_fn() -> DapValue:
        nonlocal fetches
        fetches += 1
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn, cache_ttl_millis=60_000)

    stop = threading.Event()

    def invalidate_until_stopped() -> None:
        while not stop.is_set():
            dap.invalidate_cache()

    async def list_repeatedly() -> list[dict[str, ActionMetadata]]:
        return [await dap.list_action_metadata_by_key('my-dap') for _ in range(2000)]

    # The gap this guards against is a few bytecodes wide, so the default 5ms
    # switch interval never preempts inside it.
    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    invalidator = threading.Thread(target=invalidate_until_stopped, daemon=True)
    invalidator.start()
    try:
        listings = asyncio.run(list_repeatedly())
    finally:
        stop.set()
        invalidator.join(timeout=5)
        sys.setswitchinterval(switch_interval)

    assert fetches > 0
    for listing in listings:
        assert list(listing) == ['/dynamic-action-provider/my-dap:tool/tool1']


@pytest.mark.asyncio
async def test_listing_by_key_does_not_run_the_provider_action(registry: Registry, tool1: Action) -> None:
    """Dev UI polls must not emit a provider trace, unlike a resolve that runs one."""
    runs: list[DapMetadata] = []

    async def record_fn(input: DapMetadata) -> DapMetadata:
        runs.append(input)
        return input

    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)
    dap.action = Action(name='my-dap', kind=ActionKind.DYNAMIC_ACTION_PROVIDER, fn=record_fn)

    await dap.list_action_metadata_by_key('my-dap')
    assert runs == []

    dap.invalidate_cache()
    await dap.get_action('tool', 'tool1')
    assert len(runs) == 1


async def _warm_provider(registry: Registry, tool1: Action) -> DynamicActionProvider:
    """Build a provider with a long TTL and prime its cache."""

    async def dap_fn() -> DapValue:
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn, cache_ttl_millis=60_000)
    await dap.list_action_metadata_by_key('my-dap')
    return dap


@pytest.mark.asyncio
async def test_read_path_survives_an_invalidate_at_the_ttl_check(registry: Registry, tool1: Action) -> None:
    """The staleness check works off one read, so an invalidate before the unpack cannot be seen."""
    dap = await _warm_provider(registry, tool1)

    class InvalidatingTtl(int):
        """Stands in for another thread invalidating at the TTL comparison."""

        def __ge__(self, other: int) -> bool:
            dap.invalidate_cache()
            return True

    dap._ttl_millis = InvalidatingTtl(60_000)

    listing = await dap.list_action_metadata_by_key('my-dap')

    assert list(listing) == ['/dynamic-action-provider/my-dap:tool/tool1']


@pytest.mark.asyncio
async def test_read_path_survives_an_invalidate_at_the_clock_read(
    registry: Registry, tool1: Action, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expiry compare works off the same read, so an invalidate mid-compare cannot be seen."""
    dap = await _warm_provider(registry, tool1)
    real_time = time.time

    def invalidating_time() -> float:
        dap.invalidate_cache()
        return real_time()

    monkeypatch.setattr(_dap, 'time', SimpleNamespace(time=invalidating_time))

    listing = await dap.list_action_metadata_by_key('my-dap')

    assert list(listing) == ['/dynamic-action-provider/my-dap:tool/tool1']


def test_closed_loops_are_pruned_from_the_fetch_map(registry: Registry, tool1: Action) -> None:
    """A loop that ends mid-fetch leaves an entry behind, and the next fetch clears it."""
    started: list[asyncio.Event] = []
    stalled: list[asyncio.Task[dict[str, ActionMetadata]]] = []
    calls = 0

    async def dap_fn() -> DapValue:
        nonlocal calls
        calls += 1
        if calls == 1:
            started[0].set()
            await asyncio.Event().wait()
        return {'tool': [tool1]}

    dap = define_dynamic_action_provider(registry, 'my-dap', dap_fn)

    async def start_and_walk_away() -> None:
        started.append(asyncio.Event())
        stalled.append(asyncio.ensure_future(dap.list_action_metadata_by_key('my-dap')))
        await started[0].wait()

    abandoned = asyncio.new_event_loop()
    # The fetch is abandoned on purpose, so silence the destructor's report for it.
    abandoned.set_exception_handler(lambda loop, context: None)
    try:
        abandoned.run_until_complete(start_and_walk_away())
    finally:
        abandoned.close()

    assert abandoned in dap._fetch_tasks

    dap.invalidate_cache()
    survivor = asyncio.new_event_loop()
    try:
        survivor.run_until_complete(asyncio.wait_for(dap._get_or_fetch(skip_trace=True), timeout=5))
    finally:
        survivor.close()

    assert abandoned not in dap._fetch_tasks
    assert not stalled[0].done()
