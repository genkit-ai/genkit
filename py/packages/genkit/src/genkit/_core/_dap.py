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

"""Dynamic Action Provider (DAP) support for Genkit."""

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from genkit._core._action import (
    Action,
    ActionKind,
    create_action_key,
)
from genkit._core._typing import ActionMetadata

ActionMetadataLike = Mapping[str, object]
DapValue = dict[str, list[Action[Any, Any]]]
DapFn = Callable[[], Awaitable[DapValue]]
DapMetadata = dict[str, list[ActionMetadataLike]]

# Default cache TTL in milliseconds
_DEFAULT_CACHE_TTL_MS = 3000


@dataclass
class _Fetch:
    """A loop-local fetch belonging to one cache invalidation generation."""

    task: asyncio.Task[DapValue]
    generation: int


class DynamicActionProvider:
    """Lazily resolves actions from an external source with TTL caching.

    The cached actions are shared by every event loop that lists this provider,
    so an action returned by ``dap_fn`` must resolve any loop-bound resource of
    its own when it is called, not when it is listed. In-flight fetches are
    coalesced per loop, because a task cannot be awaited from a loop other than
    the one that created it.

    The cache is one attribute holding both the value and its expiry, so a
    reader takes a consistent pair in a single read and an invalidation from
    another thread cannot land between the two.
    """

    def __init__(
        self,
        action: Action[Any, Any],
        dap_fn: DapFn,
        cache_ttl_millis: int | None = None,
    ) -> None:
        self.action = action
        self._dap_fn = dap_fn
        self._cache: tuple[DapValue, float] | None = None
        self._fetch_tasks: dict[asyncio.AbstractEventLoop, _Fetch] = {}
        self._generation = 0
        self._fetch_tasks_lock = threading.Lock()
        self._ttl_millis = (
            _DEFAULT_CACHE_TTL_MS if cache_ttl_millis is None or cache_ttl_millis == 0 else cache_ttl_millis
        )

    def invalidate_cache(self) -> None:
        """Drop the cached actions so the next call starts a fresh fetch.

        Existing callers can finish their in-flight fetch, but its result will
        not refill the cache after this invalidation.
        """
        with self._fetch_tasks_lock:
            self._generation += 1
            self._cache = None

    async def _get_or_fetch(self, skip_trace: bool = False) -> DapValue:
        """Get cached value or fetch fresh data, coalescing concurrent fetches per loop."""
        cached = self._cache
        if cached is not None and self._ttl_millis >= 0:
            value, expires_at = cached
            if time.time() * 1000 <= expires_at:
                return value

        loop = asyncio.get_running_loop()
        with self._fetch_tasks_lock:
            # A pending task strongly references its loop, so weak keys never fire.
            for ended in [known for known in self._fetch_tasks if known.is_closed()]:
                del self._fetch_tasks[ended]
            fetch = self._fetch_tasks.get(loop)
            if fetch is None or fetch.generation != self._generation or fetch.task.done():
                task = asyncio.create_task(self._do_fetch(skip_trace, self._generation, self._cache))
                fetch = _Fetch(task, self._generation)
                self._fetch_tasks[loop] = fetch
                task.add_done_callback(self._forget_fetch(loop))
            task = fetch.task

        # Shielded, so a caller that is cancelled cannot cancel the fetch every
        # other caller on this loop is waiting on.
        return await asyncio.shield(task)

    def _forget_fetch(self, loop: asyncio.AbstractEventLoop) -> Callable[[asyncio.Task[DapValue]], None]:
        """Build the callback that drops a finished fetch from the per-loop table.

        Cleanup belongs to the task rather than to whichever caller started it,
        because a shielded caller can be cancelled while the fetch it started
        keeps running, and dropping the entry then would uncoalesce it.
        """

        def forget(task: asyncio.Task[DapValue]) -> None:
            with self._fetch_tasks_lock:
                fetch = self._fetch_tasks.get(loop)
                if fetch is not None and fetch.task is task:
                    del self._fetch_tasks[loop]
            # Cancelling the last shielded caller unhooks shield's own retrieval,
            # so the fetch must take its outcome or asyncio reports it unretrieved.
            if not task.cancelled():
                task.exception()

        return forget

    async def _do_fetch(
        self, skip_trace: bool, generation: int, owned_cache: tuple[DapValue, float] | None
    ) -> DapValue:
        try:
            value = await self._dap_fn()
            with self._fetch_tasks_lock:
                # Existing callers may still use this result, but invalidation
                # prevents it from being published to subsequent callers.
                if generation == self._generation:
                    owned_cache = (value, time.time() * 1000 + self._ttl_millis)
                    self._cache = owned_cache
            if not skip_trace:
                metadata = {k: [a.metadata or {} for a in v] for k, v in value.items()}
                await self.action.run(metadata)
            return value
        except Exception:
            with self._fetch_tasks_lock:
                # A failed fetch on another loop must not evict a newer entry.
                if generation == self._generation and self._cache is owned_cache:
                    self._cache = None
            raise

    async def get_action(self, action_type: str, action_name: str) -> Action[Any, Any] | None:
        result = await self._get_or_fetch()
        for action in result.get(action_type, []):
            if action.name == action_name:
                return action
        return None

    async def list_action_metadata(self, action_type: str, action_name: str) -> list[ActionMetadataLike]:
        """List metadata matching pattern: '*'=all, 'prefix*'=prefix match, else exact."""
        result = await self._get_or_fetch()
        actions: list[Action[Any, Any]] = []
        seen: set[str] = set()
        for action in result.get(action_type, []):
            if action.name in seen:
                continue
            seen.add(action.name)
            actions.append(action)
        if not actions:
            return []

        metadata_list: list[ActionMetadataLike] = [action.metadata or {} for action in actions]

        if action_name == '*':
            return metadata_list
        if action_name.endswith('*'):
            prefix = action_name[:-1]
            return [m for m in metadata_list if str(m.get('name', '')).startswith(prefix)]
        return [m for m in metadata_list if m.get('name') == action_name]

    async def list_action_metadata_by_key(self, dap_prefix: str) -> dict[str, ActionMetadata]:
        """List every child action's reflection metadata, keyed by its fully-qualified DAP key."""
        result = await self._get_or_fetch(skip_trace=True)
        dap_actions: dict[str, ActionMetadata] = {}
        for action_type, actions in result.items():
            for action in actions:
                if not action.name:
                    raise ValueError(f'Invalid metadata from {dap_prefix} - name required')
                listed_type = action_type
                key = create_action_key(
                    ActionKind.DYNAMIC_ACTION_PROVIDER,
                    f'{dap_prefix}:{listed_type}/{action.name}',
                )
                dap_actions[key] = ActionMetadata(
                    key=key,
                    action_type=listed_type,
                    name=action.name,
                    description=action.description,
                    input_schema=action.input_schema,
                    output_schema=action.output_schema,
                    metadata=dict(action.metadata) if action.metadata else None,
                )
        return dap_actions


def is_dynamic_action_provider(obj: object) -> bool:
    if isinstance(obj, DynamicActionProvider):
        return True
    metadata = getattr(obj, 'metadata', None)
    return isinstance(metadata, dict) and metadata.get('type') == 'dynamic-action-provider'
