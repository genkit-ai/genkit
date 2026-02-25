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

"""Dynamic Action Provider (DAP) support for Genkit.

Dynamic Action Providers allow external systems to supply actions at runtime,
enabling integration with dynamic tooling systems like MCP (Model Context
Protocol) servers, plugin marketplaces, or other dynamic action sources.

Overview
========

A Dynamic Action Provider is a registered action that acts as a factory for
other actions. When Genkit needs to resolve an action that isn't statically
registered, it queries relevant DAPs to see if they can provide it.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    Dynamic Action Provider Flow                          │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
    │   │   Request    │      │     DAP      │      │   External   │          │
    │   │   Action     │ ───► │    Cache     │ ───► │   System     │          │
    │   │  Resolution  │      │   (TTL)      │      │  (e.g. MCP)  │          │
    │   └──────────────┘      └──────────────┘      └──────────────┘          │
    │          │                    │                      │                   │
    │          │                    ▼                      ▼                   │
    │          │              ┌──────────────┐      ┌──────────────┐          │
    │          └───────────── │   Return     │ ◄─── │   Actions    │          │
    │                         │   Action     │      │   Created    │          │
    │                         └──────────────┘      └──────────────┘          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Key Concepts
============

┌─────────────────────────────────────────────────────────────────────────────┐
│ Concept              │ Description                                          │
├──────────────────────┼──────────────────────────────────────────────────────┤
│ DAP                  │ Dynamic Action Provider - a factory for actions      │
│ DapFn                │ Async function that returns actions by type          │
│ DapConfig            │ Configuration including name, description, caching   │
│ DapValue             │ Dictionary mapping action types to action lists      │
│ Cache TTL            │ Time-to-live for cached actions (default: 3000ms)    │
└─────────────────────┴──────────────────────────────────────────────────────┘

Use Cases
=========

1. **MCP Integration**: Connect to MCP servers that provide tools/resources
2. **Plugin Marketplaces**: Load tools from external plugin systems
3. **Multi-tenant Systems**: Provide tenant-specific actions dynamically
4. **Feature Flags**: Enable/disable actions based on runtime configuration

Example:
    ```python
    from genkit.ai import Genkit
    from genkit.blocks.dap import define_dynamic_action_provider

    ai = Genkit()


    # Define a DAP that provides tools from an external source
    async def get_mcp_tools():
        # Connect to MCP server and get available tools
        tools = await mcp_client.list_tools()
        return {
            'tool': [
                ai.dynamic_tool(
                    name=t.name,
                    description=t.description,
                    fn=lambda input: mcp_client.call_tool(t.name, input),
                )
                for t in tools
            ]
        }


    dap = define_dynamic_action_provider(
        ai.registry,
        config={'name': 'mcp-tools', 'cache_config': {'ttl_millis': 5000}},
        fn=get_mcp_tools,
    )
    ```

Caveats:
    - DAPs are queried during action resolution, so they should be fast
    - Cache TTL balances freshness with performance - choose wisely
    - Actions from DAPs have the DAP name prefixed for identification

See Also:
    - MCP Plugin: genkit.plugins.mcp for Model Context Protocol integration
    - JS Implementation: js/core/src/dynamic-action-provider.ts
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from genkit.core.action import Action
from genkit.core.action.types import ActionKind
from genkit.core.registry import Registry

ActionMetadataLike = Mapping[str, object]
"""Type alias for action metadata - any string-keyed mapping.

This type represents objects that behave like action metadata dictionaries,
providing key-based access to action properties like 'name', 'description',
'type', etc.

Examples of compatible types:
    - dict[str, object]
    - dict[str, Any]
    - ActionMetadata (from genkit.core.action)
    - Any other Mapping[str, object]
"""

DapValue = dict[str, list[Action[Any, Any]]]
"""Dictionary mapping action type names to lists of actions."""

DapFn = Callable[[], Awaitable[DapValue]]
"""Async function that returns actions organized by type."""

DapMetadata = dict[str, list[ActionMetadataLike]]
"""Dictionary mapping action type names to lists of action metadata."""


@dataclass
class DapCacheConfig:
    """Configuration for DAP caching behavior.

    Attributes:
        ttl_millis: Time-to-live for cache in milliseconds.
            - Negative: No caching (always fetch fresh)
            - Zero/None: Default (3000 milliseconds)
            - Positive: Cache validity duration in milliseconds
    """

    ttl_millis: int | None = None


@dataclass
class DapConfig:
    """Configuration for a Dynamic Action Provider.

    Attributes:
        name: Unique name for this DAP (used as prefix for actions).
        description: Human-readable description of what this DAP provides.
        cache_config: Optional caching configuration.
        metadata: Additional metadata to attach to the DAP action.
    """

    name: str
    description: str | None = None
    cache_config: DapCacheConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SimpleCache:
    """Thread-safe cache for DAP values with TTL expiration.

    This cache ensures that concurrent requests for the same data share
    a single fetch operation, preventing thundering herd problems.
    """

    def __init__(
        self,
        dap: 'DynamicActionProvider',
        config: DapConfig,
        dap_fn: DapFn,
    ) -> None:
        """Initialize the cache.

        Args:
            dap: The parent DAP action.
            config: DAP configuration including TTL.
            dap_fn: Function to fetch actions from the external source.
        """
        self._dap = dap
        self._dap_fn = dap_fn
        self._value: DapValue | None = None
        self._expires_at: float | None = None
        self._fetch_task: asyncio.Task[DapValue] | None = None

        # Determine TTL (default 3000ms)
        ttl = config.cache_config.ttl_millis if config.cache_config else None
        self._ttl_millis = 3000 if ttl is None or ttl == 0 else ttl

    async def get_or_fetch(self, skip_trace: bool = False) -> DapValue:
        """Get cached value or fetch fresh data if stale.

        This method handles concurrent requests by sharing a single
        fetch operation across all waiters.

        Args:
            skip_trace: If True, don't run the DAP action (used by DevUI
                to avoid excessive trace entries).

        Returns:
            The DAP value containing actions by type.
        """
        # Check if cache is still valid
        is_stale = (
            self._value is None
            or self._expires_at is None
            or self._ttl_millis < 0
            or time.time() * 1000 > self._expires_at
        )

        if not is_stale and self._value is not None:
            return self._value

        # If there's already a fetch in progress, wait for it
        if self._fetch_task is not None:
            return await self._fetch_task

        # Start a new fetch
        self._fetch_task = asyncio.create_task(self._do_fetch(skip_trace))
        try:
            return await self._fetch_task
        finally:
            self._fetch_task = None

    async def _do_fetch(self, skip_trace: bool) -> DapValue:
        """Perform the actual fetch operation.

        Args:
            skip_trace: If True, skip running the DAP action.

        Returns:
            Fresh DAP value.
        """
        try:
            self._value = await self._dap_fn()
            self._expires_at = time.time() * 1000 + self._ttl_millis

            # Run the DAP action for tracing (unless skipped)
            if not skip_trace:
                metadata = transform_dap_value(self._value)
                await self._dap.action.arun(metadata)

            return self._value
        except Exception:
            self.invalidate()
            raise

    def invalidate(self) -> None:
        """Invalidate the cache, forcing a fresh fetch on next access."""
        self._value = None
        self._expires_at = None


def transform_dap_value(value: DapValue) -> DapMetadata:
    """Transform DAP value to metadata format for logging.

    Args:
        value: DAP value with actions.

    Returns:
        DAP metadata with action metadata.
    """
    metadata: DapMetadata = {}
    for action_type, actions in value.items():
        action_metadata_list: list[ActionMetadataLike] = []
        for action in actions:
            # Action.metadata is dict[str, object] which satisfies ActionMetadataLike
            meta: ActionMetadataLike = action.metadata if action.metadata else {}
            action_metadata_list.append(meta)
        metadata[action_type] = action_metadata_list
    return metadata


class DynamicActionProvider:
    """A Dynamic Action Provider that lazily resolves actions.

    This class wraps a DAP function and provides methods to query
    for actions by type and name, with caching for performance.
    """

    def __init__(
        self,
        action: Action[Any, Any],
        config: DapConfig,
        dap_fn: DapFn,
    ) -> None:
        """Initialize the DAP.

        Args:
            action: The underlying DAP action.
            config: DAP configuration.
            dap_fn: Function to fetch actions.
        """
        self.action = action
        self.config = config
        self._cache = SimpleCache(self, config, dap_fn)

    def invalidate_cache(self) -> None:
        """Invalidate the cache, forcing a fresh fetch on next access."""
        self._cache.invalidate()

    async def get_action(
        self,
        action_type: str,
        action_name: str,
    ) -> Action[Any, Any] | None:
        """Get a specific action by type and name.

        Args:
            action_type: The type of action (e.g., 'tool', 'model').
            action_name: The name of the action.

        Returns:
            The action if found, None otherwise.
        """
        result = await self._cache.get_or_fetch()
        actions = result.get(action_type, [])
        for action in actions:
            if action.name == action_name:
                return action
        return None

    async def list_action_metadata(
        self,
        action_type: str,
        action_name: str,
    ) -> list[ActionMetadataLike]:
        """List metadata for actions matching type and name pattern.

        Args:
            action_type: The type of action.
            action_name: Name or pattern to match:
                - '*': Match all actions of this type
                - 'prefix*': Match actions starting with prefix
                - 'exact': Match only this exact name

        Returns:
            List of matching action metadata.
        """
        result = await self._cache.get_or_fetch()
        actions = result.get(action_type, [])
        if not actions:
            return []

        metadata_list: list[ActionMetadataLike] = []
        for action in actions:
            meta: ActionMetadataLike = action.metadata if action.metadata else {}
            metadata_list.append(meta)

        # Match all
        if action_name == '*':
            return metadata_list

        # Prefix match
        if action_name.endswith('*'):
            prefix = action_name[:-1]
            return [m for m in metadata_list if str(m.get('name', '')).startswith(prefix)]

        # Exact match
        return [m for m in metadata_list if m.get('name') == action_name]

    async def get_action_metadata_record(
        self,
        dap_prefix: str,
    ) -> dict[str, ActionMetadataLike]:
        """Get all actions as a metadata record for reflection API.

        This is used by the DevUI to list available actions.

        Args:
            dap_prefix: Prefix to add to action keys.

        Returns:
            Dictionary mapping action keys to metadata.
        """
        dap_actions: dict[str, ActionMetadataLike] = {}

        # Skip trace to avoid excessive DevUI trace entries
        result = await self._cache.get_or_fetch(skip_trace=True)

        for action_type, actions in result.items():
            for action in actions:
                if not action.name:
                    raise ValueError(f'Invalid metadata when listing dynamic actions from {dap_prefix} - name required')
                key = f'{dap_prefix}:{action_type}/{action.name}'
                dap_actions[key] = action.metadata if action.metadata else {}

        return dap_actions


def is_dynamic_action_provider(obj: object) -> bool:
    """Check if an object is a Dynamic Action Provider.

    Args:
        obj: Object to check.

    Returns:
        True if the object is a DAP.
    """
    if isinstance(obj, DynamicActionProvider):
        return True
    if hasattr(obj, 'metadata'):
        metadata = getattr(obj, 'metadata', None)
        if isinstance(metadata, dict):
            return metadata.get('type') == 'dynamic-action-provider'
    return False


def define_dynamic_action_provider(
    registry: Registry,
    config: DapConfig | str,
    fn: DapFn,
) -> DynamicActionProvider:
    """Define and register a Dynamic Action Provider.

    A DAP is a factory that can dynamically provide actions at runtime.
    This is useful for integrating with external systems like MCP servers
    or plugin marketplaces.

    Args:
        registry: The registry to register the DAP with.
        config: DAP configuration or just a name string.
        fn: Async function that returns actions organized by type.

    Returns:
        The registered DynamicActionProvider.

    Example:
        ```python
        # Simple DAP that provides tools
        async def get_tools():
            return {
                'tool': [
                    ai.dynamic_tool(name='tool1', fn=...),
                    ai.dynamic_tool(name='tool2', fn=...),
                ]
            }


        dap = define_dynamic_action_provider(
            registry,
            config='my-tools',
            fn=get_tools,
        )

        # DAP with custom caching
        dap = define_dynamic_action_provider(
            registry,
            config=DapConfig(
                name='mcp-tools',
                description='Tools from MCP server',
                cache_config=DapCacheConfig(ttl_millis=10000),
            ),
            fn=get_tools,
        )
        ```

    See Also:
        - JS implementation: js/core/src/dynamic-action-provider.ts
    """
    # Normalize config
    cfg = DapConfig(name=config) if isinstance(config, str) else config

    # Create metadata with DAP type marker
    action_metadata = {
        **cfg.metadata,
        'type': 'dynamic-action-provider',
    }

    # Define the underlying action
    # The action itself just returns its input (for logging purposes)
    async def dap_action(input: DapMetadata) -> DapMetadata:
        return input

    action = registry.register_action(
        name=cfg.name,
        kind=ActionKind.DYNAMIC_ACTION_PROVIDER,
        description=cfg.description,
        fn=dap_action,
        metadata=action_metadata,
    )

    # Wrap in DynamicActionProvider
    dap = DynamicActionProvider(action, cfg, fn)

    return dap


__all__ = [
    'ActionMetadataLike',
    'DapCacheConfig',
    'DapConfig',
    'DapFn',
    'DapMetadata',
    'DapValue',
    'DynamicActionProvider',
    'SimpleCache',
    'define_dynamic_action_provider',
    'is_dynamic_action_provider',
    'transform_dap_value',
]
