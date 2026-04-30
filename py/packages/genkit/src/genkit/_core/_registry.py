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

"""Registry for managing Genkit resources and actions."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from dotpromptz.dotprompt import Dotprompt
from pydantic import BaseModel
from typing_extensions import Never, TypeVar

from genkit._core._action import (
    GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR,
    Action,
    ActionKind,
    ActionName,
    ActionRunContext,
    SpanAttributeValue,
    create_action_key,
    parse_action_key,
    parse_dap_qualified_name,
)
from genkit._core._error import GenkitError
from genkit._core._logger import get_logger
from genkit._core._model import (
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
)
from genkit._core._plugin import Plugin
from genkit._core._typing import (
    ActionMetadata,
    EmbedRequest,
    EmbedResponse,
    EvalRequest,
    EvalResponse,
)

logger = get_logger(__name__)

# An action store is a nested dictionary mapping ActionKind to a dictionary of
# action names and their corresponding Action instances.
#
# Structure for illustration:
#
# ```python
# {
#     ActionKind.MODEL: {
#         'gemini-2.0-flash': Action(...),
#         'gemini-2.0-pro': Action(...)
#     },
# }
# ```
ActionStore = dict[ActionKind, dict[ActionName, Action]]

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT', default=Never)

ActionFn = (
    Callable[[], OutputT | Awaitable[OutputT]]
    | Callable[[InputT], OutputT | Awaitable[OutputT]]
    | Callable[[InputT, ActionRunContext], OutputT | Awaitable[OutputT]]
)


def _action_metadata_for_registered_action(action: Action) -> ActionMetadata:
    """Build an ``ActionMetadata`` row for a directly-registered :class:`Action`."""
    return ActionMetadata(
        key=create_action_key(action.kind, action.name),
        action_type=action.kind,
        name=action.name,
        description=action.description,
        input_schema=action.input_schema,
        output_schema=action.output_schema,
        metadata=dict(action.metadata) if action.metadata else None,
    )


class Registry:
    """Central repository for Genkit resources.

    The Registry class serves as the central storage and management system for
    various Genkit resources including actions, trace stores, flow state stores,
    plugins, and schemas. It provides methods for registering new resources and
    looking them up at runtime.

    Supports a **child registry** pattern (see ``new_child``): a child registry
    delegates lookups to its parent when a name is not found locally.  This is
    used to create cheap, ephemeral registries scoped to a single generate call
    (for DAP-resolved tools) without polluting the root registry.

    This class is thread-safe and can be safely shared between multiple threads.

    Attributes:
        entries: A nested dictionary mapping ActionKind to a dictionary of
            action names and their corresponding Action instances.
    """

    def __init__(self, parent: Registry | None = None) -> None:
        """Initialize a Registry instance.

        Args:
            parent: Optional parent registry.  When provided this is a *child*
                registry that falls back to the parent for any lookup that
                returns ``None`` locally.  Use ``new_child()`` as the
                preferred factory rather than passing ``parent`` directly.
        """
        self._parent: Registry | None = parent
        self._entries: ActionStore = {}
        self._value_by_kind_and_name: dict[str, dict[str, object]] = {}
        self._schemas_by_name: dict[str, dict[str, object]] = {}
        self._schema_types_by_name: dict[str, type[BaseModel]] = {}
        self._lock: threading.RLock = threading.RLock()

        # Re-entrancy guard for _trigger_lazy_loading.  Prevents infinite
        # recursion when a lazy factory resolves its own action key (see
        # https://github.com/genkit-ai/genkit/issues/4491).
        self._loading_actions: set[str] = set()

        # Dotprompt resolves ``output.schema`` names via the registry's stored schemas.
        # Async resolver avoids thread-pool deadlock in ``resolve_json_schema``.
        async def async_schema_resolver(name: str) -> dict[str, object]:
            schema = self.lookup_schema(name)
            if schema is None:
                raise GenkitError(status='NOT_FOUND', message=f"Schema '{name}' not found")
            return schema

        # Children share the parent's Dotprompt instance (prompts are global).
        self._dotprompt: Dotprompt = (
            parent.dotprompt if parent is not None else Dotprompt(schema_resolver=async_schema_resolver)
        )
        # TODO(#4352): Figure out how to set this.
        self.api_stability: str = 'stable'

        # Plugin infrastructure
        #
        # Notes on concurrency:
        # - Registry state is protected by the thread lock (`_lock`) because the dev
        #   reflection server runs in a separate OS thread and inspects the same
        #   registry instance.
        # - Plugin initialization is lazy and "init-once" via an in-flight task
        #   cache (`_plugin_init_tasks`). This assumes a single asyncio event loop
        #   drives plugin initialization for a given registry instance. (The dev
        #   reflection server schedules coroutines onto that loop.)
        self._plugins: dict[str, Plugin] = {}
        self._plugin_init_tasks: dict[str, asyncio.Task[None]] = {}
        self._all_plugins_initialized: bool = False

    # -------------------------------------------------------------------------
    # Child registry support
    # -------------------------------------------------------------------------

    def new_child(self) -> Registry:
        """Create a cheap child registry that inherits from this registry.

        Child registries are used to create short-lived, ephemeral scopes (e.g.
        per-generate-call tool registrations from a DAP) without polluting the
        root registry.  Any lookup that fails locally falls back to this parent.
        Writes on the child never propagate back to the parent.

        Returns:
            A new ``Registry`` whose parent is ``self``.
        """
        return Registry(parent=self)

    @property
    def parent(self) -> Registry | None:
        """The parent registry, or ``None`` if this is a root registry."""
        return self._parent

    @property
    def is_child(self) -> bool:
        """``True`` if this registry has a parent."""
        return self._parent is not None

    @property
    def dotprompt(self) -> Dotprompt:
        """The shared :class:`Dotprompt` instance for this registry tree.

        Mutations (partials, helpers) propagate to all sibling and descendant
        registries because the instance is shared. Use :func:`define_partial`
        and :func:`define_helper` rather than mutating the returned instance
        directly, so the public surface stays stable.
        """
        return self._dotprompt

    def register_action(
        self,
        kind: ActionKind,
        name: str,
        fn: ActionFn[InputT, OutputT],
        metadata_fn: Callable[..., object] | None = None,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
        span_metadata: dict[str, SpanAttributeValue] | None = None,
    ) -> Action[InputT, OutputT, ChunkT]:
        """Register a new action with the registry.

        This method creates a new Action instance with the provided parameters
        and registers it in the registry under the specified kind and name.

        Args:
            kind: The type of action being registered (e.g., TOOL, MODEL).
            name: A unique name for the action within its kind.
            fn: The function to be called when the action is executed.
            metadata_fn: The function to be used to infer metadata (e.g.
                schemas).
            description: Optional human-readable description of the action.
            metadata: Optional dictionary of metadata about the action.
            span_metadata: Optional dictionary of tracing span metadata.

        Returns:
            The newly created and registered Action instance.
        """
        action = Action(
            kind=kind,
            name=name,
            fn=cast(Callable[..., Awaitable[OutputT]], fn),
            metadata_fn=metadata_fn,
            description=description,
            metadata=metadata,
            span_metadata=span_metadata,
        )
        action_typed = cast(Action[InputT, OutputT, ChunkT], action)
        with self._lock:
            if kind not in self._entries:
                self._entries[kind] = {}
            self._entries[kind][name] = action
        return action_typed

    def register_action_from_instance(self, action: Action) -> None:
        """Register an existing Action instance.

        Allows registering a pre-configured Action object, such as one created via
        `dynamic_resource` or other factory methods.

        Args:
           action: The action instance to register.
        """
        with self._lock:
            if action.kind not in self._entries:
                self._entries[action.kind] = {}
            self._entries[action.kind][action.name] = action

    async def resolve_actions_by_kind(self, kind: ActionKind) -> dict[str, Action]:
        """Returns all registered actions for a specific kind, triggering lazy loading.

        File-based prompts defer schema resolution until first access. This method
        ensures all action metadata is fully loaded before returning.

        Args:
            kind: The type of actions to retrieve (e.g., TOOL, MODEL, RESOURCE).

        Returns:
            A dictionary mapping action names to Action instances with fully loaded metadata.
        """
        with self._lock:
            actions = self._entries.get(kind, {}).copy()
        for action in actions.values():
            await self._trigger_lazy_loading(action)
        return actions

    async def list_actions(self) -> dict[str, ActionMetadata]:
        """Return reflection metadata for plugins, registered actions, and DAP-expanded tools.

        Initializes plugins, advertises plugin rows from each plugin's ``list_actions()``,
        then fills registered :class:`Action` rows and expands DAP-provided actions. Merges
        with the parent registry's catalog; entries from this registry win on duplicate keys.

        Returns:
            Map of action key string to typed :class:`ActionMetadata`.
        """
        await self.initialize_all_plugins()

        catalog: dict[str, ActionMetadata] = {}

        # 1. Plugin-advertised rows: actions the plugin claims it can resolve on demand.
        with self._lock:
            plugins = list(self._plugins.items())
        for plugin_name, plugin in plugins:
            try:
                advertised = await plugin.list_actions()
            except Exception:
                logger.exception('Error listing actions for plugin %s', plugin_name)
                continue
            for meta in advertised or []:
                if not meta.name:
                    raise ValueError(f'Invalid ActionMetadata from {plugin_name}: name required')
                if not meta.action_type:
                    raise ValueError(f'Invalid ActionMetadata from {plugin_name}: action_type required')
                key = f'/{meta.action_type}/{meta.name}'
                catalog[key] = meta.model_copy(update={'key': key})

        # 2. Concrete registered actions, plus DAP-expanded actions if the action is a provider.
        for kind in ActionKind.__members__.values():
            for name, action in (await self.resolve_actions_by_kind(kind)).items():
                key = create_action_key(kind, name)
                catalog[key] = _action_metadata_for_registered_action(action)

                dap = getattr(action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
                if dap is None:
                    continue
                try:
                    # DAP action keys are prefixed with the provider action's ``name``;
                    # see :meth:`DynamicActionProvider.list_action_metadata_by_key`.
                    dap_actions = await dap.list_action_metadata_by_key(action.name)
                except Exception:
                    logger.exception(
                        'Error listing actions for Dynamic Action Provider %s',
                        action.name,
                    )
                    continue
                # ``list_action_metadata_by_key`` already populates each entry's ``meta.key``
                # to match its DAP action key, so we can merge straight into the catalog.
                catalog.update(dap_actions)

        # 3. Merge in parent registry's catalog; entries from this registry win on duplicate keys.
        if self._parent is None:
            return catalog
        parent_catalog = await self._parent.list_actions()
        return {**parent_catalog, **catalog}

    def register_value(self, kind: str, name: str, value: object) -> None:
        """Registers a value with a given kind and name.

        This method stores a value in a nested dictionary, where the first level
        is keyed by the 'kind' and the second level is keyed by the 'name'.
        It prevents duplicate registrations for the same kind and name.

        Args:
            kind: The kind of the value (e.g., "format", "default-model").
            name: The name of the value (e.g., "json", "text").
            value: The value to be registered. Can be of any non-serializable
                type.

        Raises:
            ValueError: If a value with the given kind and name is already
            registered.
        """
        with self._lock:
            if kind not in self._value_by_kind_and_name:
                self._value_by_kind_and_name[kind] = {}

            if name in self._value_by_kind_and_name[kind]:
                raise ValueError(f'value for kind "{kind}" and name "{name}" is already registered')

            self._value_by_kind_and_name[kind][name] = value

    def lookup_value(self, kind: str, name: str) -> object | None:
        """Looks up value that us previously registered by `register_value`.

        Args:
            kind: The kind of the value (e.g., "format", "default-model").
            name: The name of the value (e.g., "json", "text").

        Returns:
            The value or None if not found.  Falls back to parent registry.
        """
        with self._lock:
            local = self._value_by_kind_and_name.get(kind, {}).get(name)
        if local is not None:
            return local
        return self._parent.lookup_value(kind, name) if self._parent is not None else None

    def list_values(self, kind: str) -> dict[str, object]:
        """List all values registered for a specific kind, merged with the parent registry.

        Entries from this registry win on duplicate names.

        Args:
            kind: The kind of values to list (e.g., ``"defaultModel"``, ``"format"``).

        Returns:
            Map of value name to value object.
        """
        with self._lock:
            local = dict(self._value_by_kind_and_name.get(kind, {}))
        if self._parent is None:
            return local
        return {**self._parent.list_values(kind), **local}

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin with the registry.

        Args:
            plugin: The plugin to register.

        Raises:
            ValueError: If a plugin with the same name is already registered.
        """
        # Guard plugin registry mutations: in dev mode the reflection server may
        # list actions while the main thread is still registering plugins.
        with self._lock:
            if plugin.name in self._plugins:
                raise ValueError(f'Plugin {plugin.name} already registered')
            self._plugins[plugin.name] = plugin
            self._all_plugins_initialized = False

    async def initialize_all_plugins(self) -> None:
        """Run ``init()`` for every plugin on this registry exactly once (until a new plugin is registered).

        Used before enumerating registered actions so plugin-registered entries exist in ``_entries``.
        """
        if self._all_plugins_initialized:
            return
        with self._lock:
            plugin_names = list(self._plugins.keys())
        for name in plugin_names:
            await self._ensure_plugin_initialized(name)
        self._all_plugins_initialized = True

    async def _ensure_plugin_initialized(self, plugin_name: str) -> None:
        """Ensure a plugin is initialized exactly once.

        This method implements lazy, once-only initialization using an in-flight
        task pattern. Multiple concurrent calls will await the same task.

        Args:
            plugin_name: The name of the plugin to initialize.

        Raises:
            KeyError: If the plugin is not registered.
        """
        # IMPORTANT: Do not hold `_lock` across any `await`. The critical section
        # below is sync-only (dict access + task creation), so it is safe to use
        # `_lock` to make the init-once behavior atomic across threads/tasks.
        with self._lock:
            task = self._plugin_init_tasks.get(plugin_name)
            if task is None:
                plugin = self._plugins.get(plugin_name)
                if plugin is None:
                    raise KeyError(f'Plugin not registered: {plugin_name}')

                async def run_init() -> None:
                    # Assert for type narrowing inside closure (pyrefly doesn't propagate from outer scope)
                    assert plugin is not None
                    actions = await plugin.init()
                    for action in actions or []:
                        self.register_action_instance(action, namespace=plugin_name)

                task = asyncio.create_task(run_init())
                self._plugin_init_tasks[plugin_name] = task

        await task

    def register_action_instance(self, action: Action, *, namespace: str | None = None) -> None:
        """Register an existing Action instance with optional namespace normalization.

        If a namespace is provided, the action name will be normalized to ensure
        it has the correct plugin prefix.

        Args:
            action: The action instance to register.
            namespace: Optional plugin namespace to prefix the action name.
        """
        name = action.name
        if namespace:
            if '/' in name:
                # Name already has a namespace, replace it
                _, local = name.split('/', 1)
                name = f'{namespace}/{local}'
            else:
                # Name is local, prefix with namespace
                name = f'{namespace}/{name}'
            # Update the action's name
            action._name = name  # pyright: ignore[reportPrivateUsage]

        with self._lock:
            if action.kind not in self._entries:
                self._entries[action.kind] = {}
            self._entries[action.kind][name] = action

    async def _trigger_lazy_loading(self, action: Action | None) -> Action | None:
        """Trigger lazy loading for an action if needed.

        File-based prompts are registered with deferred metadata (schemas). This method
        triggers the async factory to resolve that metadata before returning the action.
        The factory is memoized, so subsequent calls return immediately.

        A re-entrancy guard (``_loading_actions``) prevents infinite recursion
        when a factory resolves its own action key during initialization.
        See https://github.com/genkit-ai/genkit/issues/4491.
        """
        if action is None:
            return None
        async_factory = getattr(action, '_async_factory', None)
        if async_factory is not None and action.metadata.get('lazy'):
            action_id = f'{action.kind}/{action.name}'
            if action_id in self._loading_actions:
                return action
            self._loading_actions.add(action_id)
            try:
                await async_factory()
            except Exception as e:
                logger.warning(f'Failed to load lazy action {action.name}: {e}')
            finally:
                self._loading_actions.discard(action_id)
        return action

    async def _resolve_dap_qualified_action(self, kind: ActionKind, name: str) -> Action | None:
        """Resolve through the one registered DAP for ``provider:innerKind/innerName`` names.

        Caller must ensure :func:`parse_dap_qualified_name` accepts ``name``. Does not consult
        plugins. Returns ``None`` if the provider is not registered here (caller may delegate
        to a parent registry).
        """
        qualified = parse_dap_qualified_name(name)
        if qualified is None:
            return None
        dap_host = qualified.provider
        with self._lock:
            provider = self._entries.get(ActionKind.DYNAMIC_ACTION_PROVIDER, {}).get(dap_host)
        if provider is None:
            return None
        dap_action = await self._trigger_lazy_loading(provider)
        if dap_action is None:
            raise RuntimeError(
                f'Dynamic action provider {dap_host!r} is not registered. '
                'DAPs must be registered using define_dynamic_action_provider '
                'before referencing qualified action names.'
            )
        dap = getattr(dap_action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
        if dap is not None:
            try:
                resolved = await dap.get_action(qualified.inner_kind, qualified.inner_name)
            except Exception as e:
                raise ValueError(f'Dynamic action provider {dap_host!r} get_action failed for {kind} {name!r}') from e
            if resolved is not None and resolved.kind == kind:
                return resolved
            if resolved is None:
                raise ValueError(
                    f'Dynamic action provider {dap_host!r} has no action '
                    f'{qualified.inner_kind!r}/{qualified.inner_name!r} for {name!r}'
                )
            raise ValueError(
                f'Dynamic action provider {dap_host!r} returned {resolved.kind!r} for {name!r}, expected {kind!r}'
            )
        raise RuntimeError(
            f'Dynamic action provider {dap_host!r} is missing the Genkit DAP helper. '
            'Register it using define_dynamic_action_provider before referencing qualified action names.'
        )

    async def resolve_action(self, kind: ActionKind, name: str) -> Action | None:
        """Resolve an action by kind and name.

        Tries an exact (kind, name) cache hit first. DAP-qualified names
        (provider:innerKind/innerName) go through that provider only. If the name contains a
        slash, the first segment is treated as a plugin id: that plugin is initialized and
        plugin.resolve is used. Falls back to parent registry if nothing found.

        Args:
            kind: The type of action to resolve.
            name: Action name, optionally plugin/... for a specific plugin.

        Returns:
            The Action instance if found, None otherwise.
        """
        with self._lock:
            if kind in self._entries and name in self._entries[kind]:
                return await self._trigger_lazy_loading(self._entries[kind][name])

        # DAP-qualified names: resolve via that provider only (not plugin slash splitting).
        if kind != ActionKind.DYNAMIC_ACTION_PROVIDER and parse_dap_qualified_name(name) is not None:
            action = await self._resolve_dap_qualified_action(kind, name)
            if action is not None:
                return action
            if self._parent is not None:
                return await self._parent.resolve_action(kind, name)
            return None

        # <plugin name>/<action name>: resolve that plugin.
        if '/' in name:
            plugin_name, action_name = name.split('/', 1)
            with self._lock:
                plugin = self._plugins.get(plugin_name)

            if plugin is not None:
                await self._ensure_plugin_initialized(plugin_name)

                target = f'{plugin_name}/{action_name}'  # normalized

                # Check cache again after init - init() might have registered this action
                with self._lock:
                    if kind in self._entries and target in self._entries[kind]:
                        return await self._trigger_lazy_loading(self._entries[kind][target])

                # On-demand resolution: target may not have been in init()'s registered set.
                action = await plugin.resolve(kind, target)
                if action is not None:
                    self.register_action_instance(action, namespace=plugin_name)
                    with self._lock:
                        return await self._trigger_lazy_loading(self._entries.get(kind, {}).get(target))

                # Check cache first - init() might have registered this action
                with self._lock:
                    cached_action = self._entries.get(kind, {}).get(target)
                if cached_action is not None:
                    successes.append((plugin_name, cached_action))
                    continue

                action = await plugin.resolve(kind, target)
                if action is not None:
                    successes.append((plugin_name, action))

            if len(successes) > 1:
                plugin_names = [p for p, _ in successes]
                raise ValueError(
                    f"Ambiguous {kind} action name '{name}'. "
                    + f"Matches plugins: {plugin_names}. Use 'plugin/{name}'."
                )

            if len(successes) == 1:
                plugin_name, action = successes[0]
                self.register_action_instance(action, namespace=plugin_name)
                with self._lock:
                    return await self._trigger_lazy_loading(self._entries.get(kind, {}).get(f'{plugin_name}/{name}'))

        # Fallback: try dynamic action providers (for MCP, dynamic resources, etc.)
        # Skip if we're looking up a dynamic action provider itself to avoid recursion
        if kind != ActionKind.DYNAMIC_ACTION_PROVIDER:
            with self._lock:
                if ActionKind.DYNAMIC_ACTION_PROVIDER in self._entries:
                    providers_dict = self._entries[ActionKind.DYNAMIC_ACTION_PROVIDER]
                else:
                    providers_dict = {}
                providers = list(providers_dict.values())
            for provider_action in providers:
                dap = getattr(provider_action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
                if dap is None:
                    continue
                try:
                    resolved = await dap.get_action(str(kind), name)
                    if resolved is not None:
                        return resolved
                except Exception as e:
                    logger.debug(
                        f'Dynamic action provider {provider_action.name} failed for {kind}/{name}',
                        exc_info=e,
                    )

        # Final fallback: delegate to parent registry.
        if self._parent is not None:
            return await self._parent.resolve_action(kind, name)

        return None

    async def resolve_action_by_key(self, key: str) -> Action | None:
        """Resolve an action using its combined key string.

        The key format is ``/<kind>/<name>``, where kind must be a valid
        ``ActionKind`` and name may be prefixed with plugin namespace or
        unprefixed.

        For nested actions exposed by a dynamic action provider, use
        ``/dynamic-action-provider/<provider>:<innerKind>/<innerName>`` (for
        example ``/dynamic-action-provider/my-mcp:tool/echo``).

        Args:
            key: The action key in the format ``/<kind>/<name>``.

        Returns:
            The ``Action`` instance if found, None otherwise.

        Raises:
            ValueError: If the key format is invalid, the kind is not a valid
                ``ActionKind``, or an unprefixed name is ambiguous.
        """
        kind, name = parse_action_key(key)
        if kind == ActionKind.DYNAMIC_ACTION_PROVIDER:
            dap_parts = parse_dap_qualified_name(name)
            if dap_parts is not None:
                provider_name, inner_kind_str, inner_name = dap_parts
                provider_action = await self.resolve_action(
                    ActionKind.DYNAMIC_ACTION_PROVIDER,
                    provider_name,
                )
                if provider_action is None:
                    return None
                dap = getattr(provider_action, GENKIT_DYNAMIC_ACTION_PROVIDER_ATTR, None)
                if dap is None:
                    return None
                try:
                    resolved = await dap.get_action(inner_kind_str, inner_name)
                except Exception as e:
                    logger.debug(
                        f'Dynamic action provider {provider_name} failed for '
                        f'qualified key {inner_kind_str}/{inner_name}',
                        exc_info=e,
                    )
                    return None
                if resolved is None:
                    return None
                return resolved
        return await self.resolve_action(kind, name)

    def register_schema(self, name: str, schema: dict[str, object], schema_type: type[BaseModel] | None = None) -> None:
        """Registers a schema by name.

        Schemas registered with this method can be referenced by name in
        .prompt files using the `output.schema` field.

        Args:
            name: The name of the schema.
            schema: The schema data (JSON schema format).
            schema_type: Optional Pydantic model class for runtime validation.

        Raises:
            ValueError: If a schema with the given name is already registered.
        """
        with self._lock:
            if name in self._schemas_by_name:
                raise ValueError(f'Schema "{name}" is already registered')
            self._schemas_by_name[name] = schema
            if schema_type is not None:
                self._schema_types_by_name[name] = schema_type
            logger.debug(f'Registered schema "{name}"')

    def lookup_schema(self, name: str) -> dict[str, object] | None:
        """Looks up a schema by name.

        Args:
            name: The name of the schema to look up.

        Returns:
            The schema data if found, None otherwise.  Falls back to parent.
        """
        with self._lock:
            local = self._schemas_by_name.get(name)
        if local is not None:
            return local
        return self._parent.lookup_schema(name) if self._parent is not None else None

    def lookup_schema_type(self, name: str) -> type[BaseModel] | None:
        """Looks up a schema's Pydantic type by name.

        Args:
            name: The name of the schema to look up.

        Returns:
            The Pydantic model class if found, None otherwise.  Falls back to parent.
        """
        with self._lock:
            local = self._schema_types_by_name.get(name)
        if local is not None:
            return local
        return self._parent.lookup_schema_type(name) if self._parent is not None else None

    # ===== Typed Action Lookups =====
    #
    # These methods provide type-safe access to actions of specific kinds.
    # They wrap resolve_action() with appropriate casts to preserve generic
    # type parameters that would otherwise be erased.

    async def resolve_embedder(self, name: str) -> Action[EmbedRequest, EmbedResponse, Never] | None:
        """Resolve an embedder action by name with full type information.

        Args:
            name: The embedder name (e.g., "my-embedder" or "plugin/embedder").

        Returns:
            A fully typed embedder action, or None if not found.
        """
        action = await self.resolve_action(ActionKind.EMBEDDER, name)
        if action is None:
            return None
        return cast(Action[EmbedRequest, EmbedResponse, Never], action)

    async def resolve_model(self, name: str) -> Action[ModelRequest, ModelResponse, ModelResponseChunk] | None:
        """Resolve a model action by name with full type information.

        Args:
            name: The model name (e.g., "gemini-pro" or "plugin/model").

        Returns:
            A fully typed model action, or None if not found.
        """
        action = await self.resolve_action(ActionKind.MODEL, name)
        if action is None:
            return None
        return cast(
            Action[ModelRequest, ModelResponse, ModelResponseChunk],
            action,
        )

    async def resolve_evaluator(self, name: str) -> Action[EvalRequest, EvalResponse, Never] | None:
        """Resolve an evaluator action by name with full type information.

        Args:
            name: The evaluator name (e.g., "my-evaluator" or "plugin/evaluator").

        Returns:
            A fully typed evaluator action, or None if not found.
        """
        action = await self.resolve_action(ActionKind.EVALUATOR, name)
        if action is None:
            return None
        return cast(Action[EvalRequest, EvalResponse, Never], action)
