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

"""Registry for managing Genkit resources and actions.

This module provides the Registry class, which is the central repository for
storing and managing various Genkit resources such as actions, flows,
plugins, and schemas. The registry enables dynamic registration and lookup
of these resources during runtime.

Example:
    >>> registry = Registry()
    >>> registry.register_action('<action kind>', 'my_action', ...)
    >>> action = await registry.resolve_action('<action kind>', 'my_action')
"""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import cast

from dotpromptz.dotprompt import Dotprompt
from pydantic import BaseModel
from typing_extensions import Never, TypeVar

from genkit.core.action import (
    Action,
    ActionMetadata,
    ActionRunContext,
    SpanAttributeValue,
    parse_action_key,
)
from genkit.core.action.types import ActionKind, ActionName
from genkit.core.logging import get_logger
from genkit.core.plugin import Plugin
from genkit.core.typing import (
    EmbedRequest,
    EmbedResponse,
    EvalRequest,
    EvalResponse,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    RerankerRequest,
    RerankerResponse,
    RetrieverRequest,
    RetrieverResponse,
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


class Registry:
    """Central repository for Genkit resources.

    The Registry class serves as the central storage and management system for
    various Genkit resources including actions, trace stores, flow state stores,
    plugins, and schemas. It provides methods for registering new resources and
    looking them up at runtime.

    This class is thread-safe and can be safely shared between multiple threads.

    Attributes:
        entries: A nested dictionary mapping ActionKind to a dictionary of
            action names and their corresponding Action instances.
    """

    default_model: str | None = None

    def __init__(self) -> None:
        """Initialize an empty Registry instance."""
        self._entries: ActionStore = {}
        self._value_by_kind_and_name: dict[str, dict[str, object]] = {}
        self._schemas_by_name: dict[str, dict[str, object]] = {}
        self._schema_types_by_name: dict[str, type[BaseModel]] = {}
        self._lock: threading.RLock = threading.RLock()

        # Re-entrancy guard for _trigger_lazy_loading.  Prevents infinite
        # recursion when a lazy factory resolves its own action key (see
        # https://github.com/firebase/genkit/issues/4491).
        self._loading_actions: set[str] = set()

        # Initialize Dotprompt with schema_resolver to match JS SDK pattern
        # Use async function to avoid thread pool deadlock in resolve_json_schema
        async def async_schema_resolver(name: str) -> dict[str, object] | str:
            return self.lookup_schema(name) or name

        self.dotprompt: Dotprompt = Dotprompt(schema_resolver=async_schema_resolver)
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
            The value or None if not found.
        """
        with self._lock:
            return self._value_by_kind_and_name.get(kind, {}).get(name)

    def list_values(self, kind: str) -> list[str]:
        """List all values registered for a specific kind.

        Args:
            kind: The kind of values to list (e.g., "defaultModel").

        Returns:
            A list of registered value names.
        """
        with self._lock:
            return list(self._value_by_kind_and_name.get(kind, {}).keys())

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
        See https://github.com/firebase/genkit/issues/4491.
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

    async def resolve_action(self, kind: ActionKind, name: str) -> Action | None:
        """Resolve an action by kind and name, supporting both prefixed and unprefixed names.

        This method supports:
        1. Cache hit: Returns immediately if action is already registered
        2. Namespaced request (e.g., "plugin/model"): Resolves via specific plugin
        3. Unprefixed request (e.g., "model"): Tries all plugins, errors on ambiguity
        4. Dynamic action providers: Last-resort fallback for dynamic action creation

        Args:
            kind: The type of action to resolve.
            name: The name of the action (may be prefixed with "plugin/" or unprefixed).

        Returns:
            The Action instance if found, None otherwise.

        Raises:
            ValueError: If an unprefixed name matches multiple plugins (ambiguous).
        """
        # Cache hit
        with self._lock:
            if kind in self._entries and name in self._entries[kind]:
                return await self._trigger_lazy_loading(self._entries[kind][name])

        action: Action | None = None

        # Namespaced request
        if '/' in name:
            plugin_name, local = name.split('/', 1)
            with self._lock:
                plugin = self._plugins.get(plugin_name)

            if plugin is not None:
                await self._ensure_plugin_initialized(plugin_name)

                target = f'{plugin_name}/{local}'  # normalized

                # Check cache again after init - init() might have registered this action
                with self._lock:
                    if kind in self._entries and target in self._entries[kind]:
                        return await self._trigger_lazy_loading(self._entries[kind][target])

                action = await plugin.resolve(kind, target)
                if action is not None:
                    self.register_action_instance(action, namespace=plugin_name)
                    with self._lock:
                        return await self._trigger_lazy_loading(self._entries.get(kind, {}).get(target))
        else:
            # Unprefixed request: try all plugins
            successes: list[tuple[str, Action]] = []
            with self._lock:
                plugins = list(self._plugins.items())
            for plugin_name, plugin in plugins:
                await self._ensure_plugin_initialized(plugin_name)
                target = f'{plugin_name}/{name}'

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
                    f"Ambiguous {kind.value} action name '{name}'. "
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
            for provider in providers:
                try:
                    response = await provider.arun({'kind': kind, 'name': name})
                    if response.response:
                        self.register_action_instance(response.response)
                        return await self._trigger_lazy_loading(response.response)
                except Exception as e:
                    logger.debug(
                        f'Dynamic action provider {provider.name} failed for {kind}/{name}',
                        exc_info=e,
                    )
                    continue

        return None

    async def resolve_action_by_key(self, key: str) -> Action | None:
        """Resolve an action using its combined key string.

        The key format is `<kind>/<name>`, where kind must be a valid
        `ActionKind` and name may be prefixed with plugin namespace or unprefixed.

        Args:
            key: The action key in the format `<kind>/<name>`.

        Returns:
            The `Action` instance if found, None otherwise.

        Raises:
            ValueError: If the key format is invalid, the kind is not a valid
                `ActionKind`, or an unprefixed name is ambiguous.
        """
        kind, name = parse_action_key(key)
        return await self.resolve_action(kind, name)

    async def list_actions(self, allowed_kinds: list[ActionKind] | None = None) -> list[ActionMetadata]:
        """List all actions advertised by plugins.

        This method returns the advertised set of actions from all registered
        plugins. It does NOT trigger plugin initialization and does NOT consult
        the registry's internal action store.

        Args:
            allowed_kinds: Optional list of action kinds to filter by.

        Returns:
            A list of ActionMetadata objects describing available actions.

        Raises:
            ValueError: If a plugin returns invalid ActionMetadata.
        """
        metas: list[ActionMetadata] = []
        with self._lock:
            plugins = list(self._plugins.items())
        for plugin_name, plugin in plugins:
            plugin_metas = await plugin.list_actions()
            for meta in plugin_metas or []:
                if not meta.name:
                    raise ValueError(f'Invalid ActionMetadata from {plugin_name}: name required')

                # Normalize metadata name
                if '/' not in meta.name:
                    meta = meta.model_copy(update={'name': f'{plugin_name}/{meta.name}'})

                if allowed_kinds and meta.kind not in allowed_kinds:
                    continue
                metas.append(meta)
        return metas

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
            The schema data if found, None otherwise.
        """
        with self._lock:
            return self._schemas_by_name.get(name)

    def lookup_schema_type(self, name: str) -> type[BaseModel] | None:
        """Looks up a schema's Pydantic type by name.

        Args:
            name: The name of the schema to look up.

        Returns:
            The Pydantic model class if found, None otherwise.
        """
        with self._lock:
            return self._schema_types_by_name.get(name)

    # ===== Typed Action Lookups =====
    #
    # These methods provide type-safe access to actions of specific kinds.
    # They wrap resolve_action() with appropriate casts to preserve generic
    # type parameters that would otherwise be erased.

    async def resolve_retriever(self, name: str) -> Action[RetrieverRequest, RetrieverResponse, Never] | None:
        """Resolve a retriever action by name with full type information.

        Args:
            name: The retriever name (e.g., "my-retriever" or "plugin/retriever").

        Returns:
            A fully typed retriever action, or None if not found.
        """
        action = await self.resolve_action(ActionKind.RETRIEVER, name)
        if action is None:
            return None
        return cast(Action[RetrieverRequest, RetrieverResponse, Never], action)

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

    async def resolve_reranker(self, name: str) -> Action[RerankerRequest, RerankerResponse, Never] | None:
        """Resolve a reranker action by name with full type information.

        Args:
            name: The reranker name (e.g., "my-reranker" or "plugin/reranker").

        Returns:
            A fully typed reranker action, or None if not found.
        """
        action = await self.resolve_action(ActionKind.RERANKER, name)
        if action is None:
            return None
        return cast(Action[RerankerRequest, RerankerResponse, Never], action)

    async def resolve_model(self, name: str) -> Action[GenerateRequest, GenerateResponse, GenerateResponseChunk] | None:
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
            Action[GenerateRequest, GenerateResponse, GenerateResponseChunk],
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
