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

"""Abstract base class for Genkit plugins.

This module defines the base plugin interface that all plugins must implement.
Plugins extend Genkit by registering models, embedders, retrievers, and
other action types.

Overview:
    Plugins are the primary extension mechanism in Genkit. They allow adding
    support for AI providers (Google AI, Anthropic, OpenAI), vector stores,
    and other capabilities. Each plugin implements a standard interface for
    initialization, action resolution, and action discovery.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      Plugin Lifecycle                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌──────────┐      ┌──────────┐      ┌──────────┐      ┌──────────┐    │
    │  │ Register │ ───► │   init   │ ───► │ resolve  │ ───► │  Action  │    │
    │  │  Plugin  │      │ (async)  │      │  (lazy)  │      │ Returns  │    │
    │  └──────────┘      └──────────┘      └──────────┘      └──────────┘    │
    │       │                 │                                              │
    │       │                 ▼                                              │
    │       │          ┌──────────┐                                          │
    │       └─────────►│  List    │                                          │
    │                  │ Actions  │                                          │
    │                  └──────────┘                                          │
    └─────────────────────────────────────────────────────────────────────────┘

Terminology:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Term              │ Description                                         │
    ├───────────────────┼─────────────────────────────────────────────────────┤
    │ Plugin            │ Abstract base class that defines the plugin API     │
    │ name              │ Plugin namespace (e.g., 'googleai', 'anthropic')    │
    │ init()            │ Async method called once on first action resolve    │
    │ resolve()         │ Lazy resolution of actions by kind and name         │
    │ list_actions()    │ Returns metadata for plugin's available actions     │
    │ model()           │ Helper to create namespaced ModelReference          │
    │ embedder()        │ Helper to create namespaced EmbedderRef             │
    └───────────────────┴─────────────────────────────────────────────────────┘

Key Methods:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Method            │ Purpose                                             │
    ├───────────────────┼─────────────────────────────────────────────────────┤
    │ init()            │ One-time initialization; return pre-registered      │
    │                   │ actions (e.g., known models to pre-register)        │
    │ resolve()         │ Create/return Action for a given kind and name;     │
    │                   │ called when action is first used                    │
    │ list_actions()    │ Return ActionMetadata for dev UI action discovery   │
    │                   │ (should be fast, no heavy initialization)           │
    └───────────────────┴─────────────────────────────────────────────────────┘

Example:
    Implementing a custom plugin:

    ```python
    from genkit.core.plugin import Plugin
    from genkit.core.action import Action, ActionMetadata
    from genkit.core.action.types import ActionKind


    class MyPlugin(Plugin):
        name = 'myplugin'

        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        async def init(self) -> list[Action]:
            # Return actions to pre-register (optional)
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            if action_type == ActionKind.MODEL:
                model_name = name.replace(f'{self.name}/', '')
                return self._create_model_action(model_name)
            return None

        async def list_actions(self) -> list[ActionMetadata]:
            # Return metadata for available models (for dev UI)
            return [
                ActionMetadata(kind=ActionKind.MODEL, name='myplugin/my-model'),
            ]

        def _create_model_action(self, name: str) -> Action:
            # Create and return the model action
            ...
    ```

    Using the plugin:

    ```python
    from genkit import Genkit

    ai = Genkit(plugins=[MyPlugin(api_key='...')])
    response = await ai.generate(model='myplugin/my-model', prompt='Hello!')
    ```

Caveats:
    - init() is called lazily on first action resolution, not at registration
    - Plugin names must be unique within a Genkit instance
    - Actions returned from init() are pre-registered with the plugin namespace
    - resolve() receives the fully namespaced name (e.g., 'plugin/model')

See Also:
    - Built-in plugins: genkit.plugins.google_genai, genkit.plugins.anthropic
    - Registry: genkit.core.registry
"""

import abc
from typing import TYPE_CHECKING

from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind

if TYPE_CHECKING:
    from genkit.blocks.embedding import EmbedderRef
    from genkit.blocks.model import ModelReference


class Plugin(abc.ABC):
    """Abstract base class for implementing Genkit plugins.

    This class defines the async plugin interface that all plugins must implement.
    Plugins provide a way to extend functionality by registering new actions, models,
    or other capabilities.
    """

    name: str  # plugin namespace

    @abc.abstractmethod
    async def init(self) -> list[Action]:
        """Lazy warm-up called once per plugin per registry instance.

        This method is called lazily when the first action resolution attempt
        involving this plugin occurs. It should return a list of Action objects
        to pre-register.

        Returns:
            list[Action]: A list of Action instances to register.
        """
        ...

    @abc.abstractmethod
    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve a single action.

        The registry will call this with a namespaced name (e.g., "plugin/model-name").

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action | None: The Action instance if found, None otherwise.
        """
        ...

    @abc.abstractmethod
    async def list_actions(self) -> list[ActionMetadata]:
        """Advertised set for dev UI/reflection listing endpoint.

        This method should be safe and ideally inexpensive. It returns the set
        of actions that this plugin advertises without triggering full initialization.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects describing
                available actions.
        """
        ...

    def model(self, name: str) -> 'ModelReference':
        """Creates a model reference.

        Prefixes local name with plugin namespace.

        Args:
            name: The model name (local or namespaced).

        Returns:
            ModelReference: A reference to the model.
        """
        # Deferred import: avoid circular import with genkit.blocks.model
        from genkit.blocks.model import ModelReference  # noqa: PLC0415

        target = name if '/' in name else f'{self.name}/{name}'
        return ModelReference(name=target)

    def embedder(self, name: str) -> 'EmbedderRef':
        """Creates an embedder reference.

        Prefixes local name with plugin namespace.

        Args:
            name: The embedder name (local or namespaced).

        Returns:
            EmbedderRef: A reference to the embedder.
        """
        # Deferred import: avoid circular import with genkit.blocks.embedding
        from genkit.blocks.embedding import EmbedderRef  # noqa: PLC0415

        target = name if '/' in name else f'{self.name}/{name}'
        return EmbedderRef(name=target)
