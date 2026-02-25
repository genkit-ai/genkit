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

"""Cohere AI plugin for Genkit.

This module provides the main plugin class that registers Cohere chat
models and embedders with the Genkit registry.

Example::

    from genkit.ai import Genkit
    from genkit.plugins.cohere import Cohere, cohere_name

    ai = Genkit(
        plugins=[Cohere()],
        model=cohere_name('command-a-03-2025'),
    )

See:
    - https://docs.cohere.com/docs/models
    - https://dashboard.cohere.com/api-keys
"""

from __future__ import annotations

import os

from genkit.ai import Plugin
from genkit.blocks.embedding import EmbedRequest, EmbedResponse
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.schema import to_json_schema
from genkit.plugins.cohere.embeddings import (
    SUPPORTED_EMBEDDING_MODELS,
    CohereEmbedder,
)
from genkit.plugins.cohere.model_info import SUPPORTED_COHERE_MODELS
from genkit.plugins.cohere.models import (
    COHERE_PLUGIN_NAME,
    CohereConfig,
    CohereModel,
    cohere_name,
)

# Models that are embedders, not chat/completion models.
_EMBEDDING_MODEL_NAMES = frozenset(SUPPORTED_EMBEDDING_MODELS.keys())

# Models removed by Cohere — reject with a clear message instead of a
# confusing 404 from the API.  See https://docs.cohere.com/docs/models
_REMOVED_MODEL_NAMES: frozenset[str] = frozenset({
    'command',
    'command-light',
    'command-nightly',
    'command-light-nightly',
})


class Cohere(Plugin):
    """Cohere AI plugin for Genkit.

    This plugin provides integration with Cohere's official Python SDK,
    enabling the use of Cohere chat models (Command family) **and**
    embedders (Embed family) within the Genkit framework.

    Example::

        from genkit.ai import Genkit
        from genkit.plugins.cohere import Cohere, cohere_name

        ai = Genkit(
            plugins=[Cohere()],
            model=cohere_name('command-a-03-2025'),
        )

    Args:
        api_key: The Cohere API key. If not provided, it attempts to load
            from the ``COHERE_API_KEY`` or ``CO_API_KEY`` environment variable.
        models: Optional list of model names to register. If not provided,
            all supported models and embedders are registered.

    Raises:
        GenkitError: If no API key is provided via parameter or environment.
    """

    name = 'cohere'

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
    ) -> None:
        """Initialize the plugin and set up its configuration.

        Args:
            api_key: The Cohere API key. If not provided, it attempts to load
                from the COHERE_API_KEY or CO_API_KEY environment variable.
            models: Optional list of specific model names to register.

        Raises:
            GenkitError: If no API key is provided via parameter or environment.
        """
        self._api_key = api_key or os.environ.get('COHERE_API_KEY') or os.environ.get('CO_API_KEY')
        if not self._api_key:
            raise GenkitError(
                message=(
                    'Cohere API key is required. Set the COHERE_API_KEY (or CO_API_KEY) '
                    'environment variable or pass api_key to the Cohere plugin.'
                ),
                status='INVALID_ARGUMENT',
            )
        self._models = models

    async def init(self) -> list:
        """Initialize the plugin.

        Returns:
            Empty list (using lazy loading via resolve).
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Routes to the correct factory based on action type and model name:
        - Embedding models are resolved only for ``ActionKind.EMBEDDER``.
        - Chat models are resolved only for ``ActionKind.MODEL``.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.

        Raises:
            GenkitError: If the model has been removed by Cohere.
        """
        # Strip the plugin prefix to get the clean model name.
        clean_name = name.removeprefix(f'{COHERE_PLUGIN_NAME}/')

        # Reject models that Cohere has retired.
        if clean_name in _REMOVED_MODEL_NAMES:
            raise GenkitError(
                message=(
                    f"Model '{clean_name}' was removed by Cohere on September 15, 2025. "
                    'Use command-a-03-2025 or another supported model instead. '
                    'See https://docs.cohere.com/docs/models'
                ),
                status='NOT_FOUND',
            )

        # Only resolve known models/embedders if a specific list was provided.
        if self._models is not None and clean_name not in self._models:
            return None

        if action_type == ActionKind.EMBEDDER and clean_name in _EMBEDDING_MODEL_NAMES:
            return self._create_embedder_action(name, clean_name)

        if action_type == ActionKind.MODEL and clean_name not in _EMBEDDING_MODEL_NAMES:
            return self._create_model_action(name)

        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Cohere model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        assert self._api_key is not None
        clean_name = name.removeprefix(f'{COHERE_PLUGIN_NAME}/')
        model = CohereModel(model=clean_name, api_key=self._api_key)

        model_info = model.get_model_info() or {}

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.to_generate_fn(),
            metadata={
                'model': {
                    **model_info,
                    'customOptions': to_json_schema(CohereConfig),
                },
            },
        )

    def _create_embedder_action(self, name: str, clean_name: str) -> Action:
        """Create an Action object for a Cohere embedder.

        Args:
            name: The namespaced name of the embedder.
            clean_name: The model name without the plugin prefix.

        Returns:
            Action object for the embedder.
        """
        assert self._api_key is not None
        embedder = CohereEmbedder(model=clean_name, api_key=self._api_key)
        model_meta = SUPPORTED_EMBEDDING_MODELS.get(clean_name, {})

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            return await embedder.embed(request)

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=embed_fn,
            metadata={
                'embedder': {
                    'info': {
                        'label': model_meta.get('label', f'Cohere - {clean_name}'),
                        'dimensions': model_meta.get('dimensions'),
                        'supports': {'input': ['text']},
                    },
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available Cohere models and embedders.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects for each
                supported Cohere model and embedder.
        """
        actions: list[ActionMetadata] = []

        # Determine which models to list.
        model_names = self._models if self._models else list(SUPPORTED_COHERE_MODELS.keys())
        embed_names = self._models if self._models else list(SUPPORTED_EMBEDDING_MODELS.keys())

        for name in model_names:
            if name in _EMBEDDING_MODEL_NAMES:
                continue
            info = SUPPORTED_COHERE_MODELS.get(name)
            supports_dict = info.supports.model_dump() if info and info.supports else {}
            actions.append(
                ActionMetadata(
                    kind=ActionKind.MODEL,
                    name=cohere_name(name),
                    metadata={
                        'model': {
                            'info': {
                                'name': info.label if info else name,
                                'supports': supports_dict,
                            },
                            'customOptions': to_json_schema(CohereConfig),
                        },
                    },
                )
            )

        for name in embed_names:
            if name not in _EMBEDDING_MODEL_NAMES:
                continue
            model_meta = SUPPORTED_EMBEDDING_MODELS.get(name, {})
            actions.append(
                ActionMetadata(
                    kind=ActionKind.EMBEDDER,
                    name=cohere_name(name),
                    metadata={
                        'embedder': {
                            'info': {
                                'label': model_meta.get('label', f'Cohere - {name}'),
                                'dimensions': model_meta.get('dimensions'),
                                'supports': {'input': ['text']},
                            },
                        },
                    },
                )
            )

        return actions
