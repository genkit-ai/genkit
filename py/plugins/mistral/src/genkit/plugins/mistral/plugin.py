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

"""Mistral AI Plugin for Genkit.

Registers both chat/completion models and the ``mistral-embed`` embedder
with the Genkit framework.
"""

import os
from typing import Any

from mistralai import Mistral as MistralClient

from genkit.ai import Plugin
from genkit.blocks.embedding import (
    EmbedderOptions,
    EmbedderSupports,
    EmbedRequest,
    EmbedResponse,
    embedder_action_metadata,
)
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.schema import to_json_schema
from genkit.plugins.mistral.embeddings import (
    SUPPORTED_EMBEDDING_MODELS,
    MistralEmbedder,
)
from genkit.plugins.mistral.model_info import SUPPORTED_MISTRAL_MODELS
from genkit.plugins.mistral.models import (
    MISTRAL_PLUGIN_NAME,
    MistralConfig,
    MistralModel,
    mistral_name,
)

# Models that are embedders, not chat/completion models.
_EMBEDDING_MODEL_NAMES = frozenset(SUPPORTED_EMBEDDING_MODELS.keys())


class Mistral(Plugin):
    """Mistral AI plugin for Genkit.

    This plugin provides integration with Mistral AI's official Python SDK,
    enabling the use of Mistral chat models **and** embedders within the
    Genkit framework.

    Example::

        from genkit import Genkit
        from genkit.plugins.mistral import Mistral

        ai = Genkit(
            plugins=[Mistral()],
            model='mistral/mistral-large-latest',
        )

        # Chat completion
        response = await ai.generate(prompt='Hello!')

        # Embeddings
        embeddings = await ai.embed(
            embedder='mistral/mistral-embed',
            content=['Hello world'],
        )
    """

    name = MISTRAL_PLUGIN_NAME

    def __init__(
        self,
        api_key: str | None = None,
        models: list[str] | None = None,
        **mistral_params: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the plugin and set up its configuration.

        Args:
            api_key: The Mistral API key. If not provided, it attempts to load
                from the MISTRAL_API_KEY environment variable.
            models: An optional list of model names to register with the plugin.
                If None, all supported models will be registered.
            **mistral_params: Additional parameters for the Mistral client.

        Raises:
            GenkitError: If no API key is provided via parameter or environment.
        """
        self.api_key = api_key if api_key is not None else os.getenv('MISTRAL_API_KEY')

        if not self.api_key:
            raise GenkitError(message='Please provide api_key or set MISTRAL_API_KEY environment variable.')

        self.models = models
        self.mistral_params = mistral_params

        # Shared client for all embedder instances (created lazily).
        self._client: MistralClient | None = None

    def _get_client(self) -> MistralClient:
        """Return a shared Mistral client, creating it on first access."""
        if self._client is None:
            self._client = MistralClient(api_key=str(self.api_key), **self.mistral_params)
        return self._client

    async def init(self) -> list[Action]:
        """Initialize the plugin.

        Returns:
            Empty list (using lazy loading via resolve).
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Routes to the correct factory based on action type and model name:
        - Embedding models are resolved only for ``ActionKind.EMBEDDER``.
        - Chat/completion models are resolved only for ``ActionKind.MODEL``.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        clean_name = name.replace(MISTRAL_PLUGIN_NAME + '/', '') if name.startswith(MISTRAL_PLUGIN_NAME) else name

        if action_type == ActionKind.EMBEDDER:
            if clean_name not in _EMBEDDING_MODEL_NAMES:
                return None
            return self._create_embedder_action(name, clean_name)

        if action_type == ActionKind.MODEL:
            # Embedding models should not be resolved as chat models.
            if clean_name in _EMBEDDING_MODEL_NAMES:
                return None
            return self._create_model_action(name)

        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Mistral model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(MISTRAL_PLUGIN_NAME + '/', '') if name.startswith(MISTRAL_PLUGIN_NAME) else name

        # Create the Mistral model instance
        mistral_model = MistralModel(
            model=clean_name,
            api_key=str(self.api_key),
            **self.mistral_params,
        )

        model_info = mistral_model.get_model_info() or {}
        generate_fn = mistral_model.to_generate_fn()

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=generate_fn,
            metadata={
                'model': {
                    **model_info,
                    'customOptions': to_json_schema(MistralConfig),
                },
            },
        )

    def _create_embedder_action(self, name: str, clean_name: str) -> Action:
        """Create an Action object for a Mistral embedder.

        Args:
            name: The namespaced name of the embedder.
            clean_name: The model name without the plugin prefix.

        Returns:
            Action object for the embedder.
        """
        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(
            clean_name,
            {
                'label': f'Mistral AI Embedding - {clean_name}',
                'dimensions': 1024,
                'supports': {'input': ['text']},
            },
        )
        embedder = MistralEmbedder(model=clean_name, client=self._get_client())

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            return await embedder.embed(request)

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=embed_fn,
            metadata=embedder_action_metadata(
                name=name,
                options=EmbedderOptions(
                    label=embedder_info['label'],
                    supports=EmbedderSupports(input=embedder_info['supports']['input']),
                    dimensions=embedder_info.get('dimensions'),
                ),
            ).metadata,
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available Mistral models and embedders.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects for each
                supported Mistral model and embedder.
        """
        actions_list: list[ActionMetadata] = []

        for model, model_info in SUPPORTED_MISTRAL_MODELS.items():
            # Skip embedding models from the model list — they're listed separately.
            if model in _EMBEDDING_MODEL_NAMES:
                continue
            actions_list.append(
                model_action_metadata(
                    name=mistral_name(model), info=model_info.model_dump(), config_schema=MistralConfig
                )
            )

        for embed_model, embed_info in SUPPORTED_EMBEDDING_MODELS.items():
            actions_list.append(
                embedder_action_metadata(
                    name=mistral_name(embed_model),
                    options=EmbedderOptions(
                        label=embed_info['label'],
                        supports=EmbedderSupports(input=embed_info['supports']['input']),
                        dimensions=embed_info.get('dimensions'),
                    ),
                )
            )

        return actions_list
