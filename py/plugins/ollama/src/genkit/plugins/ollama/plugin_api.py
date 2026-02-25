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

"""Ollama Plugin for Genkit."""

from functools import partial

import structlog

import ollama as ollama_api
from genkit.ai import Plugin
from genkit.blocks.embedding import EmbedderOptions, EmbedderSupports, embedder_action_metadata
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.registry import ActionKind
from genkit.core.schema import to_json_schema
from genkit.plugins.ollama.constants import (
    DEFAULT_OLLAMA_SERVER_URL,
    OllamaAPITypes,
)
from genkit.plugins.ollama.embedders import (
    EmbeddingDefinition,
    OllamaEmbedder,
)
from genkit.plugins.ollama.models import (
    ModelDefinition,
    OllamaModel,
)
from genkit.types import GenerationCommonConfig

OLLAMA_PLUGIN_NAME = 'ollama'
logger = structlog.get_logger(__name__)


def ollama_name(name: str) -> str:
    """Get the name of the Ollama model.

    Args:
        name: The name of the Ollama model.

    Returns:
        The name of the Ollama model.
    """
    return f'{OLLAMA_PLUGIN_NAME}/{name}'


class Ollama(Plugin):
    """Ollama plugin for Genkit.

    This plugin integrates Ollama models and embedding capabilities into Genkit
    for local or custom server-based generative AI applications.
    """

    name = OLLAMA_PLUGIN_NAME

    def __init__(
        self,
        models: list[ModelDefinition] | None = None,
        embedders: list[EmbeddingDefinition] | None = None,
        server_address: str | None = None,
        request_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Ollama plugin.

        Args:
            models: An Optional list of model definitions to be registered with Genkit.
            embedders: An Optional list of embedding model definitions to be
                registered with Genkit.
            server_address: The URL of the Ollama server. Defaults to a predefined
                Ollama server URL if not provided.
            request_headers: Optional HTTP headers to include with requests to the
                Ollama server.
        """
        self.models = models or []
        self.embedders = embedders or []
        self.server_address = server_address or DEFAULT_OLLAMA_SERVER_URL
        self.request_headers = request_headers or {}

        self.client = partial(ollama_api.AsyncClient, host=self.server_address)

    async def init(self) -> list:
        """Initialize the Ollama plugin.

        Returns pre-registered models and embedders.

        Returns:
            List of Action objects for pre-configured models and embedders.
        """
        actions = []

        # Register pre-configured models
        for model_def in self.models:
            name = ollama_name(model_def.name)
            action = self._create_model_action(name)
            actions.append(action)

        # Register pre-configured embedders
        for embedder_def in self.embedders:
            name = ollama_name(embedder_def.name)
            action = self._create_embedder_action(name)
            actions.append(action)

        return actions

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Args:
            action_type: The kind of action to resolve.
            name: The namespaced name of the action to resolve.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.MODEL:
            return self._create_model_action(name)
        elif action_type == ActionKind.EMBEDDER:
            return self._create_embedder_action(name)
        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for an Ollama model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(OLLAMA_PLUGIN_NAME + '/', '') if name.startswith(OLLAMA_PLUGIN_NAME) else name

        # Try to find the model definition from pre-configured models
        model_ref = None
        for model_def in self.models:
            if model_def.name == clean_name:
                model_ref = model_def
                break

        # If not found in pre-configured models, create a default one
        if model_ref is None:
            model_ref = ModelDefinition(name=clean_name)

        model = OllamaModel(
            client=self.client,
            model_definition=model_ref,
        )

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=model.generate,
            metadata={
                'model': {
                    'label': f'Ollama - {clean_name}',
                    'multiturn': model_ref.api_type == OllamaAPITypes.CHAT,
                    'system_role': True,
                    'tools': model_ref.supports.tools,
                    'output': ['text', 'json'],
                    'constrained': 'all',
                    'customOptions': to_json_schema(GenerationCommonConfig),
                },
            },
        )

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for an Ollama embedder.

        Args:
            name: The namespaced name of the embedder.

        Returns:
            Action object for the embedder.
        """
        # Extract local name (remove plugin prefix)
        clean_name = name.replace(OLLAMA_PLUGIN_NAME + '/', '') if name.startswith(OLLAMA_PLUGIN_NAME) else name

        embedder_ref = EmbeddingDefinition(name=clean_name)
        embedder = OllamaEmbedder(
            client=self.client,
            embedding_definition=embedder_ref,
        )

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=embedder.embed,
            metadata={
                'embedder': {
                    'label': f'Ollama Embedding - {clean_name}',
                    'dimensions': embedder_ref.dimensions,
                    'supports': {'input': ['text']},
                    'customOptions': to_json_schema(ollama_api.Options),
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """Generate a list of available actions or models.

        Returns:
            list[ActionMetadata]: A list of ActionMetadata objects, each with the following attributes:
                - name (str): The name of the action or model.
                - kind (ActionKind): The type or category of the action.
                - info (dict): The metadata dictionary describing the model configuration and properties.
                - config_schema (type): The schema class used for validating the model's configuration.
        """
        client = self.client()
        response = await client.list()

        actions = []
        for model in response.models:
            name = model.model
            if not name:
                continue
            if 'embed' in name:
                actions.append(
                    embedder_action_metadata(
                        name=ollama_name(name),
                        options=EmbedderOptions(
                            config_schema=to_json_schema(ollama_api.Options),
                            label=f'Ollama Embedding - {name}',
                            supports=EmbedderSupports(input=['text']),
                        ),
                    )
                )
            else:
                actions.append(
                    model_action_metadata(
                        name=ollama_name(name),
                        config_schema=GenerationCommonConfig,
                        info={
                            'label': f'Ollama - {name}',
                            'multiturn': True,
                            'system_role': True,
                            'tools': True,
                            'output': ['text', 'json'],
                            'constrained': 'all',
                        },
                    )
                )
        return actions
