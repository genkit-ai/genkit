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

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

import structlog

import ollama as ollama_api
from genkit import Constrained, ModelInfo, ModelRequest, ModelResponse, Supports
from genkit.embedder import (
    EmbedderOptions,
    EmbedderSupports,
    EmbedRequest,
    EmbedResponse,
    embedder_action_metadata,
)
from genkit.model import model_action_metadata
from genkit.plugin_api import (
    Action,
    ActionKind,
    ActionMetadata,
    ActionRunContext,
    Plugin,
    loop_local_client,
    to_json_schema,
)

from ._errors import wrap_connection_errors
from .constants import (
    DEFAULT_OLLAMA_SERVER_URL,
    OllamaAPITypes,
)
from .embedders import (
    EmbeddingDefinition,
    OllamaEmbedder,
)
from .models import (
    ModelDefinition,
    OllamaConfig,
    OllamaModel,
)

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


def ollama_model_info(model_ref: ModelDefinition, label: str) -> dict[str, object]:
    """Build first-party model metadata for an Ollama model."""
    supports = Supports(
        multiturn=model_ref.api_type == OllamaAPITypes.CHAT,
        media=model_ref.api_type == OllamaAPITypes.CHAT and model_ref.supports.media,
        tools=model_ref.api_type == OllamaAPITypes.CHAT and model_ref.supports.tools,
        system_role=True,
        output=['text', 'json'],
        constrained=Constrained.ALL,
    )
    return ModelInfo(label=label, supports=supports).model_dump(by_alias=True, exclude_none=True)


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
        request_headers: dict[str, str]
        | Callable[[], dict[str, str]]
        | Callable[[], Awaitable[dict[str, str]]]
        | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the Ollama plugin.

        Args:
            models: Optional list of model definitions to pre-register with
                Genkit.
            embedders: Optional list of embedding model definitions to
                pre-register.
            server_address: URL of the Ollama server. Defaults to
                ``http://127.0.0.1:11434``.
            request_headers: HTTP headers to include with every request.
                Accepts a static dict, a sync callable returning a dict, or
                an async callable returning a dict. Callables are resolved
                once during :meth:`init` — use this for OAuth tokens whose
                lifetime exceeds the plugin instance's lifetime.
            timeout: Request timeout in seconds applied to the underlying
                ``httpx`` client. ``None`` uses the ollama-python default.
        """
        self.models = models or []
        self.embedders = embedders or []
        self.server_address = server_address or DEFAULT_OLLAMA_SERVER_URL
        self._request_headers_source = request_headers
        # Static dicts are usable immediately; callables resolve during init().
        self.request_headers: dict[str, str] = {**request_headers} if isinstance(request_headers, dict) else {}
        self.timeout = timeout

        self.client = loop_local_client(self._make_client)

    def _make_client(self) -> ollama_api.AsyncClient:
        """Construct an ``AsyncClient`` with the resolved headers and timeout."""
        kwargs: dict[str, Any] = {'host': self.server_address, 'headers': self.request_headers}
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
        return ollama_api.AsyncClient(**kwargs)

    async def init(self) -> list:
        """Initialize the Ollama plugin.

        Resolves request_headers (which may be a callable or coroutine) and
        returns pre-registered models and embedders.

        Returns:
            List of Action objects for pre-configured models and embedders.
        """
        self.request_headers = await self._resolve_request_headers()

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

    async def _resolve_request_headers(self) -> dict[str, str]:
        """Resolve a static dict / sync callable / async callable into a dict."""
        source = self._request_headers_source
        if source is None:
            return {}
        if isinstance(source, dict):
            return {str(k): str(v) for k, v in source.items()}
        result = source()
        if inspect.isawaitable(result):
            result = await result
        return dict(cast(dict[str, str], result))

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

        action_metadata = model_action_metadata(
            name=name,
            config_schema=OllamaConfig,
            info=ollama_model_info(model_ref=model_ref, label=f'Ollama - {clean_name}'),
        )

        server_address = self.server_address

        async def _run(request: ModelRequest, ctx: ActionRunContext | None = None) -> ModelResponse:
            async with wrap_connection_errors(server_address):
                return await model.generate(request, ctx)

        action = Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_run,
            metadata=action_metadata.metadata,
        )
        action.input_schema = action_metadata.input_json_schema  # type: ignore[invalid-assignment]
        action.output_schema = action_metadata.output_json_schema  # type: ignore[invalid-assignment]
        return action

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

        server_address = self.server_address

        async def _run(request: EmbedRequest) -> EmbedResponse:
            async with wrap_connection_errors(server_address):
                return await embedder.embed(request)

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=_run,
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
        async with wrap_connection_errors(self.server_address):
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
                        config_schema=OllamaConfig,
                        info=ollama_model_info(
                            model_ref=ModelDefinition(name=name),
                            label=f'Ollama - {name}',
                        ),
                    )
                )
        return actions
