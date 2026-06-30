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
from genkit.plugins.ollama._errors import wrap_connection_errors
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
    OllamaConfig,
    OllamaModel,
    OllamaSupports,
)

OLLAMA_PLUGIN_NAME = 'ollama'
logger = structlog.get_logger(__name__)

# Models that are dynamically discovered (``list_actions``) or resolved on
# demand can't be capability-probed, so we advertise the full generic
# capability set. This mirrors the JS plugin's ``GENERIC_MODEL_INFO`` and the
# Go plugin's ``defaultOllamaSupports``, which both enable every capability for
# models that weren't explicitly pre-configured.
_DYNAMIC_MODEL_SUPPORTS = OllamaSupports(tools=True, media=True)


def ollama_name(name: str) -> str:
    """Get the name of the Ollama model.

    Args:
        name: The name of the Ollama model.

    Returns:
        The name of the Ollama model.
    """
    return f'{OLLAMA_PLUGIN_NAME}/{name}'


def ollama_model_info(model_ref: ModelDefinition, label: str) -> dict[str, object]:
    """Build Dev UI capability metadata for an Ollama model.

    Capabilities are gated on the model's API type so the Dev UI advertises
    only what the endpoint actually supports: the ``chat`` endpoint is
    multiturn and can use tools/media, whereas the ``generate`` endpoint is
    single-turn text-in/text-out.

    Args:
        model_ref: The model definition describing its API type and supports.
        label: The human-readable label to show in the Dev UI.

    Returns:
        The serialized :class:`ModelInfo` metadata (camelCase aliases, no
        ``None`` values) ready to embed under ``metadata['model']``.
    """
    is_chat = model_ref.api_type == OllamaAPITypes.CHAT
    return ModelInfo(
        label=label,
        supports=Supports(
            multiturn=is_chat,
            media=is_chat and model_ref.supports.media,
            tools=is_chat and model_ref.supports.tools,
            system_role=True,
            # Deliberate JS/Go deviation. we match other Python plugins for Dev UI consistency.
            output=['text', 'json'],
            constrained=Constrained.ALL,
        ),
    ).model_dump(by_alias=True, exclude_none=True)


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
            models: An Optional list of model definitions to be registered with Genkit.
            embedders: An Optional list of embedding model definitions to be
                registered with Genkit.
            server_address: The URL of the Ollama server. Defaults to a predefined
                Ollama server URL if not provided.
            request_headers: Optional HTTP headers to include with requests to the
                Ollama server. May be a static dict, or a sync/async callable that
                returns a dict. A callable is resolved per request (matching the JS
                plugin) so expiring auth tokens take effect; a static dict is applied
                once to a cached client.
            timeout: Optional request timeout (seconds) forwarded to the underlying
                httpx client.
        """
        self.models = models or []
        self.embedders = embedders or []
        self.server_address = server_address or DEFAULT_OLLAMA_SERVER_URL

        self._request_headers_source = request_headers
        # Static dicts are baked into the cached client; callables resolve per request.
        self.request_headers = dict(request_headers) if isinstance(request_headers, dict) else {}
        self.timeout = timeout
        self.client = loop_local_client(self._make_client)

    def _make_client(self, headers: dict[str, str] | None = None) -> ollama_api.AsyncClient:
        """Build an Ollama AsyncClient with the given (or static) headers and timeout.

        Args:
            headers: Per-request headers to use instead of the static ``request_headers``
                (e.g. resolved from a callable). Defaults to the static headers, which is
                what the per-event-loop cached client is built with.

        Returns:
            A new ``ollama.AsyncClient`` targeting the configured server.
        """
        kwargs: dict[str, Any] = {
            'host': self.server_address,
            'headers': self.request_headers if headers is None else headers,
        }
        if self.timeout is not None:
            kwargs['timeout'] = self.timeout
        return ollama_api.AsyncClient(**kwargs)

    async def _client_for_request(self) -> ollama_api.AsyncClient:
        """Return the Ollama client to use for a single request.

        Static (or absent) headers are baked into a per-event-loop cached client.
        A header *callable* is resolved on every call and applied to a fresh client,
        so expiring auth tokens or per-request headers take effect — matching the JS
        plugin, which resolves ``request_headers`` per request.

        Returns:
            The Ollama client for this request.
        """
        if callable(self._request_headers_source):
            headers = await self._resolve_request_headers()
            return self._make_client(headers=headers)
        return self.client()

    async def _resolve_request_headers(self) -> dict[str, str]:
        """Resolve the configured request headers to a plain dict.

        Static dicts pass through; sync/async callables are invoked once.

        Returns:
            The resolved HTTP headers.
        """
        source = self._request_headers_source
        if source is None:
            return {}
        if isinstance(source, dict):
            return dict(source)
        result = source()
        if inspect.isawaitable(result):
            result = await result
        return dict(cast(dict[str, str], result))

    async def init(self) -> list:
        """Initialize the Ollama plugin.

        Returns pre-registered models and embedders.

        Returns:
            List of Action objects for pre-configured models and embedders.
        """
        # Header callables are resolved per request (see _client_for_request), so
        # there is nothing to resolve eagerly here; static headers are already set.
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

        # If not found in pre-configured models, create a generic one. Dynamically
        # resolved models advertise the full capability set (see JS/Go parity note
        # on _DYNAMIC_MODEL_SUPPORTS).
        if model_ref is None:
            model_ref = ModelDefinition(name=clean_name, supports=_DYNAMIC_MODEL_SUPPORTS)

        model = OllamaModel(
            client=self.client,
            model_definition=model_ref,
            server_address=self.server_address,
        )

        action_metadata = model_action_metadata(
            name=name,
            config_schema=OllamaConfig,
            info=ollama_model_info(model_ref, f'Ollama - {clean_name}'),
        )

        async def _run(request: ModelRequest, ctx: ActionRunContext | None = None) -> ModelResponse:
            # Resolve per-request headers (no-op for static headers) and let
            # OllamaModel wrap connection errors at the Ollama SDK boundary, so a
            # failed media-URL fetch isn't misreported as an Ollama server outage.
            client = await self._client_for_request()
            return await model.generate(request, ctx, client=client)

        action = Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=_run,
            metadata=action_metadata.metadata,
        )

        # Explicitly set schemas (always present in the action metadata).
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
            # Embedding requests never fetch media, so the whole SDK call is wrapped.
            client = await self._client_for_request()
            async with wrap_connection_errors(server_address):
                return await embedder.embed(request, client=client)

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
        client = await self._client_for_request()
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
                            ModelDefinition(name=name, supports=_DYNAMIC_MODEL_SUPPORTS),
                            f'Ollama - {name}',
                        ),
                    )
                )
        return actions
