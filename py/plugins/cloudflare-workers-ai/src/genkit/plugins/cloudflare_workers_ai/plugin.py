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

"""CF Workers AI Plugin implementation for Genkit - Cloudflare Workers AI.

This module contains the main plugin class that registers Cloudflare Workers AI
models with the Genkit framework.

See: https://developers.cloudflare.com/workers-ai/get-started/rest-api/

Environment Variables:
    CLOUDFLARE_ACCOUNT_ID: Your Cloudflare account ID.
    CLOUDFLARE_API_TOKEN: Your Cloudflare API token with Workers AI permissions.
"""

import os

import httpx

from genkit.ai import ActionRunContext, Plugin
from genkit.blocks.model import model_action_metadata
from genkit.core.action import Action, ActionMetadata
from genkit.core.action.types import ActionKind
from genkit.core.http_client import get_cached_client
from genkit.core.logging import get_logger
from genkit.core.schema import to_json_schema
from genkit.plugins.cloudflare_workers_ai.embedders.embedder import CfEmbedder
from genkit.plugins.cloudflare_workers_ai.models.model import CfModel
from genkit.plugins.cloudflare_workers_ai.models.model_info import (
    SUPPORTED_CF_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    get_model_info,
)
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig
from genkit.types import (
    Embedding,
    EmbedRequest,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    TextPart,
)

logger = get_logger(__name__)

CLOUDFLARE_WORKERS_AI_PLUGIN_NAME = 'cloudflare-workers-ai'


def cloudflare_name(model_id: str) -> str:
    """Create a fully qualified model name for Cloudflare AI models.

    Args:
        model_id: The Cloudflare model ID (e.g., '@cf/meta/llama-3.1-8b-instruct').

    Returns:
        Fully qualified model name (e.g., 'cloudflare-workers-ai/@cf/meta/llama-3.1-8b-instruct').
    """
    return f'{CLOUDFLARE_WORKERS_AI_PLUGIN_NAME}/{model_id}'


# Convenience alias
cloudflare_model = cloudflare_name


class CloudflareWorkersAI(Plugin):
    """Cloudflare Workers AI plugin for Genkit.

    This plugin provides access to Cloudflare Workers AI models for text
    generation and embeddings.

    Example:
        >>> from genkit import Genkit
        >>> from genkit.plugins.cloudflare_workers_ai import CloudflareWorkersAI, cloudflare_model
        >>>
        >>> ai = Genkit(
        ...     plugins=[CloudflareWorkersAI()],
        ...     model=cloudflare_model('@cf/meta/llama-3.1-8b-instruct'),
        ... )
        >>>
        >>> response = await ai.generate(prompt='Hello, world!')

    Attributes:
        account_id: Cloudflare account ID.
        api_token: Cloudflare API token.
        models: List of model IDs to register (default: all supported models).
        embedders: List of embedder IDs to register (default: all supported embedders).
    """

    name = CLOUDFLARE_WORKERS_AI_PLUGIN_NAME

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        models: list[str] | None = None,
        embedders: list[str] | None = None,
    ) -> None:
        """Initialize the CF AI plugin.

        Args:
            account_id: Cloudflare account ID. Defaults to CLOUDFLARE_ACCOUNT_ID env var.
            api_token: Cloudflare API token. Defaults to CLOUDFLARE_API_TOKEN env var.
            models: List of model IDs to register. Defaults to all supported models.
            embedders: List of embedder IDs to register. Defaults to all supported embedders.

        Raises:
            ValueError: If account_id or api_token is not provided and not in environment.
        """
        self._account_id = account_id or os.environ.get('CLOUDFLARE_ACCOUNT_ID')
        self._api_token = api_token or os.environ.get('CLOUDFLARE_API_TOKEN')

        if not self._account_id:
            raise ValueError(
                'Cloudflare account ID is required. '
                'Set CLOUDFLARE_ACCOUNT_ID environment variable or pass account_id parameter.'
            )

        if not self._api_token:
            raise ValueError(
                'Cloudflare API token is required. '
                'Set CLOUDFLARE_API_TOKEN environment variable or pass api_token parameter.'
            )

        self._models = models or list(SUPPORTED_CF_MODELS.keys())
        self._embedders = embedders or list(SUPPORTED_EMBEDDING_MODELS.keys())

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create an httpx client for the current event loop.

        Uses the shared per-event-loop cache from genkit.core.http_client to avoid
        creating a new client for every request while still handling multiple event
        loops correctly.

        The Cloudflare API token is static for the plugin lifetime, so it's safe
        to include in the cached client headers (unlike Google Cloud tokens which
        may expire).

        Returns:
            Configured httpx.AsyncClient for the current event loop.
        """
        return get_cached_client(
            cache_key=f'cloudflare-workers-ai/{self._account_id}',
            headers={
                'Authorization': f'Bearer {self._api_token}',
                'Content-Type': 'application/json',
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Empty list (using lazy loading via resolve).
        """
        return []

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

        if action_type == ActionKind.EMBEDDER:
            return self._create_embedder_action(name)

        return None

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for a Cloudflare AI model.

        Args:
            name: The namespaced name of the model.

        Returns:
            Action object for the model.
        """
        # Extract local name (remove plugin prefix)
        prefix = f'{CLOUDFLARE_WORKERS_AI_PLUGIN_NAME}/'
        clean_name = name.replace(prefix, '') if name.startswith(prefix) else name

        model_info = get_model_info(clean_name)

        async def generate_fn(
            request: GenerateRequest,
            ctx: ActionRunContext,
        ) -> GenerateResponse:
            model = CfModel(
                model_id=clean_name,
                account_id=self._account_id,  # type: ignore[arg-type]
                client=self._get_client(),
            )
            return await model.generate(request, ctx)

        return Action(
            kind=ActionKind.MODEL,
            name=name,
            fn=generate_fn,
            metadata={
                'model': {
                    'supports': model_info.supports.model_dump() if model_info.supports else {},
                    'customOptions': to_json_schema(CloudflareConfig),
                },
            },
        )

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for a Cloudflare AI embedder.

        Args:
            name: The namespaced name of the embedder.

        Returns:
            Action object for the embedder.
        """
        # Extract local name (remove plugin prefix)
        prefix = f'{CLOUDFLARE_WORKERS_AI_PLUGIN_NAME}/'
        clean_name = name.replace(prefix, '') if name.startswith(prefix) else name

        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(clean_name, {})

        async def embed_fn(request: EmbedRequest) -> EmbedResponse:
            embedder = CfEmbedder(
                model_id=clean_name,
                account_id=self._account_id,  # type: ignore[arg-type]
                client=self._get_client(),
            )

            # Extract text from document parts
            documents: list[str] = []
            for doc in request.input:
                for part in doc.content:
                    if isinstance(part.root, TextPart):
                        documents.append(part.root.text)

            # Extract pooling option if specified
            pooling: str | None = None
            if request.options and isinstance(request.options, dict):
                pooling = request.options.get('pooling')

            # Get embeddings
            embeddings = await embedder.embed(documents, pooling=pooling)

            return EmbedResponse(
                embeddings=[Embedding(embedding=emb) for emb in embeddings],
            )

        return Action(
            kind=ActionKind.EMBEDDER,
            name=name,
            fn=embed_fn,
            metadata={
                'embedder': {
                    'name': name,
                    'info': embedder_info,
                },
            },
        )

    async def list_actions(self) -> list[ActionMetadata]:
        """List all available actions from this plugin.

        Returns:
            List of ActionMetadata objects.
        """
        actions: list[ActionMetadata] = []

        # List models
        for model_id in self._models:
            model_info = get_model_info(model_id)
            actions.append(
                model_action_metadata(
                    name=cloudflare_name(model_id),
                    info={'supports': model_info.supports.model_dump() if model_info.supports else {}},
                    config_schema=CloudflareConfig,
                )
            )

        # List embedders
        for embedder_id in self._embedders:
            embedder_info = SUPPORTED_EMBEDDING_MODELS.get(embedder_id, {})
            actions.append(
                ActionMetadata(
                    kind=ActionKind.EMBEDDER,
                    name=cloudflare_name(embedder_id),
                    description=f'Cloudflare AI embedder: {embedder_id}',
                    metadata={'embedder': {'info': embedder_info}},
                )
            )

        return actions


__all__ = [
    # Plugin name
    'CLOUDFLARE_WORKERS_AI_PLUGIN_NAME',
    # Plugin class
    'CloudflareWorkersAI',
    # Helper functions
    'cloudflare_model',
    'cloudflare_name',
]
