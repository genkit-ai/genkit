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

"""Cloudflare Workers AI embedder implementation for Genkit.

This module implements the embedder interface for Cloudflare Workers AI
text embedding models.

See: https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/

Key Features:
    - Text embeddings using BGE, EmbeddingGemma, and Qwen models
    - Batch embedding support
    - Configurable embedding dimensions (where supported)
"""

from typing import Any

import httpx

from genkit.core.logging import get_logger
from genkit.plugins.cloudflare_workers_ai.constants import CF_API_BASE_URL
from genkit.plugins.cloudflare_workers_ai.models.model_info import SUPPORTED_EMBEDDING_MODELS

logger = get_logger(__name__)


class CfEmbedder:
    """Cloudflare Workers AI embedder for text embeddings.

    This class handles text embedding generation using Cloudflare's
    embedding models such as BGE and EmbeddingGemma.

    Attributes:
        model_id: The Cloudflare model ID (e.g., '@cf/baai/bge-base-en-v1.5').
        account_id: The Cloudflare account ID.
        client: httpx.AsyncClient for making API requests.
    """

    def __init__(
        self,
        model_id: str,
        account_id: str,
        client: httpx.AsyncClient,
    ) -> None:
        """Initialize the Cloudflare embedder.

        Args:
            model_id: Cloudflare embedding model ID.
            account_id: Cloudflare account ID.
            client: Configured httpx.AsyncClient with auth headers.
        """
        self.model_id = model_id
        self.account_id = account_id
        self.client = client
        self._model_info = SUPPORTED_EMBEDDING_MODELS.get(model_id, {})

    def _get_api_url(self) -> str:
        """Get the API URL for this model.

        Returns:
            Full URL for the model's inference endpoint.
        """
        return f'{CF_API_BASE_URL}/{self.account_id}/ai/run/{self.model_id}'

    async def embed(
        self,
        documents: list[str],
        pooling: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of documents.

        Args:
            documents: List of text strings to embed.
            pooling: Optional pooling method - 'mean' (default) or 'cls'.
                'cls' pooling generates more accurate embeddings on larger
                inputs, but embeddings created with 'cls' are NOT compatible
                with embeddings generated with 'mean'.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            httpx.HTTPStatusError: If the API returns an error status.
        """
        if not documents:
            return []

        logger.debug(
            'Cloudflare AI embed request',
            model_id=self.model_id,
            document_count=len(documents),
            pooling=pooling,
        )

        body: dict[str, Any] = {
            'text': documents,
        }

        # Add pooling method if specified
        if pooling:
            body['pooling'] = pooling

        try:
            response = await self.client.post(self._get_api_url(), json=body)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.exception(
                'Cloudflare AI embedding API call failed',
                model_id=self.model_id,
                status_code=e.response.status_code,
                error=str(e),
            )
            raise

        return self._parse_embedding_response(data)

    def _parse_embedding_response(
        self,
        data: dict[str, Any],
    ) -> list[list[float]]:
        """Parse embedding response from Cloudflare API.

        Args:
            data: Raw API response data.

        Returns:
            List of embedding vectors.
        """
        result = data.get('result', data)

        # The response format is {"data": [[...], [...], ...]}
        embeddings = result.get('data', [])

        # Ensure we return list of lists
        if embeddings and isinstance(embeddings[0], list):
            return embeddings

        # Single embedding case - wrap in list
        if embeddings and isinstance(embeddings[0], int | float):
            return [embeddings]

        return embeddings


__all__ = ['CF_API_BASE_URL', 'CfEmbedder']
