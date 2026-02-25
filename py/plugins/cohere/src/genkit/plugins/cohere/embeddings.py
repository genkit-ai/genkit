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

"""Cohere AI embeddings integration for Genkit.

This module provides embedding support using Cohere's Embed models.
Embeddings convert text into dense vector representations for use in
semantic search, retrieval-augmented generation (RAG), clustering, and
similarity comparisons.

See: https://docs.cohere.com/reference/embed
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

import cohere
from genkit.blocks.embedding import EmbedRequest, EmbedResponse
from genkit.plugins.cohere.model_info import SUPPORTED_EMBEDDING_MODELS
from genkit.types import Embedding, TextPart

__all__ = [
    'SUPPORTED_EMBEDDING_MODELS',
    'CohereEmbedConfig',
    'CohereEmbedder',
]


class CohereEmbedConfig(BaseModel):
    """Configuration options for Cohere embedding requests.

    Attributes:
        input_type: The type of input text. Helps the model optimize.
            Use ``search_document`` for indexing and ``search_query``
            for retrieval queries.
        embedding_types: The data type(s) for the returned embeddings.
        truncate: How to handle texts exceeding the token limit.
    """

    input_type: Literal['search_document', 'search_query', 'classification', 'clustering'] | None = None
    embedding_types: list[Literal['float', 'int8', 'uint8', 'binary', 'ubinary']] | None = None
    truncate: Literal['NONE', 'START', 'END'] | None = Field(default=None)


class CohereEmbedder:
    """Handles embedding requests using a Cohere Embed model.

    Converts Genkit ``EmbedRequest`` documents into vectors by calling
    the Cohere V2 ``embed`` API and returning the results as a Genkit
    ``EmbedResponse``.
    """

    def __init__(self, model: str, api_key: str) -> None:
        """Initialize the Cohere embedder.

        Args:
            model: The model identifier (e.g. ``embed-v4.0``).
            api_key: Cohere API key.
        """
        self.model = model
        self.client = cohere.AsyncClientV2(api_key=api_key)

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Generate embeddings for the given documents.

        The Cohere V2 embed API returns ``EmbedByTypeResponse`` with
        embeddings grouped by type (float, int8, etc.). We extract
        float embeddings by default.

        Args:
            request: The embedding request containing input documents.

        Returns:
            An ``EmbedResponse`` with one embedding per input document.
        """
        # Extract text from each document's content parts.
        texts: list[str] = []
        for doc in request.input:
            doc_text = ''.join(
                part.root.text for part in doc.content if isinstance(part.root, TextPart) and part.root.text
            )
            texts.append(doc_text)

        # Build optional parameters from request options.
        kwargs: dict[str, Any] = {}
        if request.options:
            if embedding_types := request.options.get('embedding_types'):
                if isinstance(embedding_types, list):
                    kwargs['embedding_types'] = embedding_types
                else:
                    kwargs['embedding_types'] = [str(embedding_types)]
            if truncate := request.options.get('truncate'):
                kwargs['truncate'] = str(truncate)

        # Determine input_type — required by the V2 API.
        input_type = 'search_document'
        if request.options and (it := request.options.get('input_type')):
            input_type = str(it)

        response = await self.client.embed(
            model=self.model,
            texts=texts,
            input_type=input_type,
            **kwargs,
        )

        # EmbedByTypeResponse.embeddings is EmbedByTypeResponseEmbeddings.
        # Extract float embeddings — the only type Genkit supports.
        emb = response.embeddings
        vectors: list[list[float]] = []
        if hasattr(emb, 'float_') and emb.float_ is not None:
            vectors = emb.float_
        else:
            raise ValueError(
                'Cohere API response did not include float embeddings, '
                'which are required by Genkit. Ensure embedding_types '
                "includes 'float' or is left unset."
            )

        embeddings = [Embedding(embedding=vec) for vec in vectors]
        return EmbedResponse(embeddings=embeddings)
