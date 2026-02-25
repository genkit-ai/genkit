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

"""Mistral AI embeddings integration for Genkit.

This module provides embedding support using Mistral AI's ``mistral-embed``
model. Embeddings convert text into dense vector representations for use in
semantic search, retrieval-augmented generation (RAG), clustering, and
similarity comparisons.

See: https://docs.mistral.ai/capabilities/embeddings/
"""

from typing import Any, Literal

from mistralai import Mistral as MistralClient
from pydantic import BaseModel, Field

from genkit.blocks.embedding import EmbedRequest, EmbedResponse
from genkit.types import Embedding, TextPart

__all__ = [
    'SUPPORTED_EMBEDDING_MODELS',
    'MistralEmbedConfig',
    'MistralEmbedder',
]

# Mistral's supported embedding models and their default dimensions.
# See: https://docs.mistral.ai/capabilities/embeddings/
# See: https://docs.mistral.ai/models/codestral-embed-25-05
SUPPORTED_EMBEDDING_MODELS: dict[str, dict[str, Any]] = {
    'mistral-embed': {
        'label': 'Mistral AI - Embed',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
    'codestral-embed-2505': {
        'label': 'Mistral AI - Codestral Embed',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
}


class MistralEmbedConfig(BaseModel):
    """Configuration options for Mistral embedding requests.

    Attributes:
        output_dimension: Optional dimensionality of the output embeddings.
            If not specified, the model's default dimension (1024) is used.
            Useful for reducing storage or matching an existing vector index.
        output_dtype: Optional data type for the returned embeddings. One of
            ``float``, ``int8``, ``uint8``, ``binary``, or ``ubinary``.
        encoding_format: Optional format of the returned embeddings. One of
            ``float`` or ``base64``.
    """

    output_dimension: int | None = Field(default=None, ge=1)
    output_dtype: Literal['float', 'int8', 'uint8', 'binary', 'ubinary'] | None = None
    encoding_format: Literal['float', 'base64'] | None = None


class MistralEmbedder:
    """Handles embedding requests using a Mistral AI embedding model.

    Converts Genkit ``EmbedRequest`` documents into vectors by calling
    the Mistral ``embeddings.create_async`` API and returning the results
    as a Genkit ``EmbedResponse``.
    """

    def __init__(self, model: str, client: MistralClient) -> None:
        """Initialize the Mistral embedder.

        Args:
            model: The model identifier (e.g. ``mistral-embed``).
            client: A configured ``MistralClient`` instance.
        """
        self.model = model
        self.client = client

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Generate embeddings for the given documents.

        Extracts text from each ``Document`` in the request, sends them to
        the Mistral embeddings API, and returns the resulting vectors.

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
            if dim_val := request.options.get('output_dimension'):
                kwargs['output_dimension'] = int(dim_val)
            if dtype_val := request.options.get('output_dtype'):
                kwargs['output_dtype'] = str(dtype_val)
            if enc_val := request.options.get('encoding_format'):
                kwargs['encoding_format'] = str(enc_val)

        response = await self.client.embeddings.create_async(
            model=self.model,
            inputs=texts,
            **kwargs,
        )

        embeddings = [Embedding(embedding=item.embedding) for item in response.data]
        return EmbedResponse(embeddings=embeddings)
