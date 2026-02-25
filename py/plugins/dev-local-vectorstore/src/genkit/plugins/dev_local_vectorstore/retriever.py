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


"""Retriever for dev-local-vectorstore."""

from typing import Any

from pydantic import BaseModel, Field

from genkit.ai import ActionRunContext, Document, Genkit
from genkit.types import Embedding, RetrieverRequest, RetrieverResponse

from .local_vector_store_api import LocalVectorStoreAPI


class ScoredDocument(BaseModel):
    """Document with an associated similarity score."""

    score: float
    document: Document


class RetrieverOptionsSchema(BaseModel):
    """Schema for retriever options."""

    limit: int | None = Field(title='Number of documents to retrieve', default=None)


class DevLocalVectorStoreRetriever(LocalVectorStoreAPI):
    """Retriever for development-level local vector store."""

    def __init__(
        self,
        ai: Genkit,
        index_name: str,
        embedder: str,
        embedder_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the DevLocalVectorStoreRetriever.

        Args:
            ai: Genkit instance used to embed queries.
            index_name: Name of the index.
            embedder: The embedder to use for query embeddings.
            embedder_options: Optional configuration to pass to the embedder.
        """
        super().__init__(index_name=index_name)
        self.ai = ai
        self.embedder = embedder
        self.embedder_options = embedder_options

    async def retrieve(self, request: RetrieverRequest, _: ActionRunContext) -> RetrieverResponse:
        """Retrieve documents from the vector store."""
        document = Document.from_document_data(document_data=request.query)

        embed_resp = await self.ai.embed(
            embedder=self.embedder,
            content=document,
            options=self.embedder_options,
        )
        if not embed_resp:
            raise ValueError('Embedder returned no embeddings for query')

        k = 3
        if isinstance(request.options, dict) and (limit_val := request.options.get('limit')) is not None:
            k = int(limit_val)

        docs = await self._get_closest_documents(
            k=k,
            query_embeddings=Embedding(embedding=embed_resp[0].embedding),
        )

        return RetrieverResponse(documents=[d.document for d in docs])

    async def _get_closest_documents(self, k: int, query_embeddings: Embedding) -> list[ScoredDocument]:
        # pyrefly: ignore[missing-attribute] - _load_filestore inherited from LocalVectorStoreAPI
        db = await self._load_filestore()
        scored_documents = []

        for val in db.values():
            this_embedding = val.embedding.embedding
            score = self.cosine_similarity(query_embeddings.embedding, this_embedding)
            scored_documents.append(
                ScoredDocument(
                    score=score,
                    document=Document.from_document_data(document_data=val.doc),
                )
            )

        scored_documents = sorted(scored_documents, key=lambda d: d.score, reverse=True)
        return scored_documents[:k]

    @classmethod
    def cosine_similarity(cls, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return cls.dot(a, b) / ((cls.dot(a, a) ** 0.5) * (cls.dot(b, b) ** 0.5))

    @staticmethod
    def dot(a: list[float], b: list[float]) -> float:
        """Calculate dot product of two vectors."""
        return sum(av * bv for av, bv in zip(a, b, strict=False))
