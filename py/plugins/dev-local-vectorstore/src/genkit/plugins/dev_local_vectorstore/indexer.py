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


"""Indexer for dev-local-vectorstore."""

import asyncio
import hashlib
import json
from typing import Any

from genkit.ai import Genkit
from genkit.blocks.document import Document
from genkit.blocks.retriever import IndexerRequest
from genkit.types import Embedding

from .constant import DbValue
from .local_vector_store_api import LocalVectorStoreAPI


class DevLocalVectorStoreIndexer(LocalVectorStoreAPI):
    """Indexer for development-level local vector store."""

    def __init__(
        self,
        ai: Genkit,
        index_name: str,
        embedder: str,
        embedder_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the DevLocalVectorStoreIndexer.

        Args:
            ai: Genkit instance used to embed documents.
            index_name: Name of the index.
            embedder: The embedder to use for document embeddings.
            embedder_options: Optional configuration to pass to the embedder.
        """
        super().__init__(index_name=index_name)
        self.ai = ai
        self.embedder = embedder
        self.embedder_options = embedder_options

    async def index(self, request: IndexerRequest) -> None:
        """Index documents into the local vector store."""
        docs = request.documents
        # pyrefly: ignore[missing-attribute] - inherited from LocalVectorStoreAPI
        data = await self._load_filestore()

        embed_resp = await self.ai.embed_many(
            embedder=self.embedder,
            content=docs,
            options=self.embedder_options,
        )
        if not embed_resp:
            raise ValueError('Embedder returned no embeddings for documents')

        tasks = []
        for doc_data, emb in zip(docs, embed_resp, strict=True):
            tasks.append(
                self.process_document(
                    document=Document.from_document_data(document_data=doc_data),
                    embedding=Embedding(embedding=emb.embedding),
                    data=data,
                )
            )

        await asyncio.gather(*tasks)

        # pyrefly: ignore[missing-attribute] - _dump_filestore inherited from LocalVectorStoreAPI
        await self._dump_filestore(data)

    async def process_document(self, document: Document, embedding: Embedding, data: dict[str, DbValue]) -> None:
        """Process a single document and add its embedding to the store."""
        embedding_docs = document.get_embedding_documents([embedding])
        self._add_document(data=data, embedding=embedding, doc=embedding_docs[0])

    def _add_document(
        self,
        data: dict[str, DbValue],
        embedding: Embedding,
        doc: Document,
    ) -> None:
        # pyrefly: ignore[missing-attribute] - _serialize_data inherited from LocalVectorStoreAPI
        data_str = json.dumps(self._serialize_data(data=data), ensure_ascii=False)
        # MD5 used for content-based ID generation, not security
        idx = hashlib.md5(data_str.encode('utf-8'), usedforsecurity=False).hexdigest()
        if idx not in data:
            data[idx] = DbValue(
                doc=doc,
                embedding=embedding,
            )
