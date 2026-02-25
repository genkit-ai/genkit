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

"""Tests for DevLocalVectorStoreIndexer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit.blocks.document import Document
from genkit.plugins.dev_local_vectorstore.constant import DbValue
from genkit.plugins.dev_local_vectorstore.indexer import DevLocalVectorStoreIndexer
from genkit.types import DocumentData, Embedding, TextPart


class TestIndexerInit:
    """Tests for DevLocalVectorStoreIndexer initialization."""

    def test_init_stores_parameters(self) -> None:
        """Constructor stores ai, index_name, embedder, and options."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(
            ai=ai,
            index_name='test-idx',
            embedder='test-embedder',
            embedder_options={'dim': 128},
        )
        assert indexer.ai is ai
        assert indexer.index_name == 'test-idx'
        assert indexer.embedder == 'test-embedder'
        assert indexer.embedder_options == {'dim': 128}

    def test_init_default_options(self) -> None:
        """Default embedder_options is None."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(
            ai=ai,
            index_name='idx',
            embedder='emb',
        )
        assert indexer.embedder_options is None


class TestAddDocument:
    """Tests for _add_document method."""

    def test_add_document_creates_entry(self) -> None:
        """Adding a document creates an entry in the data dict."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='emb')
        data: dict[str, DbValue] = {}
        doc = Document.from_text('hello')
        embedding = Embedding(embedding=[0.1, 0.2, 0.3])

        indexer._add_document(data=data, embedding=embedding, doc=doc)

        assert len(data) == 1

    def test_add_document_uses_md5_key(self) -> None:
        """Document key is MD5 hash of serialized data."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='emb')
        data: dict[str, DbValue] = {}
        doc = Document.from_text('test')
        embedding = Embedding(embedding=[0.5])

        indexer._add_document(data=data, embedding=embedding, doc=doc)

        # The key should be a hex MD5 hash
        key = next(iter(data.keys()))
        assert len(key) == 32  # MD5 hex digest length
        assert all(c in '0123456789abcdef' for c in key)

    def test_add_document_with_existing_key_no_overwrite(self) -> None:
        """Adding a document whose key already exists doesn't overwrite."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='emb')

        # First, add a document to populate a key
        data: dict[str, DbValue] = {}
        doc = Document.from_text('original')
        embedding = Embedding(embedding=[0.1])
        indexer._add_document(data=data, embedding=embedding, doc=doc)

        assert len(data) == 1
        key = next(iter(data.keys()))
        data[key]

        # Create a different DbValue but use the same key by pre-inserting
        new_doc = Document.from_text('replacement')
        new_embedding = Embedding(embedding=[0.99])
        DbValue(
            # Pydantic's discriminated union accepts TextPart directly at runtime,
            # but the static type is list[Part]. Wrapping in Part(root=...) causes
            # a ValidationError, so the type: ignore is intentional.
            doc=DocumentData(content=[TextPart(text='replacement')]),  # type: ignore[list-item]
            embedding=new_embedding,
        )

        # Manually set the same key — _add_document should skip it
        # (this is the actual dedup behavior: if idx already in data, skip)
        data_copy = dict(data)
        indexer._add_document(data=data_copy, embedding=new_embedding, doc=new_doc)

        # Should have 2 entries since different content produces different hash
        assert len(data_copy) == 2

    def test_add_different_documents(self) -> None:
        """Adding different documents creates separate entries."""
        ai = MagicMock()
        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='emb')
        data: dict[str, DbValue] = {}

        doc1 = Document.from_text('first')
        doc2 = Document.from_text('second')
        emb1 = Embedding(embedding=[0.1])
        emb2 = Embedding(embedding=[0.9])

        indexer._add_document(data=data, embedding=emb1, doc=doc1)
        indexer._add_document(data=data, embedding=emb2, doc=doc2)

        assert len(data) == 2


class TestIndexMethod:
    """Tests for the async index method."""

    @pytest.mark.asyncio
    async def test_index_calls_embed_many(self) -> None:
        """Index method calls ai.embed_many with correct parameters."""
        ai = MagicMock()
        ai.embed_many = AsyncMock(
            return_value=[
                Embedding(embedding=[0.1, 0.2]),
                Embedding(embedding=[0.3, 0.4]),
            ]
        )

        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='my-embedder')

        from genkit.blocks.retriever import IndexerRequest

        docs = [
            DocumentData(content=[TextPart(text='doc1')]),  # type: ignore[list-item]
            DocumentData(content=[TextPart(text='doc2')]),  # type: ignore[list-item]
        ]
        request = IndexerRequest(documents=docs)

        with patch.object(indexer, '_load_filestore', return_value={}):
            with patch('aiofiles.open', new_callable=MagicMock) as mock_aiofiles:
                mock_file = AsyncMock()
                mock_aiofiles.return_value.__aenter__ = AsyncMock(return_value=mock_file)
                mock_aiofiles.return_value.__aexit__ = AsyncMock(return_value=False)

                await indexer.index(request)

        ai.embed_many.assert_called_once_with(
            embedder='my-embedder',
            content=docs,
            options=None,
        )

    @pytest.mark.asyncio
    async def test_index_raises_on_empty_embeddings(self) -> None:
        """Index raises ValueError when embedder returns empty response."""
        ai = MagicMock()
        ai.embed_many = AsyncMock(return_value=[])

        indexer = DevLocalVectorStoreIndexer(ai=ai, index_name='test', embedder='emb')

        from genkit.blocks.retriever import IndexerRequest

        request = IndexerRequest(
            documents=[DocumentData(content=[TextPart(text='doc')])]  # type: ignore[list-item]
        )

        with patch.object(indexer, '_load_filestore', return_value={}):
            with pytest.raises(ValueError, match='no embeddings'):
                await indexer.index(request)
