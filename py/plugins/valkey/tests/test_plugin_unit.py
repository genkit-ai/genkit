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

"""Unit tests for the Valkey plugin error paths.

These tests use mocked GlideClient and embedder to verify error propagation
without requiring a running Valkey instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit._ai._retriever import IndexerRequest, RetrieverRequest
from genkit._core._model import Document
from genkit._core._typing import EmbedResponse, Embedding
from genkit.plugins.valkey.plugin import (
    Valkey,
    ValkeyConfig,
    _doc_id,
    _float32_to_bytes,
)


# --- Helper utilities ---


class FakeRegistry:
    """Minimal registry mock for plugin tests."""

    def __init__(self, embedder_fn=None, embedder_error=None):
        self._embedder_fn = embedder_fn
        self._embedder_error = embedder_error

    async def resolve_embedder(self, name: str):
        if self._embedder_error:
            raise self._embedder_error
        action = AsyncMock()
        action.run = self._embedder_fn
        return action


def make_embed_fn(embeddings: list[list[float]] | None = None, error: Exception | None = None):
    """Create a mock embedder function that returns given embeddings or raises an error."""
    async def embed_fn(req):
        if error:
            raise error
        vecs = embeddings or []
        resp = MagicMock()
        resp.response = EmbedResponse(
            embeddings=[Embedding(embedding=v) for v in vecs]
        )
        return resp
    return embed_fn


# --- Unit tests for utility functions ---


class TestDocId:
    def test_deterministic(self):
        d1 = Document.from_text('hello')
        d2 = Document.from_text('hello')
        assert _doc_id(d1) == _doc_id(d2)

    def test_different_content(self):
        d1 = Document.from_text('hello')
        d2 = Document.from_text('world')
        assert _doc_id(d1) != _doc_id(d2)

    def test_md5_hex_length(self):
        d = Document.from_text('test')
        assert len(_doc_id(d)) == 32


class TestFloat32ToBytes:
    def test_basic_conversion(self):
        result = _float32_to_bytes([1.0, 2.0, 3.0])
        assert len(result) == 12  # 3 floats * 4 bytes

    def test_empty_list(self):
        result = _float32_to_bytes([])
        assert result == b''

    def test_round_trip(self):
        import struct
        values = [1.5, -2.5, 0.0]
        result = _float32_to_bytes(values)
        unpacked = struct.unpack(f'<{len(values)}f', result)
        for expected, actual in zip(values, unpacked):
            assert abs(expected - actual) < 1e-6


# --- Unit tests for indexer error paths ---


@pytest.mark.asyncio
class TestIndexerErrors:
    async def _make_index_fn(self, embed_fn, dimension=3):
        """Create an indexer function with a mocked client and embedder."""
        cfg = ValkeyConfig(
            index_name='test-index',
            embedder='mock/embedder',
            dimension=dimension,
            host='localhost',
            port=6379,
        )
        plugin = Valkey(configs=[cfg])
        mock_client = AsyncMock()
        mock_client.hset = AsyncMock(return_value=1)

        registry = MagicMock()
        registry.resolve_embedder = AsyncMock(return_value=MagicMock(run=embed_fn))
        plugin._registry = registry

        prefix = cfg.prefix or cfg.index_name
        index_fn = plugin._make_index_fn(mock_client, cfg, prefix)
        return index_fn, mock_client

    async def test_embedder_error_propagates(self):
        """Embedder errors should propagate to the caller."""
        embed_fn = make_embed_fn(error=RuntimeError('connection timeout'))
        index_fn, _ = await self._make_index_fn(embed_fn)

        docs = [Document.from_text('hello')]
        with pytest.raises(RuntimeError, match='connection timeout'):
            await index_fn(IndexerRequest(documents=docs))

    async def test_embedding_count_mismatch(self):
        """Should raise when embedder returns wrong number of embeddings."""
        # Return 1 embedding for 2 documents
        embed_fn = make_embed_fn(embeddings=[[0.1, 0.2, 0.3]])
        index_fn, _ = await self._make_index_fn(embed_fn)

        docs = [Document.from_text('doc1'), Document.from_text('doc2')]
        with pytest.raises(ValueError, match='1 embeddings for 2 docs'):
            await index_fn(IndexerRequest(documents=docs))

    async def test_dimension_mismatch(self):
        """Should raise when embedding dimension doesn't match config."""
        # Config expects dim=3, embedder returns dim=2
        embed_fn = make_embed_fn(embeddings=[[0.1, 0.2]])
        index_fn, _ = await self._make_index_fn(embed_fn, dimension=3)

        docs = [Document.from_text('hello')]
        with pytest.raises(ValueError, match='2-dim vector, expected 3'):
            await index_fn(IndexerRequest(documents=docs))

    async def test_empty_documents_succeeds(self):
        """Indexing empty list should succeed (embedder returns 0 embeddings, loop is no-op)."""
        embed_fn = make_embed_fn(embeddings=[])
        index_fn, _ = await self._make_index_fn(embed_fn)

        result = await index_fn(IndexerRequest(documents=[]))
        assert result is not None


# --- Unit tests for retriever error paths ---


@pytest.mark.asyncio
class TestRetrieverErrors:
    async def _make_retrieve_fn(self, embed_fn):
        """Create a retriever function with a mocked client and embedder."""
        cfg = ValkeyConfig(
            index_name='test-index',
            embedder='mock/embedder',
            dimension=3,
            host='localhost',
            port=6379,
        )
        plugin = Valkey(configs=[cfg])
        mock_client = AsyncMock()

        registry = MagicMock()
        registry.resolve_embedder = AsyncMock(return_value=MagicMock(run=embed_fn))
        plugin._registry = registry

        prefix = cfg.prefix or cfg.index_name
        retrieve_fn = plugin._make_retrieve_fn(mock_client, cfg, prefix)
        return retrieve_fn, mock_client

    async def test_embedder_error_propagates(self):
        """Embedder errors should propagate to the caller."""
        embed_fn = make_embed_fn(error=RuntimeError('rate limited'))
        retrieve_fn, _ = await self._make_retrieve_fn(embed_fn)

        req = RetrieverRequest(query=Document.from_text('query'))
        with pytest.raises(RuntimeError, match='rate limited'):
            await retrieve_fn(req)

    async def test_empty_embeddings_raises(self):
        """Should raise when embedder returns no embeddings."""
        embed_fn = make_embed_fn(embeddings=[])
        retrieve_fn, _ = await self._make_retrieve_fn(embed_fn)

        req = RetrieverRequest(query=Document.from_text('query'))
        with pytest.raises(ValueError, match='no embeddings'):
            await retrieve_fn(req)
