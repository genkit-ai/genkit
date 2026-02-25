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

"""Tests for Cohere embeddings configuration and embedding extraction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from genkit.blocks.document import Document
from genkit.blocks.embedding import EmbedRequest
from genkit.plugins.cohere.embeddings import CohereEmbedConfig, CohereEmbedder


class TestCohereEmbedConfig:
    """Tests for CohereEmbedConfig validation."""

    def test_defaults(self) -> None:
        """Test defaults."""
        config = CohereEmbedConfig()
        assert config.input_type is None
        assert config.embedding_types is None
        assert config.truncate is None

    def test_valid_input_types(self) -> None:
        """Test valid input types."""
        for it in ('search_document', 'search_query', 'classification', 'clustering'):
            config = CohereEmbedConfig(input_type=it)
            assert config.input_type == it

    def test_valid_embedding_types(self) -> None:
        """Test valid embedding types."""
        config = CohereEmbedConfig(embedding_types=['float', 'int8'])
        assert config.embedding_types == ['float', 'int8']

    def test_valid_truncate_values(self) -> None:
        """Test valid truncate values."""
        for t in ('NONE', 'START', 'END'):
            config = CohereEmbedConfig(truncate=t)
            assert config.truncate == t


class TestCohereEmbedder:
    """Tests for CohereEmbedder.embed()."""

    def test_init(self) -> None:
        """Test init."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        assert embedder.model == 'embed-v4.0'

    @pytest.mark.asyncio
    async def test_embed_extracts_float_embeddings(self) -> None:
        """Test embed extracts float embeddings."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(input=[Document.from_text('hello'), Document.from_text('world')])
        result = await embedder.embed(request)

        assert len(result.embeddings) == 2
        assert result.embeddings[0].embedding == [0.1, 0.2, 0.3]
        assert result.embeddings[1].embedding == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_raises_on_raw_list_embeddings(self) -> None:
        """Test embed raises on raw list embeddings without float_ attribute."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=[[1.0, 2.0], [3.0, 4.0]])
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(input=[Document.from_text('a'), Document.from_text('b')])
        with pytest.raises(ValueError, match='float embeddings'):
            await embedder.embed(request)

    @pytest.mark.asyncio
    async def test_embed_raises_on_non_float(self) -> None:
        """Test embed raises on non float."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=SimpleNamespace(float_=None, int8=[[1, 2, 3]]))
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(input=[Document.from_text('test')])
        with pytest.raises(ValueError, match='float embeddings'):
            await embedder.embed(request)

    @pytest.mark.asyncio
    async def test_embed_passes_options(self) -> None:
        """Test embed passes options."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1]]))
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(
            input=[Document.from_text('test')],
            options={
                'input_type': 'search_query',
                'embedding_types': ['float'],
                'truncate': 'END',
            },
        )
        await embedder.embed(request)

        call_kwargs = embedder.client.embed.call_args
        assert call_kwargs.kwargs['input_type'] == 'search_query'
        assert call_kwargs.kwargs['embedding_types'] == ['float']
        assert call_kwargs.kwargs['truncate'] == 'END'

    @pytest.mark.asyncio
    async def test_embed_default_input_type(self) -> None:
        """Test embed default input type."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1]]))
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(input=[Document.from_text('test')])
        await embedder.embed(request)

        call_kwargs = embedder.client.embed.call_args
        assert call_kwargs.kwargs['input_type'] == 'search_document'

    @pytest.mark.asyncio
    async def test_embed_single_embedding_type_string(self) -> None:
        """Test embed single embedding type string."""
        embedder = CohereEmbedder(model='embed-v4.0', api_key='test-key')
        mock_response = SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1]]))
        embedder.client = AsyncMock()
        embedder.client.embed = AsyncMock(return_value=mock_response)

        request = EmbedRequest(
            input=[Document.from_text('test')],
            options={'embedding_types': 'float'},
        )
        await embedder.embed(request)

        call_kwargs = embedder.client.embed.call_args
        assert call_kwargs.kwargs['embedding_types'] == ['float']
