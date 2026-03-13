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

"""Tests for Mistral AI embeddings integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from genkit.blocks.document import Document
from genkit.blocks.embedding import EmbedRequest
from genkit.core.action.types import ActionKind
from genkit.plugins.mistral import Mistral
from genkit.plugins.mistral.embeddings import (
    SUPPORTED_EMBEDDING_MODELS,
    MistralEmbedConfig,
    MistralEmbedder,
)

# ---------------------------------------------------------------------------
# Unit tests for MistralEmbedder
# ---------------------------------------------------------------------------


def _make_embed_request(texts: list[str]) -> EmbedRequest:
    """Helper to build an EmbedRequest from a list of strings."""
    docs = [Document.from_text(t) for t in texts]
    return EmbedRequest(input=docs)


@pytest.mark.asyncio
async def test_embedder_embed_single_text() -> None:
    """Test embedding a single text document."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3]
    mock_response.data = [mock_item]
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)

    embedder = MistralEmbedder(model='mistral-embed', client=mock_client)
    request = _make_embed_request(['Hello world'])
    response = await embedder.embed(request)

    mock_client.embeddings.create_async.assert_called_once_with(
        model='mistral-embed',
        inputs=['Hello world'],
    )
    assert len(response.embeddings) == 1
    assert response.embeddings[0].embedding == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embedder_embed_multiple_texts() -> None:
    """Test embedding multiple text documents in a single call."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item1 = MagicMock()
    mock_item1.embedding = [0.1, 0.2]
    mock_item2 = MagicMock()
    mock_item2.embedding = [0.3, 0.4]
    mock_response.data = [mock_item1, mock_item2]
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)

    embedder = MistralEmbedder(model='mistral-embed', client=mock_client)
    request = _make_embed_request(['Hello', 'World'])
    response = await embedder.embed(request)

    assert len(response.embeddings) == 2
    assert response.embeddings[0].embedding == [0.1, 0.2]
    assert response.embeddings[1].embedding == [0.3, 0.4]


@pytest.mark.asyncio
async def test_embedder_passes_options() -> None:
    """Test that embedding options are forwarded to the API call."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_item = MagicMock()
    mock_item.embedding = [0.5]
    mock_response.data = [mock_item]
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)

    embedder = MistralEmbedder(model='mistral-embed', client=mock_client)
    request = _make_embed_request(['Test'])
    request.options = {
        'output_dimension': 512,
        'output_dtype': 'float',
        'encoding_format': 'float',
    }
    await embedder.embed(request)

    mock_client.embeddings.create_async.assert_called_once_with(
        model='mistral-embed',
        inputs=['Test'],
        output_dimension=512,
        output_dtype='float',
        encoding_format='float',
    )


@pytest.mark.asyncio
async def test_embedder_empty_options() -> None:
    """Test embedding with no options passes no extra kwargs."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = []
    mock_client.embeddings.create_async = AsyncMock(return_value=mock_response)

    embedder = MistralEmbedder(model='mistral-embed', client=mock_client)
    request = _make_embed_request([])
    await embedder.embed(request)

    mock_client.embeddings.create_async.assert_called_once_with(
        model='mistral-embed',
        inputs=[],
    )


# ---------------------------------------------------------------------------
# Unit tests for MistralEmbedConfig
# ---------------------------------------------------------------------------


def test_embed_config_defaults() -> None:
    """Test MistralEmbedConfig has sensible defaults."""
    config = MistralEmbedConfig()
    assert config.output_dimension is None
    assert config.output_dtype is None
    assert config.encoding_format is None


def test_embed_config_with_values() -> None:
    """Test MistralEmbedConfig accepts valid values."""
    config = MistralEmbedConfig(
        output_dimension=512,
        output_dtype='int8',
        encoding_format='base64',
    )
    assert config.output_dimension == 512
    assert config.output_dtype == 'int8'
    assert config.encoding_format == 'base64'


# ---------------------------------------------------------------------------
# Plugin integration tests for embedder
# ---------------------------------------------------------------------------


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_resolve_embedder_action(mock_client: MagicMock) -> None:
    """Test plugin resolves embedder actions for mistral-embed."""
    plugin = Mistral(api_key='test-key')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'mistral/mistral-embed')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'mistral/mistral-embed'


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_resolve_embedder_rejects_chat_models(mock_client: MagicMock) -> None:
    """Test plugin does not resolve chat models as embedders."""
    plugin = Mistral(api_key='test-key')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'mistral/mistral-large-latest')
    assert action is None


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_resolve_model_rejects_embedder(mock_client: MagicMock) -> None:
    """Test plugin does not resolve mistral-embed as a chat model."""
    plugin = Mistral(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'mistral/mistral-embed')
    assert action is None


@pytest.mark.asyncio
async def test_plugin_list_actions_includes_embedders() -> None:
    """Test list_actions includes both models and embedders."""
    plugin = Mistral(api_key='test-key')
    actions = await plugin.list_actions()

    action_names = [a.name for a in actions]
    assert 'mistral/mistral-embed' in action_names
    assert 'mistral/mistral-large-latest' in action_names

    # Embedder should appear exactly once and not also as a model.
    embed_actions = [a for a in actions if a.name == 'mistral/mistral-embed']
    assert len(embed_actions) == 1
    assert embed_actions[0].kind == ActionKind.EMBEDDER


def test_supported_embedding_models_metadata() -> None:
    """Test SUPPORTED_EMBEDDING_MODELS has required fields."""
    for name, info in SUPPORTED_EMBEDDING_MODELS.items():
        assert 'label' in info, f'Embedding model {name} missing label'
        assert 'dimensions' in info, f'Embedding model {name} missing dimensions'
        assert 'supports' in info, f'Embedding model {name} missing supports'
        assert 'input' in info['supports'], f'Embedding model {name} missing supports.input'
