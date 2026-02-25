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

"""Tests for Google GenAI plugin."""

import os
from unittest.mock import MagicMock, patch

import pytest

from genkit.core.registry import ActionKind
from genkit.plugins.google_genai import (
    EmbeddingTaskType,
    GeminiConfigSchema,
    GeminiEmbeddingModels,
    GoogleAI,
    GoogleAIGeminiVersion,
    VertexAI,
    VertexAIGeminiVersion,
    VertexEmbeddingModels,
)
from genkit.plugins.google_genai.google import (
    GOOGLEAI_PLUGIN_NAME,
    VERTEXAI_PLUGIN_NAME,
    GenaiModels,
    googleai_name,
    vertexai_name,
)


def test_googleai_name() -> None:
    """Test googleai_name helper function."""
    assert googleai_name('gemini-2.0-flash') == 'googleai/gemini-2.0-flash'
    assert googleai_name('gemini-embedding-001') == 'googleai/gemini-embedding-001'


def test_vertexai_name() -> None:
    """Test vertexai_name helper function."""
    assert vertexai_name('gemini-2.0-flash') == 'vertexai/gemini-2.0-flash'
    assert vertexai_name('imagen-3.0-generate-001') == 'vertexai/imagen-3.0-generate-001'


def test_plugin_names() -> None:
    """Test plugin name constants."""
    assert GOOGLEAI_PLUGIN_NAME == 'googleai'
    assert VERTEXAI_PLUGIN_NAME == 'vertexai'


def test_googleai_initialization_with_api_key() -> None:
    """Test GoogleAI plugin initializes with API key parameter."""
    with patch('genkit.plugins.google_genai.google.genai.client.Client'):
        plugin = GoogleAI(api_key='test-key')
        assert plugin.name == 'googleai'
        assert plugin._vertexai is False


def test_googleai_initialization_from_env() -> None:
    """Test GoogleAI plugin reads API key from environment."""
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
        with patch('genkit.plugins.google_genai.google.genai.client.Client'):
            plugin = GoogleAI()
            assert plugin.name == 'googleai'


def test_googleai_initialization_without_api_key() -> None:
    """Test GoogleAI plugin raises error without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            GoogleAI()
        assert 'GEMINI_API_KEY' in str(exc_info.value)


def test_vertexai_initialization() -> None:
    """Test VertexAI plugin initializes correctly."""
    with patch('genkit.plugins.google_genai.google.genai.client.Client'):
        plugin = VertexAI(project='test-project', location='us-central1')
        assert plugin.name == 'vertexai'
        assert plugin._vertexai is True


def test_vertexai_initialization_from_env() -> None:
    """Test VertexAI plugin reads project from environment."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-project'}):
        with patch('genkit.plugins.google_genai.google.genai.client.Client'):
            plugin = VertexAI()
            assert plugin.name == 'vertexai'


def test_genai_models_container() -> None:
    """Test GenaiModels container initialization."""
    models = GenaiModels()
    assert models.gemini == []
    assert models.imagen == []
    assert models.embedders == []
    assert models.veo == []


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves model actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/gemini-2.0-flash')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'googleai/gemini-2.0-flash'


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_imagen_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves Imagen image generation models."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'googleai/imagen-3.0-generate-002')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'googleai/imagen-3.0-generate-002'


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_init_registers_imagen_models(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI init registers Imagen models from dynamic discovery."""
    models = GenaiModels()
    models.imagen = ['imagen-3.0-generate-002']
    mock_list_models.return_value = models

    plugin = GoogleAI(api_key='test-key')
    actions = await plugin.init()

    imagen_actions = [a for a in actions if 'imagen' in a.name]
    assert len(imagen_actions) == 1
    assert imagen_actions[0].name == 'googleai/imagen-3.0-generate-002'
    assert imagen_actions[0].kind == ActionKind.MODEL


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_list_actions_includes_imagen(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI list_actions includes Imagen models."""
    models = GenaiModels()
    models.imagen = ['imagen-3.0-generate-002']
    mock_list_models.return_value = models

    plugin = GoogleAI(api_key='test-key')
    actions_list = await plugin.list_actions()

    imagen_actions = [a for a in actions_list if 'imagen' in a.name]
    assert len(imagen_actions) == 1
    assert imagen_actions[0].name == 'googleai/imagen-3.0-generate-002'


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_embedder(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin resolves embedder actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'googleai/gemini-embedding-001')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'googleai/gemini-embedding-001'


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_googleai_resolve_non_model_returns_none(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test GoogleAI plugin returns None for unsupported action kinds."""
    mock_list_models.return_value = GenaiModels()

    plugin = GoogleAI(api_key='test-key')
    action = await plugin.resolve(ActionKind.PROMPT, 'some-prompt')
    assert action is None


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_model(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test VertexAI plugin resolves model actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.MODEL, 'vertexai/gemini-2.0-flash')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'vertexai/gemini-2.0-flash'


@patch('genkit.plugins.google_genai.google.genai.client.Client')
@patch('genkit.plugins.google_genai.google._list_genai_models')
@pytest.mark.asyncio
async def test_vertexai_resolve_embedder(mock_list_models: MagicMock, mock_client: MagicMock) -> None:
    """Test VertexAI plugin resolves embedder actions."""
    mock_list_models.return_value = GenaiModels()

    plugin = VertexAI(project='test-project')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'vertexai/gemini-embedding-001')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'vertexai/gemini-embedding-001'


def test_embedding_task_types() -> None:
    """Test EmbeddingTaskType enum values."""
    assert EmbeddingTaskType.RETRIEVAL_QUERY is not None
    assert EmbeddingTaskType.RETRIEVAL_DOCUMENT is not None
    assert EmbeddingTaskType.SEMANTIC_SIMILARITY is not None
    assert EmbeddingTaskType.CLASSIFICATION is not None
    assert EmbeddingTaskType.CLUSTERING is not None


def test_gemini_embedding_models_enum() -> None:
    """Test GeminiEmbeddingModels enum has values."""
    # Check that the enum has at least one value
    assert len(list(GeminiEmbeddingModels)) > 0


def test_vertex_embedding_models_enum() -> None:
    """Test VertexEmbeddingModels enum has values."""
    # Check that the enum has at least one value
    assert len(list(VertexEmbeddingModels)) > 0


def test_googleai_gemini_version_enum() -> None:
    """Test GoogleAIGeminiVersion enum has values."""
    # Check that the enum has at least one value
    assert len(list(GoogleAIGeminiVersion)) > 0


def test_vertexai_gemini_version_enum() -> None:
    """Test VertexAIGeminiVersion enum has values."""
    # Check that the enum has at least one value
    assert len(list(VertexAIGeminiVersion)) > 0


def test_gemini_config_schema() -> None:
    """Test GeminiConfigSchema can be instantiated."""
    config = GeminiConfigSchema(temperature=0.7, max_output_tokens=1000)
    assert config.temperature == 0.7
    assert config.max_output_tokens == 1000


def test_gemini_config_schema_defaults() -> None:
    """Test GeminiConfigSchema has proper defaults."""
    config = GeminiConfigSchema()
    # All fields should be optional with None defaults
    assert config.temperature is None
    assert config.max_output_tokens is None
