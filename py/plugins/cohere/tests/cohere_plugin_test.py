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

"""Tests for Cohere AI plugin."""

import os
from unittest.mock import patch

import pytest

from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.plugins.cohere import (
    COHERE_PLUGIN_NAME,
    SUPPORTED_COHERE_MODELS,
    Cohere,
    cohere_name,
)
from genkit.plugins.cohere.embeddings import SUPPORTED_EMBEDDING_MODELS
from genkit.plugins.cohere.model_info import get_default_model_info


def test_cohere_name() -> None:
    """Test cohere_name helper function."""
    assert cohere_name('command-a-03-2025') == 'cohere/command-a-03-2025'
    assert cohere_name('embed-v4.0') == 'cohere/embed-v4.0'


def test_plugin_name() -> None:
    """Test plugin name constant."""
    assert COHERE_PLUGIN_NAME == 'cohere'


def test_plugin_initialization_with_api_key() -> None:
    """Test plugin initializes with API key parameter."""
    plugin = Cohere(api_key='test-key')
    assert plugin.name == 'cohere'


def test_plugin_initialization_from_env_cohere_api_key() -> None:
    """Test plugin reads API key from COHERE_API_KEY env var."""
    with patch.dict(os.environ, {'COHERE_API_KEY': 'env-key'}, clear=True):
        plugin = Cohere()
        assert plugin.name == 'cohere'


def test_plugin_initialization_from_env_co_api_key() -> None:
    """Test plugin reads API key from CO_API_KEY env var."""
    with patch.dict(os.environ, {'CO_API_KEY': 'co-env-key'}, clear=True):
        plugin = Cohere()
        assert plugin.name == 'cohere'


def test_plugin_initialization_without_api_key() -> None:
    """Test plugin raises error without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GenkitError, match='COHERE_API_KEY'):
            Cohere()


def test_plugin_initialization_with_models() -> None:
    """Test plugin accepts models parameter."""
    models = ['command-a-03-2025', 'command-r-plus']
    plugin = Cohere(api_key='test-key', models=models)
    assert plugin.name == 'cohere'


@pytest.mark.asyncio
async def test_plugin_init_returns_empty_list() -> None:
    """Test plugin init returns empty list for lazy loading."""
    plugin = Cohere(api_key='test-key')
    result = await plugin.init()
    assert result == []


@pytest.mark.asyncio
async def test_plugin_resolve_model_action() -> None:
    """Test plugin resolves model actions."""
    plugin = Cohere(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'cohere/command-a-03-2025')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'cohere/command-a-03-2025'


@pytest.mark.asyncio
async def test_plugin_resolve_embedder_action() -> None:
    """Test plugin resolves embedder actions."""
    plugin = Cohere(api_key='test-key')
    action = await plugin.resolve(ActionKind.EMBEDDER, 'cohere/embed-v4.0')

    assert action is not None
    assert action.kind == ActionKind.EMBEDDER
    assert action.name == 'cohere/embed-v4.0'


@pytest.mark.asyncio
async def test_plugin_resolve_non_model_returns_none() -> None:
    """Test plugin returns None for non-model action kinds."""
    plugin = Cohere(api_key='test-key')
    action = await plugin.resolve(ActionKind.PROMPT, 'some-prompt')
    assert action is None


@pytest.mark.asyncio
async def test_plugin_list_actions() -> None:
    """Test plugin lists supported models and embedders."""
    plugin = Cohere(api_key='test-key')
    actions = await plugin.list_actions()

    # All chat models + all embed models.
    expected_count = len(SUPPORTED_COHERE_MODELS) + len(SUPPORTED_EMBEDDING_MODELS)
    assert len(actions) == expected_count

    action_names = [action.name for action in actions]
    assert 'cohere/command-a-03-2025' in action_names
    assert 'cohere/embed-v4.0' in action_names


def test_supported_models_have_required_fields() -> None:
    """Test all supported models have required fields."""
    assert len(SUPPORTED_COHERE_MODELS) >= 5
    for name, info in SUPPORTED_COHERE_MODELS.items():
        assert info.label, f'Model {name} missing label'
        assert info.label.startswith('Cohere'), f'Model {name} label should start with "Cohere"'
        assert info.supports, f'Model {name} missing supports'


def test_supported_embedding_models_have_required_fields() -> None:
    """Test all supported embedding models have required metadata."""
    assert len(SUPPORTED_EMBEDDING_MODELS) >= 3
    for name, meta in SUPPORTED_EMBEDDING_MODELS.items():
        assert 'label' in meta, f'Embedding model {name} missing label'
        assert 'dimensions' in meta, f'Embedding model {name} missing dimensions'


def test_get_default_model_info() -> None:
    """Test default model info for unknown models."""
    info = get_default_model_info('custom-model')
    assert info.label == 'Cohere - custom-model'
    assert info.supports is not None
    assert info.supports.multiturn
