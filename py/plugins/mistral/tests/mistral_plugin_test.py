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

"""Tests for Mistral AI plugin."""

import os
from unittest.mock import MagicMock, patch

import pytest

from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.plugins.mistral import (
    DEFAULT_MISTRAL_API_URL,
    MISTRAL_PLUGIN_NAME,
    SUPPORTED_MISTRAL_MODELS,
    Mistral,
    mistral_name,
)
from genkit.plugins.mistral.embeddings import SUPPORTED_EMBEDDING_MODELS
from genkit.plugins.mistral.model_info import get_default_model_info


def test_mistral_name() -> None:
    """Test mistral_name helper function."""
    assert mistral_name('mistral-large-latest') == 'mistral/mistral-large-latest'
    assert mistral_name('codestral-latest') == 'mistral/codestral-latest'


def test_plugin_name() -> None:
    """Test plugin name constant."""
    assert MISTRAL_PLUGIN_NAME == 'mistral'


def test_default_api_url() -> None:
    """Test default API URL constant."""
    assert DEFAULT_MISTRAL_API_URL == 'https://api.mistral.ai'


def test_plugin_initialization_with_api_key() -> None:
    """Test plugin initializes with API key parameter."""
    plugin = Mistral(api_key='test-key')
    assert plugin.name == 'mistral'
    assert plugin.api_key == 'test-key'


def test_plugin_initialization_from_env() -> None:
    """Test plugin reads API key from environment."""
    with patch.dict(os.environ, {'MISTRAL_API_KEY': 'env-key'}):
        plugin = Mistral()
        assert plugin.api_key == 'env-key'


def test_plugin_initialization_without_api_key() -> None:
    """Test plugin raises error without API key."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GenkitError) as exc_info:
            Mistral()
        assert 'MISTRAL_API_KEY' in str(exc_info.value)


def test_plugin_initialization_with_models() -> None:
    """Test plugin accepts models parameter."""
    models = ['mistral-large-latest', 'mistral-small-latest']
    plugin = Mistral(api_key='test-key', models=models)
    assert plugin.models == models


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_init_returns_empty_list(mock_client: MagicMock) -> None:
    """Test plugin init returns empty list for lazy loading."""
    plugin = Mistral(api_key='test-key')
    result = await plugin.init()
    assert result == []


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_resolve_model_action(mock_client: MagicMock) -> None:
    """Test plugin resolves model actions."""
    plugin = Mistral(api_key='test-key')
    action = await plugin.resolve(ActionKind.MODEL, 'mistral/mistral-large-latest')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'mistral/mistral-large-latest'


@patch('genkit.plugins.mistral.models.MistralClient')
@pytest.mark.asyncio
async def test_plugin_resolve_non_model_returns_none(mock_client: MagicMock) -> None:
    """Test plugin returns None for non-model action kinds."""
    plugin = Mistral(api_key='test-key')
    action = await plugin.resolve(ActionKind.PROMPT, 'some-prompt')
    assert action is None


@pytest.mark.asyncio
async def test_plugin_list_actions() -> None:
    """Test plugin lists supported models and embedders."""
    plugin = Mistral(api_key='test-key')
    actions = await plugin.list_actions()

    # Total = (chat models - embedding models) + embedding models.
    expected_count = len(SUPPORTED_MISTRAL_MODELS) - len(SUPPORTED_EMBEDDING_MODELS) + len(SUPPORTED_EMBEDDING_MODELS)
    assert len(actions) == expected_count
    action_names = [action.name for action in actions]
    assert 'mistral/mistral-large-latest' in action_names
    assert 'mistral/mistral-small-latest' in action_names
    assert 'mistral/mistral-embed' in action_names


def test_supported_models_have_required_fields() -> None:
    """Test all supported models have required fields."""
    assert len(SUPPORTED_MISTRAL_MODELS) >= 6
    for name, info in SUPPORTED_MISTRAL_MODELS.items():
        assert info.label, f'Model {name} missing label'
        assert info.label.startswith('Mistral AI - '), f'Model {name} label should start with "Mistral AI - "'
        assert info.supports, f'Model {name} missing supports'
        assert info.versions, f'Model {name} missing versions'


def test_get_default_model_info() -> None:
    """Test default model info for unknown models."""
    info = get_default_model_info('custom-model')
    assert info.label == 'Mistral AI - custom-model'
    assert info.supports is not None
    assert info.supports.multiturn is True
