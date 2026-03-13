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

"""Tests for Hugging Face plugin."""

import os
from unittest.mock import MagicMock, patch

import pytest

from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.plugins.huggingface import (
    HUGGINGFACE_PLUGIN_NAME,
    POPULAR_HUGGINGFACE_MODELS,
    HuggingFace,
    huggingface_name,
)
from genkit.plugins.huggingface.model_info import get_default_model_info


def test_huggingface_name() -> None:
    """Test huggingface_name helper function."""
    assert huggingface_name('meta-llama/Llama-3.3-70B-Instruct') == 'huggingface/meta-llama/Llama-3.3-70B-Instruct'
    assert huggingface_name('google/gemma-2-27b-it') == 'huggingface/google/gemma-2-27b-it'


def test_plugin_name() -> None:
    """Test plugin name constant."""
    assert HUGGINGFACE_PLUGIN_NAME == 'huggingface'


def test_plugin_initialization_with_token() -> None:
    """Test plugin initializes with token parameter."""
    plugin = HuggingFace(token='test-token')
    assert plugin.name == 'huggingface'
    assert plugin.token == 'test-token'


def test_plugin_initialization_from_env() -> None:
    """Test plugin reads token from environment."""
    with patch.dict(os.environ, {'HF_TOKEN': 'env-token'}):
        plugin = HuggingFace()
        assert plugin.token == 'env-token'


def test_plugin_initialization_without_token() -> None:
    """Test plugin raises error without token."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(GenkitError) as exc_info:
            HuggingFace()
        assert 'HF_TOKEN' in str(exc_info.value)


def test_plugin_initialization_with_provider() -> None:
    """Test plugin accepts provider parameter."""
    plugin = HuggingFace(token='test-token', provider='groq')
    assert plugin.provider == 'groq'


def test_plugin_initialization_with_models() -> None:
    """Test plugin accepts models parameter."""
    models = ['meta-llama/Llama-3.3-70B-Instruct', 'google/gemma-2-27b-it']
    plugin = HuggingFace(token='test-token', models=models)
    assert plugin.models == models


@patch('genkit.plugins.huggingface.models.InferenceClient')
@pytest.mark.asyncio
async def test_plugin_init_returns_empty_list(mock_client: MagicMock) -> None:
    """Test plugin init returns empty list for lazy loading."""
    plugin = HuggingFace(token='test-token')
    result = await plugin.init()
    assert result == []


@patch('genkit.plugins.huggingface.models.InferenceClient')
@pytest.mark.asyncio
async def test_plugin_resolve_model_action(mock_client: MagicMock) -> None:
    """Test plugin resolves model actions."""
    plugin = HuggingFace(token='test-token')
    action = await plugin.resolve(ActionKind.MODEL, 'huggingface/meta-llama/Llama-3.3-70B-Instruct')

    assert action is not None
    assert action.kind == ActionKind.MODEL
    assert action.name == 'huggingface/meta-llama/Llama-3.3-70B-Instruct'


@patch('genkit.plugins.huggingface.models.InferenceClient')
@pytest.mark.asyncio
async def test_plugin_resolve_non_model_returns_none(mock_client: MagicMock) -> None:
    """Test plugin returns None for non-model action kinds."""
    plugin = HuggingFace(token='test-token')
    action = await plugin.resolve(ActionKind.PROMPT, 'some-prompt')
    assert action is None


@pytest.mark.asyncio
async def test_plugin_list_actions() -> None:
    """Test plugin lists popular models."""
    plugin = HuggingFace(token='test-token')
    actions = await plugin.list_actions()

    assert len(actions) == len(POPULAR_HUGGINGFACE_MODELS)
    action_names = [action.name for action in actions]
    assert 'huggingface/meta-llama/Llama-3.3-70B-Instruct' in action_names
    assert 'huggingface/google/gemma-2-27b-it' in action_names


def test_popular_models_have_required_fields() -> None:
    """Test all popular models have required fields."""
    assert len(POPULAR_HUGGINGFACE_MODELS) >= 10
    for name, info in POPULAR_HUGGINGFACE_MODELS.items():
        assert info.label, f'Model {name} missing label'
        assert info.supports, f'Model {name} missing supports'


def test_get_default_model_info() -> None:
    """Test default model info for unknown models."""
    info = get_default_model_info('custom-org/custom-model')
    assert info.label == 'Hugging Face - custom-model'
    assert info.supports is not None
    assert info.supports.multiturn is True


def test_get_default_model_info_simple_name() -> None:
    """Test default model info for simple model names."""
    info = get_default_model_info('simple-model')
    assert info.label == 'Hugging Face - simple-model'
