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

"""Tests for DeepSeek model information."""

from genkit.plugins.deepseek.model_info import (
    SUPPORTED_DEEPSEEK_MODELS,
    get_default_model_info,
    is_reasoning_model,
)


def test_supported_models_exist() -> None:
    """Test that supported models are defined."""
    assert 'deepseek-reasoner' in SUPPORTED_DEEPSEEK_MODELS
    assert 'deepseek-chat' in SUPPORTED_DEEPSEEK_MODELS
    assert 'deepseek-v3' in SUPPORTED_DEEPSEEK_MODELS
    assert 'deepseek-r1' in SUPPORTED_DEEPSEEK_MODELS


def test_model_order() -> None:
    """Test models are in correct order (matching JS)."""
    keys = list(SUPPORTED_DEEPSEEK_MODELS.keys())
    assert keys[0] == 'deepseek-reasoner'
    assert keys[1] == 'deepseek-chat'


def test_chat_model_info_structure() -> None:
    """Test chat model info has required fields and correct capabilities."""
    for model_name in ('deepseek-chat', 'deepseek-v3'):
        model_info = SUPPORTED_DEEPSEEK_MODELS[model_name]
        assert model_info.label
        assert model_info.supports is not None
        assert model_info.supports.multiturn is True
        assert model_info.supports.tools is True
        assert model_info.supports.media is False
        assert model_info.supports.system_role is True
        assert model_info.supports.output is not None
        assert 'text' in model_info.supports.output
        assert 'json' in model_info.supports.output


def test_reasoning_model_info_structure() -> None:
    """Test reasoning model info has correct capabilities (no tools, text only)."""
    for model_name in ('deepseek-reasoner', 'deepseek-r1'):
        model_info = SUPPORTED_DEEPSEEK_MODELS[model_name]
        assert model_info.label
        assert model_info.supports is not None
        assert model_info.supports.multiturn is True
        assert model_info.supports.tools is False
        assert model_info.supports.media is False
        assert model_info.supports.system_role is True
        assert model_info.supports.output is not None
        assert 'text' in model_info.supports.output
        assert 'json' not in model_info.supports.output


def test_is_reasoning_model() -> None:
    """Test reasoning model detection."""
    assert is_reasoning_model('deepseek-reasoner') is True
    assert is_reasoning_model('deepseek-r1') is True
    assert is_reasoning_model('deepseek-chat') is False
    assert is_reasoning_model('deepseek-v3') is False


def test_is_reasoning_model_with_prefix() -> None:
    """Test reasoning model detection with plugin prefix."""
    assert is_reasoning_model('deepseek/deepseek-reasoner') is True
    assert is_reasoning_model('deepseek/deepseek-r1') is True
    assert is_reasoning_model('deepseek/deepseek-chat') is False
    assert is_reasoning_model('deepseek/deepseek-v3') is False


def test_get_default_model_info_chat() -> None:
    """Test getting default info for unknown chat models."""
    info = get_default_model_info('deepseek-future-model')
    assert 'deepseek-future-model' in str(info.label)
    assert info.supports is not None
    assert info.supports.multiturn is True
    assert info.supports.tools is True


def test_get_default_model_info_reasoning() -> None:
    """Test getting default info for unknown reasoning models doesn't apply."""
    # Unknown model names are treated as chat models by default.
    info = get_default_model_info('deepseek-some-other')
    assert info.supports is not None
    assert info.supports.tools is True
