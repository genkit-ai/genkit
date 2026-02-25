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

"""Tests for Anthropic plugin."""

from unittest.mock import patch

import pytest

from genkit.core.registry import ActionKind
from genkit.plugins.anthropic import Anthropic, anthropic_name
from genkit.plugins.anthropic.model_info import (
    SUPPORTED_ANTHROPIC_MODELS as SUPPORTED_MODELS,
    get_model_info,
)
from genkit.types import (
    GenerateRequest,
    GenerationCommonConfig,
    Message,
    Part,
    Role,
    TextPart,
    ToolDefinition,
)


def test_anthropic_name() -> None:
    """Test anthropic_name helper function."""
    assert anthropic_name('claude-sonnet-4') == 'anthropic/claude-sonnet-4'


def test_init_with_api_key() -> None:
    """Test plugin initialization with API key."""
    plugin = Anthropic(api_key='test-key')
    assert plugin._anthropic_client.api_key == 'test-key'
    assert plugin.models == list(SUPPORTED_MODELS.keys())


def test_init_without_api_key_raises() -> None:
    """Test plugin initialization without API key uses default behavior."""
    with patch.dict('os.environ', {}, clear=True):
        # AsyncAnthropic allows initialization without API key
        # Error only occurs when making actual API calls
        plugin = Anthropic()
        assert plugin._anthropic_client is not None


def test_init_with_env_var() -> None:
    """Test plugin initialization with environment variable."""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'}):
        plugin = Anthropic()
        assert plugin._anthropic_client.api_key == 'env-key'


def test_custom_models() -> None:
    """Test plugin initialization with custom models."""
    plugin = Anthropic(api_key='test-key', models=['claude-sonnet-4'])
    assert plugin.models == ['claude-sonnet-4']


@pytest.mark.asyncio
async def test_plugin_init() -> None:
    """Test plugin init method."""
    plugin = Anthropic(api_key='test-key', models=['claude-sonnet-4'])

    # init() should return an empty list (using lazy loading)
    result = await plugin.init()
    assert result == []


@pytest.mark.asyncio
async def test_resolve_action_model() -> None:
    """Test resolve method for model."""
    plugin = Anthropic(api_key='test-key')

    # Test resolving with unprefixed name
    action = await plugin.resolve(ActionKind.MODEL, 'anthropic/claude-sonnet-4')

    assert action is not None
    assert action.name == 'anthropic/claude-sonnet-4'
    assert action.kind == ActionKind.MODEL


def test_supported_models() -> None:
    """Test that all supported models have proper metadata."""
    assert len(SUPPORTED_MODELS) == 10
    for _name, info in SUPPORTED_MODELS.items():
        assert info.label is not None
        assert info.label.startswith('Anthropic - ')
        assert info.versions is not None
        assert len(info.versions) > 0
        assert info.supports is not None
        assert info.supports.multiturn is True
        assert info.supports.tools is True
        if _name == 'claude-3-5-haiku':
            assert info.supports.media is False
        else:
            assert info.supports.media is True
        assert info.supports.system_role is True


def test_get_model_info_known() -> None:
    """Test get_model_info returns correct info for known model."""
    info = get_model_info('claude-sonnet-4')
    assert info.label == 'Anthropic - Claude Sonnet 4'
    assert info.supports is not None
    assert info.supports.multiturn is True
    assert info.supports.tools is True


def test_get_model_info_unknown() -> None:
    """Test get_model_info returns default info for unknown model."""
    info = get_model_info('unknown-model')
    assert info.label == 'Anthropic - unknown-model'
    assert info.supports is not None
    assert info.supports.multiturn is True
    assert info.supports.tools is True


def _create_sample_request() -> GenerateRequest:
    """Create a sample generation request for testing."""
    return GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello, how are you?'))],
            )
        ],
        config=GenerationCommonConfig(),
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather for a location',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string', 'description': 'Location name'}},
                    'required': ['location'],
                },
            )
        ],
    )
