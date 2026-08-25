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

"""Tests for the Anthropic.claude_model typed ref constructor."""

from typing import get_args, get_type_hints

import pytest
from genkit_anthropic import Anthropic, AnthropicConfig, KnownClaude
from genkit_anthropic.model_info import SUPPORTED_ANTHROPIC_MODELS

from genkit import GenkitError
from genkit.model import ModelRef


def test_claude_model_prefixes_anthropic() -> None:
    """claude_model namespaces the name and binds AnthropicConfig."""
    ref = Anthropic.claude_model('claude-sonnet-4-5')

    assert isinstance(ref, ModelRef)
    assert ref.name == 'anthropic/claude-sonnet-4-5'
    assert ref.config_schema is AnthropicConfig


def test_claude_model_strips_own_prefix() -> None:
    """An already-namespaced name is not double-prefixed."""
    ref = Anthropic.claude_model('anthropic/claude-sonnet-4-5')
    assert ref.name == 'anthropic/claude-sonnet-4-5'


def test_claude_model_keeps_foreign_prefix() -> None:
    """A Vertex Model Garden paste is a different name, not remapped onto Anthropic."""
    assert Anthropic.claude_model('vertexai/claude-sonnet-4-5').name == 'anthropic/vertexai/claude-sonnet-4-5'


def test_claude_model_carries_config() -> None:
    """A default config passed at construction survives into the ref."""
    ref = Anthropic.claude_model('claude-sonnet-4-5', config=AnthropicConfig(max_output_tokens=256))
    assert ref.config is not None
    assert ref.config.max_output_tokens == 256


def test_claude_model_allows_unknown_ids() -> None:
    """A brand-new Claude release works before this plugin learns its name."""
    assert Anthropic.claude_model('claude-next-99').name == 'anthropic/claude-next-99'


def test_claude_model_requires_a_name() -> None:
    """A bare prefix with no id left is an invalid argument."""
    with pytest.raises(GenkitError) as exc_info:
        Anthropic.claude_model('anthropic/')
    assert exc_info.value.status == 'INVALID_ARGUMENT'


def test_claude_model_rejects_non_string_name() -> None:
    """A non-string must not become a name via str() (None → 'None')."""
    with pytest.raises(GenkitError) as exc_info:
        Anthropic.claude_model(None)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'must be a string' in str(exc_info.value)

    with pytest.raises(GenkitError) as exc_info:
        Anthropic.claude_model(123)  # type: ignore[arg-type]
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'must be a string' in str(exc_info.value)


def test_known_claude_matches_catalog() -> None:
    """Quote autocomplete and the catalog are the same set of ids."""
    assert set(get_args(KnownClaude)) == set(SUPPORTED_ANTHROPIC_MODELS)


def test_create_model_action_types_anthropic_config() -> None:
    """Anthropic model actions opt into ModelRequest[AnthropicConfig]."""
    plugin = Anthropic(api_key='test-key', models=['claude-sonnet-4'])
    action = plugin._create_model_action('anthropic/claude-sonnet-4')

    hints = get_type_hints(action._fn)  # noqa: SLF001
    request_type = hints['request']
    args = get_args(request_type) or (getattr(request_type, '__pydantic_generic_metadata__', {}) or {}).get('args')
    assert args and args[0] is AnthropicConfig
