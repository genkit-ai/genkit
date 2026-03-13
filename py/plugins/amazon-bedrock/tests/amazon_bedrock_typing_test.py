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

"""Tests for Amazon Bedrock typing and config schemas."""

import pytest
from pydantic import ValidationError

from genkit.plugins.amazon_bedrock.typing import (
    AnthropicConfig,
    BedrockConfig,
    CohereConfig,
    CohereSafetyMode,
    CohereToolChoice,
    MetaLlamaConfig,
)


class TestStrEnums:
    """Tests for Bedrock StrEnum types."""

    def test_cohere_safety_mode_values(self) -> None:
        """Test Cohere safety mode values."""
        assert CohereSafetyMode.CONTEXTUAL == 'CONTEXTUAL'
        assert CohereSafetyMode.STRICT == 'STRICT'
        assert CohereSafetyMode.OFF == 'OFF'

    def test_cohere_tool_choice_values(self) -> None:
        """Test Cohere tool choice values."""
        assert CohereToolChoice.REQUIRED == 'REQUIRED'
        assert CohereToolChoice.NONE == 'NONE'


class TestBedrockConfig:
    """Tests for the base BedrockConfig."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = BedrockConfig()
        assert cfg.max_tokens is None
        assert cfg.temperature is None
        assert cfg.top_p is None

    def test_genkit_common_params(self) -> None:
        """Test Genkit common params."""
        cfg = BedrockConfig(temperature=0.5, max_output_tokens=1024, top_p=0.9)
        assert cfg.temperature == 0.5
        assert cfg.max_output_tokens == 1024
        assert cfg.top_p == 0.9

    def test_temperature_bounds(self) -> None:
        """Test Temperature bounds."""
        cfg = BedrockConfig(temperature=0.0)
        assert cfg.temperature == 0.0
        with pytest.raises(ValidationError):
            BedrockConfig(temperature=-0.1)

    def test_top_p_bounds(self) -> None:
        """Test Top p bounds."""
        cfg = BedrockConfig(top_p=1.0)
        assert cfg.top_p == 1.0
        with pytest.raises(ValidationError):
            BedrockConfig(top_p=1.1)

    def test_extra_fields_allowed(self) -> None:
        """Test Extra fields allowed."""
        cfg = BedrockConfig.model_validate({'custom_param': 42})
        assert cfg.model_extra is not None
        assert cfg.model_extra['custom_param'] == 42


class TestAnthropicConfig:
    """Tests for AnthropicConfig."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = AnthropicConfig()
        assert cfg.top_k is None
        assert cfg.thinking is None

    def test_top_k_valid(self) -> None:
        """Test Top k valid."""
        cfg = AnthropicConfig(top_k=40)
        assert cfg.top_k == 40

    def test_thinking_dict(self) -> None:
        """Test Thinking dict."""
        cfg = AnthropicConfig.model_validate({'thinking': {'type': 'enabled', 'budget_tokens': 1024}})
        assert cfg.thinking is not None
        assert cfg.thinking['type'] == 'enabled'

    def test_inherits_bedrock_config(self) -> None:
        """Test Inherits bedrock config."""
        cfg = AnthropicConfig(temperature=0.7, max_output_tokens=2048)
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 2048


class TestCohereConfig:
    """Tests for CohereConfig."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = CohereConfig()
        assert cfg.safety_mode is None
        assert cfg.tool_choice is None

    def test_safety_mode_enum(self) -> None:
        """Test Safety mode enum."""
        cfg = CohereConfig(safety_mode=CohereSafetyMode.STRICT)
        assert cfg.safety_mode == 'STRICT'

    def test_tool_choice_enum(self) -> None:
        """Test Tool choice enum."""
        cfg = CohereConfig(tool_choice=CohereToolChoice.REQUIRED)
        assert cfg.tool_choice == 'REQUIRED'


class TestMetaLlamaConfig:
    """Tests for MetaLlamaConfig."""

    def test_defaults(self) -> None:
        """Test Defaults."""
        cfg = MetaLlamaConfig()
        assert cfg.max_tokens is None

    def test_inherits_bedrock(self) -> None:
        """Test Inherits bedrock."""
        cfg = MetaLlamaConfig(temperature=0.3, top_p=0.8)
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.8


class TestCamelCaseAliases:
    """Tests for camelCase alias generation."""

    def test_max_output_tokens_alias(self) -> None:
        """Test Max output tokens alias."""
        cfg = BedrockConfig.model_validate({'maxOutputTokens': 512})
        assert cfg.max_output_tokens == 512

    def test_stop_sequences_alias(self) -> None:
        """Test Stop sequences alias."""
        cfg = BedrockConfig.model_validate({'stopSequences': ['END']})
        assert cfg.stop_sequences == ['END']

    def test_round_trip_by_alias(self) -> None:
        """Test Round trip by alias."""
        cfg = BedrockConfig(max_output_tokens=100, stop_sequences=['STOP'])
        data = cfg.model_dump(by_alias=True, exclude_none=True)
        assert 'maxOutputTokens' in data
        assert 'stopSequences' in data
