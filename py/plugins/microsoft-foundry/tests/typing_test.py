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

"""Tests for Microsoft Foundry typing module.

Tests cover configuration schemas, enum completeness, camelCase alias
generation, extra-field passthrough, and Pydantic validation constraints.
"""

import pytest
from pydantic import ValidationError

from genkit.plugins.microsoft_foundry.typing import (
    AI21JambaConfig,
    AnthropicConfig,
    AnthropicServiceTier,
    CohereConfig,
    CohereSafetyMode,
    CohereToolChoice,
    DeepSeekConfig,
    DeepSeekThinkingType,
    GrokConfig,
    LlamaConfig,
    MicrosoftFoundryConfig,
    MistralConfig,
    OpenAIConfig,
    PhiConfig,
    ReasoningEffort,
    TimeSeriesConfig,
    Verbosity,
    VisualDetailLevel,
)


# ---------------------------------------------------------------------------
# StrEnum Completeness
# ---------------------------------------------------------------------------
class TestEnumCompleteness:
    """Tests for EnumCompleteness."""

    def test_visual_detail_level(self) -> None:
        """Visual detail level."""
        assert set(VisualDetailLevel) == {
            VisualDetailLevel.AUTO,
            VisualDetailLevel.LOW,
            VisualDetailLevel.HIGH,
        }

    def test_reasoning_effort(self) -> None:
        """Reasoning effort."""
        values = {e.value for e in ReasoningEffort}
        assert values == {'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}

    def test_verbosity(self) -> None:
        """Verbosity."""
        assert set(Verbosity) == {Verbosity.LOW, Verbosity.MEDIUM, Verbosity.HIGH}

    def test_cohere_safety_mode(self) -> None:
        """Cohere safety mode."""
        assert set(CohereSafetyMode) == {
            CohereSafetyMode.CONTEXTUAL,
            CohereSafetyMode.STRICT,
            CohereSafetyMode.OFF,
        }

    def test_cohere_tool_choice(self) -> None:
        """Cohere tool choice."""
        assert set(CohereToolChoice) == {
            CohereToolChoice.REQUIRED,
            CohereToolChoice.NONE,
        }

    def test_deepseek_thinking_type(self) -> None:
        """Deepseek thinking type."""
        assert set(DeepSeekThinkingType) == {
            DeepSeekThinkingType.ENABLED,
            DeepSeekThinkingType.DISABLED,
        }

    def test_anthropic_service_tier(self) -> None:
        """Anthropic service tier."""
        assert set(AnthropicServiceTier) == {
            AnthropicServiceTier.STANDARD,
            AnthropicServiceTier.PRIORITY,
        }


# ---------------------------------------------------------------------------
# MicrosoftFoundryConfig — Base Config
# ---------------------------------------------------------------------------
class TestMicrosoftFoundryConfig:
    """Tests for MicrosoftFoundryConfig."""

    def test_defaults_are_none(self) -> None:
        """Defaults are none."""
        cfg = MicrosoftFoundryConfig()
        assert cfg.temperature is None
        assert cfg.max_output_tokens is None
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.stop_sequences is None
        assert cfg.version is None

    def test_temperature_validation(self) -> None:
        """Temperature validation."""
        cfg = MicrosoftFoundryConfig(temperature=0.5)
        assert cfg.temperature == 0.5

    def test_temperature_min_bound(self) -> None:
        """Temperature min bound."""
        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(temperature=-0.1)

    def test_temperature_max_bound(self) -> None:
        """Temperature max bound."""
        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(temperature=2.1)

    def test_top_p_validation(self) -> None:
        """Top p validation."""
        cfg = MicrosoftFoundryConfig(top_p=0.9)
        assert cfg.top_p == 0.9

    def test_top_p_out_of_range(self) -> None:
        """Top p out of range."""
        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(top_p=1.5)

    def test_frequency_penalty_bounds(self) -> None:
        """Frequency penalty bounds."""
        cfg = MicrosoftFoundryConfig(frequency_penalty=-2.0)
        assert cfg.frequency_penalty == -2.0

        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(frequency_penalty=2.5)

    def test_top_logprobs_bounds(self) -> None:
        """Top logprobs bounds."""
        MicrosoftFoundryConfig(top_logprobs=0)
        MicrosoftFoundryConfig(top_logprobs=20)
        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(top_logprobs=21)

    def test_n_minimum(self) -> None:
        """N minimum."""
        cfg = MicrosoftFoundryConfig(n=1)
        assert cfg.n == 1
        with pytest.raises(ValidationError):
            MicrosoftFoundryConfig(n=0)

    def test_camel_case_alias(self) -> None:
        """Verify camelCase aliases are generated for JSON interop."""
        cfg = MicrosoftFoundryConfig(max_output_tokens=100)
        dumped = cfg.model_dump(by_alias=True)
        assert 'maxOutputTokens' in dumped

    def test_extra_fields_allowed(self) -> None:
        """extra='allow' lets unknown params pass through to the backend."""
        cfg = MicrosoftFoundryConfig.model_validate({'my_custom_param': 'hello'})
        assert cfg.model_extra is not None
        assert cfg.model_extra['my_custom_param'] == 'hello'

    def test_openai_config_is_alias(self) -> None:
        """Openai config is alias."""
        assert OpenAIConfig is MicrosoftFoundryConfig


# ---------------------------------------------------------------------------
# Model-Specific Configs
# ---------------------------------------------------------------------------
class TestMistralConfig:
    """Tests for MistralConfig."""

    def test_mistral_specific_params(self) -> None:
        """Mistral specific params."""
        cfg = MistralConfig(random_seed=42, safe_prompt=True)
        assert cfg.random_seed == 42
        assert cfg.safe_prompt is True

    def test_inherits_base_params(self) -> None:
        """Inherits base params."""
        cfg = MistralConfig(temperature=0.7, random_seed=42)
        assert cfg.temperature == 0.7
        assert cfg.random_seed == 42


class TestLlamaConfig:
    """Tests for LlamaConfig."""

    def test_llama_specific_params(self) -> None:
        """Llama specific params."""
        cfg = LlamaConfig(
            max_new_tokens=256,
            repetition_penalty=1.2,
            do_sample=True,
        )
        assert cfg.max_new_tokens == 256
        assert cfg.repetition_penalty == 1.2
        assert cfg.do_sample is True

    def test_llama_tgi_params(self) -> None:
        """Llama tgi params."""
        cfg = LlamaConfig(truncate=True, return_full_text=False, watermark=True)
        assert cfg.truncate is True
        assert cfg.return_full_text is False
        assert cfg.watermark is True


class TestCohereConfig:
    """Tests for CohereConfig."""

    def test_cohere_sampling_params(self) -> None:
        """Cohere sampling params."""
        cfg = CohereConfig(k=50, p=0.8)
        assert cfg.k == 50
        assert cfg.p == 0.8

    def test_cohere_k_bounds(self) -> None:
        """Cohere k bounds."""
        with pytest.raises(ValidationError):
            CohereConfig(k=-1)
        with pytest.raises(ValidationError):
            CohereConfig(k=501)

    def test_cohere_p_bounds(self) -> None:
        """Cohere p bounds."""
        with pytest.raises(ValidationError):
            CohereConfig(p=0.001)  # Below 0.01
        with pytest.raises(ValidationError):
            CohereConfig(p=1.0)  # Above 0.99

    def test_cohere_safety_mode(self) -> None:
        """Cohere safety mode."""
        cfg = CohereConfig(safety_mode=CohereSafetyMode.STRICT)
        assert cfg.safety_mode == CohereSafetyMode.STRICT


class TestDeepSeekConfig:
    """Tests for DeepSeekConfig."""

    def test_deepseek_specific_params(self) -> None:
        """Deepseek specific params."""
        cfg = DeepSeekConfig(thinking={'type': 'enabled'}, prefix=True)
        assert cfg.thinking == {'type': 'enabled'}
        assert cfg.prefix is True


class TestAnthropicConfig:
    """Tests for AnthropicConfig."""

    def test_anthropic_specific_params(self) -> None:
        """Anthropic specific params."""
        cfg = AnthropicConfig(
            service_tier=AnthropicServiceTier.PRIORITY,
            metadata={'user_id': 'test'},
        )
        assert cfg.service_tier == AnthropicServiceTier.PRIORITY
        assert cfg.metadata == {'user_id': 'test'}


class TestTimeSeriesConfig:
    """Tests for TimeSeriesConfig."""

    def test_time_series_params(self) -> None:
        """Time series params."""
        cfg = TimeSeriesConfig(
            prediction_length=24,
            context_length=96,
            num_samples=100,
        )
        assert cfg.prediction_length == 24
        assert cfg.context_length == 96
        assert cfg.num_samples == 100


class TestPassthroughInheritance:
    """All subconfigs should inherit extra='allow' from base."""

    def test_phi_passthrough(self) -> None:
        """Phi passthrough."""
        cfg = PhiConfig.model_validate({'custom_param': 42})
        assert cfg.model_extra is not None
        assert cfg.model_extra['custom_param'] == 42

    def test_grok_passthrough(self) -> None:
        """Grok passthrough."""
        cfg = GrokConfig.model_validate({'any_future_param': 'value'})
        assert cfg.model_extra is not None
        assert cfg.model_extra['any_future_param'] == 'value'

    def test_jamba_passthrough(self) -> None:
        """Jamba passthrough."""
        cfg = AI21JambaConfig.model_validate({'special': 'yes'})
        assert cfg.model_extra is not None
        assert cfg.model_extra['special'] == 'yes'
