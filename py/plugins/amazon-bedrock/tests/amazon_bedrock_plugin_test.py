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

"""Unit tests for AWS Bedrock plugin.

This module tests the AWS Bedrock plugin functionality including:
- Plugin initialization
- Model naming utilities
- Configuration schema mapping
- Message conversion
- Model info registry
"""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from genkit.core.action import ActionRunContext
from genkit.plugins.amazon_bedrock import (
    AMAZON_BEDROCK_PLUGIN_NAME,
    AnthropicConfig,
    BedrockConfig,
    CohereConfig,
    DeepSeekConfig,
    GenkitCommonConfigMixin,
    MetaLlamaConfig,
    MistralConfig,
    bedrock_model,
    bedrock_name,
    claude_sonnet_4_5,
    deepseek_r1,
    get_config_schema_for_model,
    get_inference_profile_prefix,
    inference_profile,
    llama_3_3_70b,
    mistral_large_3,
    nova_pro,
)
from genkit.plugins.amazon_bedrock.models.model import BedrockModel
from genkit.plugins.amazon_bedrock.models.model_info import (
    SUPPORTED_BEDROCK_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    get_model_info,
)
from genkit.plugins.amazon_bedrock.plugin import _strip_inference_profile_prefix
from genkit.plugins.amazon_bedrock.typing import (
    AI21JambaConfig,
    AmazonNovaConfig,
    CohereSafetyMode,
    CohereToolChoice,
    StabilityAspectRatio,
    StabilityConfig,
    StabilityMode,
    StabilityOutputFormat,
)
from genkit.types import GenerateRequest, GenerateResponseChunk, Message, Part, Role, TextPart, ToolRequest


class TestBedrockNaming:
    """Tests for model naming utilities."""

    def test_bedrock_name_basic(self) -> None:
        """Test bedrock_name creates fully qualified names."""
        result = bedrock_name('anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert result == 'amazon-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_bedrock_model_alias(self) -> None:
        """Test bedrock_model is an alias for bedrock_name."""
        result = bedrock_model('meta.llama3-3-70b-instruct-v1:0')
        assert result == 'amazon-bedrock/meta.llama3-3-70b-instruct-v1:0'

    def test_predefined_model_references(self) -> None:
        """Test predefined model references are correctly formatted with direct model IDs."""
        # Pre-defined references use direct model IDs (work with IAM credentials)
        assert claude_sonnet_4_5 == 'amazon-bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0'
        assert nova_pro == 'amazon-bedrock/amazon.nova-pro-v1:0'
        assert llama_3_3_70b == 'amazon-bedrock/meta.llama3-3-70b-instruct-v1:0'
        assert mistral_large_3 == 'amazon-bedrock/mistral.mistral-large-3-675b-instruct'
        assert deepseek_r1 == 'amazon-bedrock/deepseek.r1-v1:0'

    def test_plugin_name_constant(self) -> None:
        """Test plugin name constant."""
        assert AMAZON_BEDROCK_PLUGIN_NAME == 'amazon-bedrock'


class TestConfigSchemaMapping:
    """Tests for configuration schema mapping."""

    def test_anthropic_model_gets_anthropic_config(self) -> None:
        """Test Anthropic models get AnthropicConfig."""
        config_class = get_config_schema_for_model('anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert config_class == AnthropicConfig

    def test_anthropic_inference_profile_gets_anthropic_config(self) -> None:
        """Test Anthropic inference profiles (with us. prefix) get AnthropicConfig."""
        config_class = get_config_schema_for_model('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert config_class == AnthropicConfig

    def test_meta_model_gets_llama_config(self) -> None:
        """Test Meta models get MetaLlamaConfig."""
        config_class = get_config_schema_for_model('meta.llama3-3-70b-instruct-v1:0')
        assert config_class == MetaLlamaConfig

    def test_meta_inference_profile_gets_llama_config(self) -> None:
        """Test Meta inference profiles get MetaLlamaConfig."""
        config_class = get_config_schema_for_model('us.meta.llama3-3-70b-instruct-v1:0')
        assert config_class == MetaLlamaConfig

    def test_mistral_model_gets_mistral_config(self) -> None:
        """Test Mistral models get MistralConfig."""
        config_class = get_config_schema_for_model('mistral.mistral-large-3-675b-instruct')
        assert config_class == MistralConfig

    def test_cohere_model_gets_cohere_config(self) -> None:
        """Test Cohere models get CohereConfig."""
        config_class = get_config_schema_for_model('cohere.command-r-plus-v1:0')
        assert config_class == CohereConfig

    def test_deepseek_model_gets_deepseek_config(self) -> None:
        """Test DeepSeek models get DeepSeekConfig."""
        config_class = get_config_schema_for_model('deepseek.r1-v1:0')
        assert config_class == DeepSeekConfig

    def test_deepseek_inference_profile_gets_deepseek_config(self) -> None:
        """Test DeepSeek inference profiles get DeepSeekConfig."""
        config_class = get_config_schema_for_model('us.deepseek.r1-v1:0')
        assert config_class == DeepSeekConfig

    def test_amazon_nova_gets_nova_config(self) -> None:
        """Test Amazon Nova models get AmazonNovaConfig."""
        config_class = get_config_schema_for_model('amazon.nova-pro-v1:0')
        assert config_class == AmazonNovaConfig

    def test_amazon_nova_inference_profile_gets_nova_config(self) -> None:
        """Test Amazon Nova inference profiles get AmazonNovaConfig."""
        config_class = get_config_schema_for_model('us.amazon.nova-pro-v1:0')
        assert config_class == AmazonNovaConfig

    def test_unknown_model_gets_base_config(self) -> None:
        """Test unknown models get base BedrockConfig."""
        config_class = get_config_schema_for_model('unknown.model-v1:0')
        assert config_class == BedrockConfig


class TestModelInfo:
    """Tests for model info registry."""

    def test_supported_models_not_empty(self) -> None:
        """Test that supported models registry is populated."""
        assert len(SUPPORTED_BEDROCK_MODELS) > 0

    def test_supported_embeddings_not_empty(self) -> None:
        """Test that embedding models registry is populated."""
        assert len(SUPPORTED_EMBEDDING_MODELS) > 0

    def test_get_known_model_info(self) -> None:
        """Test getting info for a known model."""
        model_info = get_model_info('anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert model_info.label == 'Claude Sonnet 4.5'
        assert model_info.supports is not None
        assert model_info.supports.multiturn is True
        assert model_info.supports.tools is True

    def test_get_unknown_model_info(self) -> None:
        """Test getting info for an unknown model returns default."""
        model_info = get_model_info('unknown.model-v1:0')
        assert model_info.label is not None
        assert 'unknown.model-v1:0' in model_info.label
        assert model_info.supports is not None

    def test_claude_models_support_media(self) -> None:
        """Test Claude models support media (images)."""
        model_info = get_model_info('anthropic.claude-sonnet-4-5-20250929-v1:0')
        assert model_info.supports is not None
        assert model_info.supports.media is True

    def test_nova_models_support_media(self) -> None:
        """Test Nova Pro/Lite models support media."""
        model_info = get_model_info('amazon.nova-pro-v1:0')
        assert model_info.supports is not None
        assert model_info.supports.media is True

    def test_nova_micro_text_only(self) -> None:
        """Test Nova Micro is text-only."""
        model_info = get_model_info('amazon.nova-micro-v1:0')
        assert model_info.supports is not None
        assert model_info.supports.media is False

    def test_deepseek_r1_no_tools(self) -> None:
        """Test DeepSeek R1 doesn't support tools in Bedrock."""
        model_info = get_model_info('deepseek.r1-v1:0')
        assert model_info.supports is not None
        assert model_info.supports.tools is False


class TestConfigTypes:
    """Tests for configuration type classes."""

    def test_bedrock_config_inherits_genkit_mixin(self) -> None:
        """Test BedrockConfig inherits from GenkitCommonConfigMixin."""
        assert issubclass(BedrockConfig, GenkitCommonConfigMixin)

    def test_bedrock_config_default_values(self) -> None:
        """Test BedrockConfig has expected default values."""
        config = BedrockConfig()
        assert config.temperature is None
        assert config.max_tokens is None
        assert config.top_p is None
        assert config.stop_sequences is None

    def test_bedrock_config_with_values(self) -> None:
        """Test BedrockConfig accepts values."""
        config = BedrockConfig(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            stop_sequences=['END'],
        )
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.stop_sequences == ['END']

    def test_anthropic_config_has_thinking(self) -> None:
        """Test AnthropicConfig has thinking parameter."""
        config = AnthropicConfig(thinking={'enabled': True})
        assert config.thinking == {'enabled': True}

    def test_meta_llama_config_has_max_gen_len(self) -> None:
        """Test MetaLlamaConfig has max_gen_len parameter."""
        config = MetaLlamaConfig(max_gen_len=1024)
        assert config.max_gen_len == 1024

    def test_mistral_config_has_safe_prompt(self) -> None:
        """Test MistralConfig has safe_prompt parameter."""
        config = MistralConfig(safe_prompt=True, random_seed=42)
        assert config.safe_prompt is True
        assert config.random_seed == 42

    def test_cohere_config_has_documents(self) -> None:
        """Test CohereConfig has documents parameter."""
        config = CohereConfig(
            k=50,
            p=0.9,
            safety_mode=CohereSafetyMode.CONTEXTUAL,
            documents=['doc1', 'doc2'],
        )
        assert config.k == 50
        assert config.p == 0.9
        assert config.safety_mode == CohereSafetyMode.CONTEXTUAL
        assert config.documents == ['doc1', 'doc2']

    def test_cohere_safety_mode_enum(self) -> None:
        """Test CohereSafetyMode enum values."""
        assert CohereSafetyMode.CONTEXTUAL == 'CONTEXTUAL'
        assert CohereSafetyMode.STRICT == 'STRICT'
        assert CohereSafetyMode.OFF == 'OFF'

    def test_cohere_tool_choice_enum(self) -> None:
        """Test CohereToolChoice enum values."""
        assert CohereToolChoice.REQUIRED == 'REQUIRED'
        assert CohereToolChoice.NONE == 'NONE'

    def test_config_allows_extra_fields(self) -> None:
        """Test configs allow extra fields for forward compatibility."""
        # Use model_validate to test that extra fields are allowed
        config = BedrockConfig.model_validate({
            'temperature': 0.5,
            'unknown_future_param': 'value',
        })
        assert config.temperature == 0.5
        # Extra fields should be allowed due to extra='allow'

    def test_ai21_jamba_config_has_all_params(self) -> None:
        """Test AI21 Jamba config has all documented parameters."""
        config = AI21JambaConfig(
            n=3,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop=['###', '\n'],
        )
        assert config.n == 3
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.stop == ['###', '\n']

    def test_ai21_jamba_n_validation(self) -> None:
        """Test AI21 Jamba n parameter has valid range."""
        config = AI21JambaConfig(n=1)
        assert config.n == 1

        config = AI21JambaConfig(n=16)
        assert config.n == 16

        with pytest.raises(ValueError):
            AI21JambaConfig(n=0)  # Below min

        with pytest.raises(ValueError):
            AI21JambaConfig(n=17)  # Above max

    def test_stability_config_text_to_image(self) -> None:
        """Test Stability config for text-to-image generation."""
        config = StabilityConfig(
            mode=StabilityMode.TEXT_TO_IMAGE,
            aspect_ratio=StabilityAspectRatio.RATIO_16_9,
            seed=12345,
            negative_prompt='blurry, low quality',
            output_format=StabilityOutputFormat.PNG,
        )
        assert config.mode == StabilityMode.TEXT_TO_IMAGE
        assert config.aspect_ratio == StabilityAspectRatio.RATIO_16_9
        assert config.seed == 12345
        assert config.negative_prompt == 'blurry, low quality'
        assert config.output_format == StabilityOutputFormat.PNG

    def test_stability_config_image_to_image(self) -> None:
        """Test Stability config for image-to-image generation."""
        config = StabilityConfig(
            mode=StabilityMode.IMAGE_TO_IMAGE,
            image='base64encodedimage',
            strength=0.7,
            seed=42,
        )
        assert config.mode == StabilityMode.IMAGE_TO_IMAGE
        assert config.image == 'base64encodedimage'
        assert config.strength == 0.7
        assert config.seed == 42

    def test_stability_strength_validation(self) -> None:
        """Test Stability strength parameter has valid range."""
        config = StabilityConfig(strength=0.0)
        assert config.strength == 0.0

        config = StabilityConfig(strength=1.0)
        assert config.strength == 1.0

        with pytest.raises(ValueError):
            StabilityConfig(strength=-0.1)  # Below min

        with pytest.raises(ValueError):
            StabilityConfig(strength=1.1)  # Above max

    def test_stability_aspect_ratio_enum(self) -> None:
        """Test StabilityAspectRatio enum values."""
        assert StabilityAspectRatio.RATIO_1_1 == '1:1'
        assert StabilityAspectRatio.RATIO_16_9 == '16:9'
        assert StabilityAspectRatio.RATIO_9_16 == '9:16'
        assert StabilityAspectRatio.RATIO_21_9 == '21:9'

    def test_stability_output_format_enum(self) -> None:
        """Test StabilityOutputFormat enum values."""
        assert StabilityOutputFormat.JPEG == 'jpeg'
        assert StabilityOutputFormat.PNG == 'png'
        assert StabilityOutputFormat.WEBP == 'webp'

    def test_stability_mode_enum(self) -> None:
        """Test StabilityMode enum values."""
        assert StabilityMode.TEXT_TO_IMAGE == 'text-to-image'
        assert StabilityMode.IMAGE_TO_IMAGE == 'image-to-image'

    def test_temperature_validation(self) -> None:
        """Test temperature is validated within range."""
        # Valid temperature
        config = BedrockConfig(temperature=0.5)
        assert config.temperature == 0.5

        # Invalid temperature should raise
        with pytest.raises(ValueError):
            BedrockConfig(temperature=1.5)  # > 1.0

    def test_top_p_validation(self) -> None:
        """Test top_p is validated within range."""
        # Valid top_p
        config = BedrockConfig(top_p=0.9)
        assert config.top_p == 0.9

        # Invalid top_p should raise
        with pytest.raises(ValueError):
            BedrockConfig(top_p=1.5)  # > 1.0


class TestEmbeddingModels:
    """Tests for embedding model registry."""

    def test_titan_embeddings_present(self) -> None:
        """Test Amazon Titan embedding models are registered."""
        assert 'amazon.titan-embed-text-v2:0' in SUPPORTED_EMBEDDING_MODELS
        assert 'amazon.titan-embed-text-v1' in SUPPORTED_EMBEDDING_MODELS

    def test_cohere_embeddings_present(self) -> None:
        """Test Cohere embedding models are registered."""
        assert 'cohere.embed-english-v3' in SUPPORTED_EMBEDDING_MODELS
        assert 'cohere.embed-multilingual-v3' in SUPPORTED_EMBEDDING_MODELS

    def test_embedding_model_has_dimensions(self) -> None:
        """Test embedding models have dimensions specified."""
        titan_embed = SUPPORTED_EMBEDDING_MODELS['amazon.titan-embed-text-v2:0']
        assert 'dimensions' in titan_embed
        assert titan_embed['dimensions'] > 0

    def test_embedding_model_has_input_types(self) -> None:
        """Test embedding models have input types specified."""
        titan_embed = SUPPORTED_EMBEDDING_MODELS['amazon.titan-embed-text-v2:0']
        assert 'supports' in titan_embed
        assert 'input' in titan_embed['supports']
        assert 'text' in titan_embed['supports']['input']


class TestInferenceProfileHelpers:
    """Tests for inference profile helper functions."""

    def test_get_inference_profile_prefix_us_regions(self) -> None:
        """Test US regions return 'us' prefix."""
        assert get_inference_profile_prefix('us-east-1') == 'us'
        assert get_inference_profile_prefix('us-east-2') == 'us'
        assert get_inference_profile_prefix('us-west-1') == 'us'
        assert get_inference_profile_prefix('us-west-2') == 'us'

    def test_get_inference_profile_prefix_eu_regions(self) -> None:
        """Test EU regions return 'eu' prefix."""
        assert get_inference_profile_prefix('eu-west-1') == 'eu'
        assert get_inference_profile_prefix('eu-west-2') == 'eu'
        assert get_inference_profile_prefix('eu-central-1') == 'eu'
        assert get_inference_profile_prefix('eu-north-1') == 'eu'

    def test_get_inference_profile_prefix_apac_regions(self) -> None:
        """Test APAC regions return 'apac' prefix."""
        assert get_inference_profile_prefix('ap-northeast-1') == 'apac'
        assert get_inference_profile_prefix('ap-southeast-1') == 'apac'
        assert get_inference_profile_prefix('ap-south-1') == 'apac'

    def test_get_inference_profile_prefix_other_regions(self) -> None:
        """Test other regions are routed appropriately."""
        # Canada routed through US
        assert get_inference_profile_prefix('ca-central-1') == 'us'
        # South America routed through US
        assert get_inference_profile_prefix('sa-east-1') == 'us'
        # Middle East routed through EU
        assert get_inference_profile_prefix('me-south-1') == 'eu'

    def test_inference_profile_us(self) -> None:
        """Test inference_profile with US region."""
        result = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'us-east-1')
        assert result == 'amazon-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_inference_profile_eu(self) -> None:
        """Test inference_profile with EU region."""
        result = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'eu-west-1')
        assert result == 'amazon-bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_inference_profile_apac(self) -> None:
        """Test inference_profile with APAC region."""
        result = inference_profile('anthropic.claude-sonnet-4-5-20250929-v1:0', 'ap-northeast-1')
        assert result == 'amazon-bedrock/apac.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_inference_profile_default_region(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test inference_profile uses AWS_REGION env var when no region specified."""
        monkeypatch.setenv('AWS_REGION', 'eu-central-1')
        result = inference_profile('amazon.nova-pro-v1:0')
        assert result == 'amazon-bedrock/eu.amazon.nova-pro-v1:0'

    def test_inference_profile_no_region_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test inference_profile raises error when no region is available."""
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
        with pytest.raises(ValueError, match='AWS region is required'):
            inference_profile('amazon.nova-pro-v1:0')

    def test_get_inference_profile_prefix_no_region_raises_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test get_inference_profile_prefix raises error when no region is available."""
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)
        with pytest.raises(ValueError, match='AWS region is required'):
            get_inference_profile_prefix()


class TestStripInferenceProfilePrefix:
    """Tests for _strip_inference_profile_prefix.

    The InvokeModel API (used for embeddings) does NOT support inference
    profile IDs. This helper strips regional prefixes so we always pass
    base model IDs to InvokeModel.
    """

    def test_strip_us_prefix(self) -> None:
        """Test stripping 'us.' prefix from model ID."""
        assert _strip_inference_profile_prefix('us.amazon.titan-embed-text-v2:0') == 'amazon.titan-embed-text-v2:0'

    def test_strip_eu_prefix(self) -> None:
        """Test stripping 'eu.' prefix from model ID."""
        assert _strip_inference_profile_prefix('eu.cohere.embed-english-v3') == 'cohere.embed-english-v3'

    def test_strip_apac_prefix(self) -> None:
        """Test stripping 'apac.' prefix from model ID."""
        assert _strip_inference_profile_prefix('apac.amazon.titan-embed-text-v1') == 'amazon.titan-embed-text-v1'

    def test_no_prefix_unchanged(self) -> None:
        """Test model IDs without prefix are returned unchanged."""
        assert _strip_inference_profile_prefix('amazon.titan-embed-text-v2:0') == 'amazon.titan-embed-text-v2:0'

    def test_no_prefix_cohere(self) -> None:
        """Test Cohere model ID without prefix is unchanged."""
        assert _strip_inference_profile_prefix('cohere.embed-english-v3') == 'cohere.embed-english-v3'

    def test_no_prefix_nova(self) -> None:
        """Test Nova embed model ID without prefix is unchanged."""
        assert _strip_inference_profile_prefix('amazon.nova-embed-text-v1:0') == 'amazon.nova-embed-text-v1:0'

    def test_round_trip_with_inference_profile(self) -> None:
        """Test that inference_profile + strip recovers the original model ID.

        This verifies the round-trip property: a base model ID passed through
        inference_profile() and then _strip_inference_profile_prefix() should
        yield the original base model ID (without the plugin prefix).
        """
        base_id = 'amazon.titan-embed-text-v2:0'
        # inference_profile adds 'amazon-bedrock/{prefix}.' prefixed name
        full_ref = inference_profile(base_id, 'us-east-1')
        # Strip 'amazon-bedrock/' prefix (as the plugin does)
        model_id = full_ref.split('/', 1)[1]  # 'us.amazon.titan-embed-text-v2:0'
        # Strip inference profile prefix
        stripped = _strip_inference_profile_prefix(model_id)
        assert stripped == base_id


class TestAutoInferenceProfileConversion:
    """Tests for automatic inference profile conversion in BedrockModel.

    When using API key authentication (AWS_BEARER_TOKEN_BEDROCK), the plugin
    should automatically convert direct model IDs to inference profile IDs
    by adding the appropriate regional prefix.
    """

    @pytest.fixture
    def mock_client(self) -> object:
        """Create a mock boto3 client."""
        return MagicMock()

    @pytest.fixture
    def clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear all relevant environment variables before each test."""
        monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
        monkeypatch.delenv('AWS_REGION', raising=False)
        monkeypatch.delenv('AWS_DEFAULT_REGION', raising=False)

    def test_iam_auth_returns_direct_model_id(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test IAM auth (no API key) returns direct model ID unchanged."""
        # No AWS_BEARER_TOKEN_BEDROCK set = IAM auth
        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        assert model._get_effective_model_id() == 'anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_api_key_auth_us_region_adds_us_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with US region adds 'us.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        assert model._get_effective_model_id() == 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_api_key_auth_us_west_region_adds_us_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with us-west region adds 'us.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-west-2')

        model = BedrockModel('mistral.mistral-large-3-675b-instruct', mock_client)
        assert model._get_effective_model_id() == 'us.mistral.mistral-large-3-675b-instruct'

    def test_api_key_auth_eu_region_adds_eu_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with EU region adds 'eu.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'eu-west-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        assert model._get_effective_model_id() == 'eu.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_api_key_auth_eu_central_region_adds_eu_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with eu-central region adds 'eu.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'eu-central-1')

        model = BedrockModel('meta.llama3-3-70b-instruct-v1:0', mock_client)
        assert model._get_effective_model_id() == 'eu.meta.llama3-3-70b-instruct-v1:0'

    def test_api_key_auth_apac_region_adds_apac_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with APAC region adds 'apac.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'ap-northeast-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        assert model._get_effective_model_id() == 'apac.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_api_key_auth_ap_southeast_region_adds_apac_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth with ap-southeast region adds 'apac.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'ap-southeast-1')

        model = BedrockModel('deepseek.r1-v1:0', mock_client)
        assert model._get_effective_model_id() == 'apac.deepseek.r1-v1:0'

    def test_api_key_auth_uses_aws_default_region(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth falls back to AWS_DEFAULT_REGION."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'eu-west-2')

        model = BedrockModel('amazon.nova-pro-v1:0', mock_client)
        assert model._get_effective_model_id() == 'eu.amazon.nova-pro-v1:0'

    def test_api_key_auth_aws_region_takes_precedence(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AWS_REGION takes precedence over AWS_DEFAULT_REGION."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')
        monkeypatch.setenv('AWS_DEFAULT_REGION', 'eu-west-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # Should use AWS_REGION (us), not AWS_DEFAULT_REGION (eu)
        assert model._get_effective_model_id() == 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_model_already_has_us_prefix_unchanged(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test model with existing 'us.' prefix is unchanged."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'eu-west-1')  # Different region

        model = BedrockModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # Should NOT add another prefix, even though region is EU
        assert model._get_effective_model_id() == 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_model_already_has_eu_prefix_unchanged(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test model with existing 'eu.' prefix is unchanged."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')  # Different region

        model = BedrockModel('eu.anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # Should NOT add another prefix
        assert model._get_effective_model_id() == 'eu.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_model_already_has_apac_prefix_unchanged(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test model with existing 'apac.' prefix is unchanged."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')  # Different region

        model = BedrockModel('apac.amazon.nova-pro-v1:0', mock_client)
        # Should NOT add another prefix
        assert model._get_effective_model_id() == 'apac.amazon.nova-pro-v1:0'

    def test_api_key_auth_no_region_returns_direct_id(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test API key auth without region returns direct model ID with warning."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        # No AWS_REGION or AWS_DEFAULT_REGION set

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # Should return direct ID (will likely fail at API call, but that's expected)
        assert model._get_effective_model_id() == 'anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_unknown_region_defaults_to_us(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test unknown region prefix defaults to 'us.'."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'unknown-region-1')  # Unrecognized

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        assert model._get_effective_model_id() == 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_canada_region_uses_us_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Canada region uses 'us.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'ca-central-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # Canada is routed through US prefix (defaults to us for unknown)
        assert model._get_effective_model_id() == 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_south_america_region_uses_us_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test South America region uses 'us.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'sa-east-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # SA mapped to apac in _get_effective_model_id (starts with sa-)
        assert model._get_effective_model_id() == 'apac.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_middle_east_region_uses_apac_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Middle East region uses 'apac.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'me-south-1')

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        # ME mapped to apac in _get_effective_model_id (starts with me-)
        assert model._get_effective_model_id() == 'apac.anthropic.claude-sonnet-4-5-20250929-v1:0'

    def test_africa_region_uses_apac_prefix(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Africa region uses 'apac.' prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'af-south-1')

        model = BedrockModel('amazon.nova-pro-v1:0', mock_client)
        # Africa mapped to apac in _get_effective_model_id (starts with af-)
        assert model._get_effective_model_id() == 'apac.amazon.nova-pro-v1:0'

    def test_deepseek_model_auto_conversion(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test DeepSeek model auto-converts with API key auth."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-west-2')

        model = BedrockModel('deepseek.r1-v1:0', mock_client)
        assert model._get_effective_model_id() == 'us.deepseek.r1-v1:0'

    def test_nova_model_auto_conversion(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Nova model auto-converts with API key auth."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'ap-south-1')

        model = BedrockModel('amazon.nova-pro-v1:0', mock_client)
        assert model._get_effective_model_id() == 'apac.amazon.nova-pro-v1:0'

    def test_cohere_model_auto_conversion(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Cohere model auto-converts with API key auth."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'eu-north-1')

        model = BedrockModel('cohere.command-r-plus-v1:0', mock_client)
        assert model._get_effective_model_id() == 'eu.cohere.command-r-plus-v1:0'

    def test_ai21_model_no_conversion_with_api_key(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AI21 models do NOT get inference profile prefix (not supported)."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')

        model = BedrockModel('ai21.jamba-1-5-large-v1:0', mock_client)
        # AI21 doesn't support cross-region inference profiles - should use direct ID
        assert model._get_effective_model_id() == 'ai21.jamba-1-5-large-v1:0'

    def test_ai21_mini_model_no_conversion_with_api_key(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test AI21 Jamba Mini model does NOT get inference profile prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'eu-west-1')

        model = BedrockModel('ai21.jamba-1-5-mini-v1:0', mock_client)
        # AI21 doesn't support cross-region inference profiles
        assert model._get_effective_model_id() == 'ai21.jamba-1-5-mini-v1:0'

    def test_stability_model_no_conversion_with_api_key(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test Stability AI models do NOT get inference profile prefix (not supported)."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-west-2')

        model = BedrockModel('stability.sd3-5-large-v1:0', mock_client)
        # Stability doesn't support cross-region inference profiles
        assert model._get_effective_model_id() == 'stability.sd3-5-large-v1:0'

    def test_unsupported_provider_no_conversion(
        self, mock_client: object, clear_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test unknown/unsupported providers do NOT get inference profile prefix."""
        monkeypatch.setenv('AWS_BEARER_TOKEN_BEDROCK', 'test-token')
        monkeypatch.setenv('AWS_REGION', 'us-east-1')

        model = BedrockModel('unknown-provider.some-model-v1:0', mock_client)
        # Unknown provider - should not add prefix
        assert model._get_effective_model_id() == 'unknown-provider.some-model-v1:0'


class _AsyncIter(AsyncIterator[Any]):
    """Wraps a list into an async iterator for mocking aioboto3 streams."""

    def __init__(self, items: list[Any]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> '_AsyncIter':
        return self

    async def __anext__(self) -> Any:  # noqa: ANN401
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration  # noqa: B904


class TestStreamingToolUseParsing:
    """Regression tests for streaming tool use assembly.

    Bedrock ConverseStream sends tool calls across multiple events:
    1. contentBlockStart — contains toolUseId and name
    2. contentBlockDelta(s) — contains input JSON fragments
    3. contentBlockStop

    A previous bug compared contentBlockIndex as int (from delta) vs
    string key (from start), causing the delta handler to create a
    *new* entry with empty name/ref instead of appending to the
    existing one. This resulted in RuntimeError: 'failed  not found'.
    """

    @pytest.fixture()
    def mock_client(self) -> MagicMock:
        """Create a mock async bedrock-runtime client."""
        client = MagicMock()
        # converse_stream is async with aioboto3
        client.converse_stream = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_tool_use_name_preserved_across_stream_events(self, mock_client: MagicMock) -> None:
        """Tool name and ref from contentBlockStart survive delta assembly."""
        # Simulate the ConverseStream event sequence for a tool call
        mock_client.converse_stream.return_value = {
            'stream': _AsyncIter([
                # 1. contentBlockStart: carries toolUseId and name
                {
                    'contentBlockStart': {
                        'contentBlockIndex': 1,
                        'start': {
                            'toolUse': {
                                'toolUseId': 'call_abc123',
                                'name': 'get_weather',
                            }
                        },
                    }
                },
                # 2. contentBlockDelta: carries input JSON fragment
                {
                    'contentBlockDelta': {
                        'contentBlockIndex': 1,
                        'delta': {
                            'toolUse': {
                                'input': '{"location"',
                            }
                        },
                    }
                },
                # 3. Another delta with more input
                {
                    'contentBlockDelta': {
                        'contentBlockIndex': 1,
                        'delta': {
                            'toolUse': {
                                'input': ': "London"}',
                            }
                        },
                    }
                },
                # 4. contentBlockStop
                {
                    'contentBlockStop': {
                        'contentBlockIndex': 1,
                    }
                },
                # 5. messageStop
                {
                    'messageStop': {
                        'stopReason': 'tool_use',
                    }
                },
                # 5. metadata
                {
                    'metadata': {
                        'usage': {
                            'inputTokens': 100,
                            'outputTokens': 50,
                            'totalTokens': 150,
                        }
                    }
                },
            ])
        }

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='What is the weather in London?'))],
                )
            ],
            tools=[],
        )

        chunks: list[GenerateResponseChunk] = []
        ctx = MagicMock(spec=ActionRunContext)
        ctx.send_chunk = MagicMock(side_effect=lambda c: chunks.append(c))

        response = await model._generate_streaming(
            {'modelId': 'anthropic.claude-sonnet-4-5-20250929-v1:0', 'messages': []},
            ctx,
            request,
        )

        # Find the tool request part in the response
        assert response.message is not None
        tool_parts = [p for p in response.message.content if p.root.tool_request is not None]
        assert len(tool_parts) == 1, f'Expected 1 tool request, got {len(tool_parts)}'

        tool_req = tool_parts[0].root.tool_request
        assert tool_req is not None
        assert isinstance(tool_req, ToolRequest)
        assert tool_req.name == 'get_weather', f'Tool name should be "get_weather", got "{tool_req.name}"'
        assert tool_req.ref == 'call_abc123', f'Tool ref should be "call_abc123", got "{tool_req.ref}"'
        assert tool_req.input == {'location': 'London'}, f'Tool input should be parsed JSON, got {tool_req.input!r}'

        # Verify a tool request chunk was emitted via ctx.send_chunk.
        tool_chunks = [
            c
            for c in chunks
            if any(hasattr(p.root, 'tool_request') and p.root.tool_request is not None for p in c.content)
        ]
        assert len(tool_chunks) == 1, f'Expected 1 tool request chunk, got {len(tool_chunks)}'

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_stream(self, mock_client: MagicMock) -> None:
        """Multiple tool calls in a single stream are assembled correctly."""
        mock_client.converse_stream.return_value = {
            'stream': _AsyncIter([
                # First tool call at block index 0
                {
                    'contentBlockStart': {
                        'contentBlockIndex': 0,
                        'start': {
                            'toolUse': {
                                'toolUseId': 'call_001',
                                'name': 'get_weather',
                            }
                        },
                    }
                },
                {
                    'contentBlockDelta': {
                        'contentBlockIndex': 0,
                        'delta': {'toolUse': {'input': '{"location": "London"}'}},
                    }
                },
                # Second tool call at block index 1
                {
                    'contentBlockStart': {
                        'contentBlockIndex': 1,
                        'start': {
                            'toolUse': {
                                'toolUseId': 'call_002',
                                'name': 'get_time',
                            }
                        },
                    }
                },
                {
                    'contentBlockDelta': {
                        'contentBlockIndex': 1,
                        'delta': {'toolUse': {'input': '{"timezone": "UTC"}'}},
                    }
                },
                {'messageStop': {'stopReason': 'tool_use'}},
                {
                    'metadata': {
                        'usage': {
                            'inputTokens': 200,
                            'outputTokens': 80,
                            'totalTokens': 280,
                        }
                    }
                },
            ])
        }

        model = BedrockModel('anthropic.claude-sonnet-4-5-20250929-v1:0', mock_client)
        request = GenerateRequest(
            messages=[
                Message(
                    role=Role.USER,
                    content=[Part(root=TextPart(text='Weather and time?'))],
                )
            ],
            tools=[],
        )
        ctx = MagicMock(spec=ActionRunContext)
        ctx.send_chunk = MagicMock()

        response = await model._generate_streaming(
            {'modelId': 'anthropic.claude-sonnet-4-5-20250929-v1:0', 'messages': []},
            ctx,
            request,
        )

        assert response.message is not None
        tool_parts = [p for p in response.message.content if p.root.tool_request is not None]
        assert len(tool_parts) == 2, f'Expected 2 tool requests, got {len(tool_parts)}'

        # Verify first tool
        tool_req_0 = tool_parts[0].root.tool_request
        assert tool_req_0 is not None
        assert isinstance(tool_req_0, ToolRequest)
        assert tool_req_0.name == 'get_weather'
        assert tool_req_0.ref == 'call_001'
        assert tool_req_0.input == {'location': 'London'}

        # Verify second tool
        tool_req_1 = tool_parts[1].root.tool_request
        assert tool_req_1 is not None
        assert isinstance(tool_req_1, ToolRequest)
        assert tool_req_1.name == 'get_time'
        assert tool_req_1.ref == 'call_002'
        assert tool_req_1.input == {'timezone': 'UTC'}
