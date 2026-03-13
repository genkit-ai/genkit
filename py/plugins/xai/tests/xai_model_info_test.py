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

"""Tests for xAI model info registry and version enum."""

from genkit.plugins.xai.model_info import (
    SUPPORTED_XAI_MODELS,
    XAIGrokVersion,
    get_model_info,
)


class TestXAIGrokVersion:
    """Tests for the XAIGrokVersion StrEnum."""

    def test_all_enum_members_are_strings(self) -> None:
        """Test All enum members are strings."""
        for member in XAIGrokVersion:
            assert isinstance(member.value, str)

    def test_grok_3_value(self) -> None:
        """Test Grok 3 value."""
        assert XAIGrokVersion.GROK_3 == 'grok-3'

    def test_grok_4_value(self) -> None:
        """Test Grok 4 value."""
        assert XAIGrokVersion.GROK_4 == 'grok-4'

    def test_grok_vision_value(self) -> None:
        """Test Grok vision value."""
        assert XAIGrokVersion.GROK_2_VISION_1212 == 'grok-2-vision-1212'

    def test_reasoning_models_in_enum(self) -> None:
        """Test Reasoning models in enum."""
        assert XAIGrokVersion.GROK_4_FAST_REASONING == 'grok-4-fast-reasoning'
        assert XAIGrokVersion.GROK_4_1_FAST_REASONING == 'grok-4-1-fast-reasoning'

    def test_enum_count(self) -> None:
        """Test Enum count."""
        assert len(XAIGrokVersion) == 11


class TestSupportedXAIModels:
    """Tests for the SUPPORTED_XAI_MODELS registry."""

    def test_registry_has_all_enum_members(self) -> None:
        """Test Registry has all enum members."""
        for member in XAIGrokVersion:
            assert member in SUPPORTED_XAI_MODELS, f'{member} missing from registry'

    def test_registry_size_matches_enum(self) -> None:
        """Test Registry size matches enum."""
        assert len(SUPPORTED_XAI_MODELS) == len(XAIGrokVersion)

    def test_all_have_labels(self) -> None:
        """Test All have labels."""
        for model_id, info in SUPPORTED_XAI_MODELS.items():
            assert info.label, f'{model_id} missing label'
            assert info.label is not None
            assert 'xAI' in info.label

    def test_all_have_versions(self) -> None:
        """Test All have versions."""
        for model_id, info in SUPPORTED_XAI_MODELS.items():
            assert info.versions, f'{model_id} missing versions'

    def test_vision_model_has_media(self) -> None:
        """Test Vision model has media."""
        vision = SUPPORTED_XAI_MODELS[XAIGrokVersion.GROK_2_VISION_1212]
        assert vision.supports is not None
        assert vision.supports.media is True

    def test_language_model_no_media(self) -> None:
        """Test Language model no media."""
        grok3 = SUPPORTED_XAI_MODELS[XAIGrokVersion.GROK_3]
        assert grok3.supports is not None
        assert grok3.supports.media is False

    def test_reasoning_model_has_tools(self) -> None:
        """Test Reasoning model has tools."""
        grok4 = SUPPORTED_XAI_MODELS[XAIGrokVersion.GROK_4]
        assert grok4.supports is not None
        assert grok4.supports.tools is True

    def test_vision_model_no_multiturn(self) -> None:
        """Test Vision model no multiturn."""
        vision = SUPPORTED_XAI_MODELS[XAIGrokVersion.GROK_2_VISION_1212]
        assert vision.supports is not None
        assert vision.supports.multiturn is False


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_known_model(self) -> None:
        """Test Known model."""
        info = get_model_info('grok-3')
        assert info.label == 'xAI - Grok 3'

    def test_unknown_model_fallback(self) -> None:
        """Test Unknown model fallback."""
        info = get_model_info('grok-99-turbo')
        assert info.label is not None
        assert 'xAI' in info.label
        assert 'grok-99-turbo' in info.label

    def test_unknown_model_has_language_supports(self) -> None:
        """Test Unknown model has language supports."""
        info = get_model_info('unknown-model')
        assert info.supports is not None
        assert info.supports.multiturn is True
        assert info.supports.tools is True
        assert info.supports.media is False
