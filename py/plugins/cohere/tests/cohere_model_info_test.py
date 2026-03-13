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

"""Tests for Cohere model metadata and configuration."""

from genkit.plugins.cohere.model_info import (
    SUPPORTED_COHERE_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    ModelInfo,
    ModelSupports,
    get_default_model_info,
)


class TestSupportedCohereModels:
    """Tests for SUPPORTED_COHERE_MODELS registry."""

    def test_contains_command_a(self) -> None:
        """Test contains command a."""
        assert 'command-a-03-2025' in SUPPORTED_COHERE_MODELS

    def test_contains_command_a_reasoning(self) -> None:
        """Test contains command a reasoning."""
        assert 'command-a-reasoning-08-2025' in SUPPORTED_COHERE_MODELS

    def test_contains_command_a_translate(self) -> None:
        """Test contains command a translate."""
        assert 'command-a-translate-08-2025' in SUPPORTED_COHERE_MODELS

    def test_contains_command_a_vision(self) -> None:
        """Test contains command a vision."""
        assert 'command-a-vision-07-2025' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r7b(self) -> None:
        """Test contains command r7b."""
        assert 'command-r7b-12-2024' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r_plus(self) -> None:
        """Test contains command r plus."""
        assert 'command-r-plus' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r_plus_08_2024(self) -> None:
        """Test contains command r plus 08 2024."""
        assert 'command-r-plus-08-2024' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r_plus_04_2024(self) -> None:
        """Test contains command r plus 04 2024."""
        assert 'command-r-plus-04-2024' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r(self) -> None:
        """Test contains command r."""
        assert 'command-r' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r_08_2024(self) -> None:
        """Test contains command r 08 2024."""
        assert 'command-r-08-2024' in SUPPORTED_COHERE_MODELS

    def test_contains_command_r_03_2024(self) -> None:
        """Test contains command r 03 2024."""
        assert 'command-r-03-2024' in SUPPORTED_COHERE_MODELS

    def test_deprecated_command_not_present(self) -> None:
        """Test deprecated command not present."""
        assert 'command' not in SUPPORTED_COHERE_MODELS

    def test_deprecated_command_light_not_present(self) -> None:
        """Test deprecated command light not present."""
        assert 'command-light' not in SUPPORTED_COHERE_MODELS

    def test_total_model_count(self) -> None:
        """Test total model count."""
        assert len(SUPPORTED_COHERE_MODELS) == 11

    def test_all_models_have_labels(self) -> None:
        """Test all models have labels."""
        for name, info in SUPPORTED_COHERE_MODELS.items():
            assert info.label, f'Model {name} missing label'
            assert info.label.startswith('Cohere'), f'Model {name} label should start with "Cohere"'

    def test_all_models_have_supports(self) -> None:
        """Test all models have supports."""
        for name, info in SUPPORTED_COHERE_MODELS.items():
            assert info.supports is not None, f'Model {name} missing supports'

    def test_vision_model_supports_media(self) -> None:
        """Test vision model supports media."""
        vision = SUPPORTED_COHERE_MODELS['command-a-vision-07-2025']
        assert vision.supports.media is True

    def test_vision_model_no_tools(self) -> None:
        """Test vision model no tools."""
        vision = SUPPORTED_COHERE_MODELS['command-a-vision-07-2025']
        assert vision.supports.tools is False

    def test_command_a_supports_tools(self) -> None:
        """Test command a supports tools."""
        model = SUPPORTED_COHERE_MODELS['command-a-03-2025']
        assert model.supports.tools is True

    def test_command_a_no_media(self) -> None:
        """Test command a no media."""
        model = SUPPORTED_COHERE_MODELS['command-a-03-2025']
        assert model.supports.media is False

    def test_reasoning_model_supports_tools(self) -> None:
        """Test reasoning model supports tools."""
        model = SUPPORTED_COHERE_MODELS['command-a-reasoning-08-2025']
        assert model.supports.tools is True

    def test_translate_model_supports_tools(self) -> None:
        """Test translate model supports tools."""
        model = SUPPORTED_COHERE_MODELS['command-a-translate-08-2025']
        assert model.supports.tools is True

    def test_all_models_support_multiturn(self) -> None:
        """Test all models support multiturn."""
        for name, info in SUPPORTED_COHERE_MODELS.items():
            assert info.supports.multiturn is True, f'{name} should support multiturn'

    def test_all_models_support_system_role(self) -> None:
        """Test all models support system role."""
        for name, info in SUPPORTED_COHERE_MODELS.items():
            assert info.supports.system_role is True, f'{name} should support system role'

    def test_all_non_vision_models_support_json_output(self) -> None:
        """Test all non vision models support json output."""
        for name, info in SUPPORTED_COHERE_MODELS.items():
            if name != 'command-a-vision-07-2025':
                assert 'json' in (info.supports.output or []), f'{name} should support JSON output'


class TestSupportedEmbeddingModels:
    """Tests for SUPPORTED_EMBEDDING_MODELS registry."""

    def test_contains_embed_v4(self) -> None:
        """Test contains embed v4."""
        assert 'embed-v4.0' in SUPPORTED_EMBEDDING_MODELS

    def test_contains_embed_english_v3(self) -> None:
        """Test contains embed english v3."""
        assert 'embed-english-v3.0' in SUPPORTED_EMBEDDING_MODELS

    def test_contains_embed_english_light_v3(self) -> None:
        """Test contains embed english light v3."""
        assert 'embed-english-light-v3.0' in SUPPORTED_EMBEDDING_MODELS

    def test_contains_embed_multilingual_v3(self) -> None:
        """Test contains embed multilingual v3."""
        assert 'embed-multilingual-v3.0' in SUPPORTED_EMBEDDING_MODELS

    def test_contains_embed_multilingual_light_v3(self) -> None:
        """Test contains embed multilingual light v3."""
        assert 'embed-multilingual-light-v3.0' in SUPPORTED_EMBEDDING_MODELS

    def test_total_embedding_count(self) -> None:
        """Test total embedding count."""
        assert len(SUPPORTED_EMBEDDING_MODELS) == 5

    def test_all_have_label(self) -> None:
        """Test all have label."""
        for name, meta in SUPPORTED_EMBEDDING_MODELS.items():
            assert 'label' in meta, f'Embedding model {name} missing label'

    def test_all_have_dimensions(self) -> None:
        """Test all have dimensions."""
        for name, meta in SUPPORTED_EMBEDDING_MODELS.items():
            assert 'dimensions' in meta, f'Embedding model {name} missing dimensions'
            assert isinstance(meta['dimensions'], int)

    def test_full_models_have_1024_dims(self) -> None:
        """Test full models have 1024 dims."""
        for name in ('embed-v4.0', 'embed-english-v3.0', 'embed-multilingual-v3.0'):
            assert SUPPORTED_EMBEDDING_MODELS[name]['dimensions'] == 1024

    def test_light_models_have_384_dims(self) -> None:
        """Test light models have 384 dims."""
        for name in ('embed-english-light-v3.0', 'embed-multilingual-light-v3.0'):
            assert SUPPORTED_EMBEDDING_MODELS[name]['dimensions'] == 384


class TestGetDefaultModelInfo:
    """Tests for get_default_model_info."""

    def test_label_format(self) -> None:
        """Test label format."""
        info = get_default_model_info('custom-model')
        assert info.label == 'Cohere - custom-model'

    def test_conservative_defaults(self) -> None:
        """Test conservative defaults."""
        info = get_default_model_info('unknown')
        assert info.supports.multiturn is True
        assert info.supports.tools is False
        assert info.supports.media is False
        assert info.supports.system_role is True
        assert info.supports.output == ['text']

    def test_returns_model_info_type(self) -> None:
        """Test returns model info type."""
        info = get_default_model_info('anything')
        assert isinstance(info, ModelInfo)
        assert isinstance(info.supports, ModelSupports)
