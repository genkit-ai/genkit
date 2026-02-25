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

"""Tests for Cloudflare Workers AI model info registry."""

from genkit.plugins.cloudflare_workers_ai.models.model_info import (
    SUPPORTED_CF_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    get_model_info,
)


class TestSupportedCfModels:
    """Tests for the SUPPORTED_CF_MODELS registry."""

    def test_registry_not_empty(self) -> None:
        """Test Registry not empty."""
        assert len(SUPPORTED_CF_MODELS) > 0

    def test_all_models_have_labels(self) -> None:
        """Test All models have labels."""
        for model_id, info in SUPPORTED_CF_MODELS.items():
            assert info.label, f'{model_id} missing label'

    def test_all_models_have_versions(self) -> None:
        """Test All models have versions."""
        for model_id, info in SUPPORTED_CF_MODELS.items():
            assert info.versions, f'{model_id} missing versions'

    def test_all_models_have_supports(self) -> None:
        """Test All models have supports."""
        for model_id, info in SUPPORTED_CF_MODELS.items():
            assert info.supports is not None, f'{model_id} missing supports'

    def test_version_contains_model_id(self) -> None:
        """Test Version contains model id."""
        for model_id, info in SUPPORTED_CF_MODELS.items():
            assert info.versions is not None
            assert model_id in info.versions, f'{model_id} not in its own versions list'

    def test_llama_model_exists(self) -> None:
        """Test Llama model exists."""
        assert '@cf/meta/llama-3.1-8b-instruct' in SUPPORTED_CF_MODELS

    def test_multimodal_model_has_media_support(self) -> None:
        """Test Multimodal model has media support."""
        scout = SUPPORTED_CF_MODELS['@cf/meta/llama-4-scout-17b-16e-instruct']
        assert scout.supports is not None
        assert scout.supports.media is True

    def test_text_only_model_no_media(self) -> None:
        """Test Text only model no media."""
        llama = SUPPORTED_CF_MODELS['@cf/meta/llama-3.1-8b-instruct']
        assert llama.supports is not None
        assert llama.supports.media is False

    def test_model_ids_use_cf_or_hf_prefix(self) -> None:
        """Test Model ids use cf or hf prefix."""
        for model_id in SUPPORTED_CF_MODELS:
            assert model_id.startswith('@cf/') or model_id.startswith('@hf/'), f'{model_id} has unexpected prefix'


class TestSupportedEmbeddingModels:
    """Tests for the SUPPORTED_EMBEDDING_MODELS registry."""

    def test_registry_not_empty(self) -> None:
        """Test Registry not empty."""
        assert len(SUPPORTED_EMBEDDING_MODELS) > 0

    def test_all_have_dimensions(self) -> None:
        """Test All have dimensions."""
        for model_id, info in SUPPORTED_EMBEDDING_MODELS.items():
            assert 'dimensions' in info, f'{model_id} missing dimensions'
            assert isinstance(info['dimensions'], int)
            assert info['dimensions'] > 0

    def test_all_have_labels(self) -> None:
        """Test All have labels."""
        for model_id, info in SUPPORTED_EMBEDDING_MODELS.items():
            assert 'label' in info, f'{model_id} missing label'

    def test_bge_base_dimensions(self) -> None:
        """Test Bge base dimensions."""
        bge = SUPPORTED_EMBEDDING_MODELS['@cf/baai/bge-base-en-v1.5']
        assert bge['dimensions'] == 768


class TestGetModelInfo:
    """Tests for the get_model_info function."""

    def test_known_model_returns_registry_info(self) -> None:
        """Test Known model returns registry info."""
        info = get_model_info('@cf/meta/llama-3.1-8b-instruct')
        assert info.label == 'Meta - Llama 3.1 8B Instruct'

    def test_unknown_model_returns_default(self) -> None:
        """Test Unknown model returns default."""
        info = get_model_info('@cf/unknown/model-v1')
        assert info.label is not None
        assert 'Cloudflare' in info.label
        assert '@cf/unknown/model-v1' in info.label

    def test_unknown_model_has_default_supports(self) -> None:
        """Test Unknown model has default supports."""
        info = get_model_info('@cf/unknown/model-v1')
        assert info.supports is not None
        assert info.supports.multiturn is True
        assert info.supports.tools is True

    def test_unknown_model_versions_contain_id(self) -> None:
        """Test Unknown model versions contain id."""
        info = get_model_info('@cf/custom/my-model')
        assert info.versions is not None
        assert '@cf/custom/my-model' in info.versions
