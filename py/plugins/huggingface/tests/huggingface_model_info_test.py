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

"""Tests for Hugging Face model info registry."""

from genkit.plugins.huggingface.model_info import (
    POPULAR_HUGGINGFACE_MODELS,
    get_default_model_info,
)


class TestPopularModels:
    """Tests for the POPULAR_HUGGINGFACE_MODELS registry."""

    def test_registry_not_empty(self) -> None:
        """Test Registry not empty."""
        assert len(POPULAR_HUGGINGFACE_MODELS) > 0

    def test_all_have_labels(self) -> None:
        """Test All have labels."""
        for model_id, info in POPULAR_HUGGINGFACE_MODELS.items():
            assert info.label, f'{model_id} missing label'

    def test_all_have_versions(self) -> None:
        """Test All have versions."""
        for model_id, info in POPULAR_HUGGINGFACE_MODELS.items():
            assert info.versions, f'{model_id} missing versions'

    def test_all_have_supports(self) -> None:
        """Test All have supports."""
        for model_id, info in POPULAR_HUGGINGFACE_MODELS.items():
            assert info.supports is not None, f'{model_id} missing supports'

    def test_version_matches_model_id(self) -> None:
        """Test Version matches model id."""
        for model_id, info in POPULAR_HUGGINGFACE_MODELS.items():
            assert info.versions is not None
            assert model_id in info.versions

    def test_llama_model_exists(self) -> None:
        """Test Llama model exists."""
        assert 'meta-llama/Llama-3.1-8B-Instruct' in POPULAR_HUGGINGFACE_MODELS

    def test_embedding_model_no_multiturn(self) -> None:
        """Test Embedding model no multiturn."""
        embed = POPULAR_HUGGINGFACE_MODELS['sentence-transformers/all-MiniLM-L6-v2']
        assert embed.supports is not None
        assert embed.supports.multiturn is False
        assert embed.supports.tools is False

    def test_text_model_has_multiturn(self) -> None:
        """Test Text model has multiturn."""
        llama = POPULAR_HUGGINGFACE_MODELS['meta-llama/Llama-3.1-8B-Instruct']
        assert llama.supports is not None
        assert llama.supports.multiturn is True

    def test_code_model_supports_json(self) -> None:
        """Test Code model supports json."""
        coder = POPULAR_HUGGINGFACE_MODELS['Qwen/Qwen2.5-Coder-32B-Instruct']
        assert coder.supports is not None
        assert 'json' in (coder.supports.output or [])

    def test_model_ids_contain_slash(self) -> None:
        """Test Model ids contain slash."""
        for model_id in POPULAR_HUGGINGFACE_MODELS:
            assert '/' in model_id, f'{model_id} should be org/model format'


class TestGetDefaultModelInfo:
    """Tests for get_default_model_info fallback."""

    def test_returns_model_info(self) -> None:
        """Test Returns model info."""
        info = get_default_model_info('org/my-custom-model')
        assert info.label is not None

    def test_label_contains_model_name(self) -> None:
        """Test Label contains model name."""
        info = get_default_model_info('user/awesome-llm')
        assert info.label is not None
        assert 'awesome-llm' in info.label

    def test_label_for_no_slash(self) -> None:
        """Test Label for no slash."""
        info = get_default_model_info('standalone-model')
        assert info.label is not None
        assert 'standalone-model' in info.label

    def test_default_has_multiturn(self) -> None:
        """Test Default has multiturn."""
        info = get_default_model_info('any/model')
        assert info.supports is not None
        assert info.supports.multiturn is True

    def test_default_has_tools(self) -> None:
        """Test Default has tools."""
        info = get_default_model_info('any/model')
        assert info.supports is not None
        assert info.supports.tools is True
