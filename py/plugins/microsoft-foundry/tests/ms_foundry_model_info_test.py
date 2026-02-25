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

"""Tests for Microsoft Foundry model info registry.

Tests cover model registry completeness, capability matrix verification,
embedding model info, response format support, and the get_model_info fallback.
"""

from genkit.plugins.microsoft_foundry.models.model_info import (
    MODELS_SUPPORTING_RESPONSE_FORMAT,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_MICROSOFT_FOUNDRY_MODELS,
    get_model_info,
)


# ---------------------------------------------------------------------------
# Model Registry Completeness
# ---------------------------------------------------------------------------
class TestModelRegistryCompleteness:
    """Verify the model registry has the expected model families."""

    def test_gpt4_models_present(self) -> None:
        """Gpt4 models present."""
        gpt4_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.startswith('gpt-4')]
        assert len(gpt4_models) >= 4, f'Expected at least 4 GPT-4 models, got {len(gpt4_models)}'

    def test_o_series_models_present(self) -> None:
        """O series models present."""
        o_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.startswith('o') and not k.startswith('op')]
        assert len(o_models) >= 4, f'Expected at least 4 o-series models, got {len(o_models)}'

    def test_claude_models_present(self) -> None:
        """Claude models present."""
        claude_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if 'claude' in k.lower()]
        assert len(claude_models) >= 2, f'Expected at least 2 Claude models, got {len(claude_models)}'

    def test_deepseek_models_present(self) -> None:
        """Deepseek models present."""
        ds_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.startswith('DeepSeek')]
        assert len(ds_models) >= 2, f'Expected at least 2 DeepSeek models, got {len(ds_models)}'

    def test_grok_models_present(self) -> None:
        """Grok models present."""
        grok_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.startswith('grok')]
        assert len(grok_models) >= 2, f'Expected at least 2 Grok models, got {len(grok_models)}'

    def test_llama_models_present(self) -> None:
        """Llama models present."""
        llama_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.startswith('Llama')]
        assert len(llama_models) >= 1, f'Expected at least 1 Llama model, got {len(llama_models)}'

    def test_mistral_models_present(self) -> None:
        """Mistral models present."""
        mistral_models = [k for k in SUPPORTED_MICROSOFT_FOUNDRY_MODELS if k.lower().startswith('mistral')]
        assert len(mistral_models) >= 1, f'Expected at least 1 Mistral model, got {len(mistral_models)}'


# ---------------------------------------------------------------------------
# Capability Matrix
# ---------------------------------------------------------------------------
class TestCapabilityMatrix:
    """Verify capability declarations match expected patterns per model family."""

    def test_gpt4o_supports_multimodal(self) -> None:
        """Gpt4o supports multimodal."""
        info = SUPPORTED_MICROSOFT_FOUNDRY_MODELS['gpt-4o']
        assert info.supports is not None
        assert info.supports.multiturn is True
        assert info.supports.media is True
        assert info.supports.tools is True
        assert info.supports.system_role is True

    def test_o1_is_reasoning(self) -> None:
        """O1 is reasoning."""
        info = SUPPORTED_MICROSOFT_FOUNDRY_MODELS['o1']
        assert info.supports is not None
        assert info.supports.multiturn is True
        # o1 is a reasoning model — tool support varies by model variant
        assert isinstance(info.supports.tools, bool)

    def test_claude_supports_multimodal(self) -> None:
        """Claude supports multimodal."""
        for model_name in SUPPORTED_MICROSOFT_FOUNDRY_MODELS:
            if 'claude' in model_name.lower():
                info = SUPPORTED_MICROSOFT_FOUNDRY_MODELS[model_name]
                assert info.supports is not None
                assert info.supports.multiturn is True
                assert info.supports.tools is True

    def test_deepseek_no_media(self) -> None:
        """Deepseek no media."""
        for model_name in SUPPORTED_MICROSOFT_FOUNDRY_MODELS:
            if model_name.startswith('DeepSeek'):
                info = SUPPORTED_MICROSOFT_FOUNDRY_MODELS[model_name]
                assert info.supports is not None
                assert info.supports.media is False

    def test_all_models_have_supports(self) -> None:
        """All models have supports."""
        for name, info in SUPPORTED_MICROSOFT_FOUNDRY_MODELS.items():
            assert info.supports is not None, f'Model {name} is missing supports declaration'

    def test_all_models_have_labels(self) -> None:
        """All models have labels."""
        for name, info in SUPPORTED_MICROSOFT_FOUNDRY_MODELS.items():
            assert info.label is not None, f'Model {name} is missing label'
            assert info.label.startswith('Microsoft Foundry'), (
                f'Model {name} label should start with "Microsoft Foundry"'
            )


# ---------------------------------------------------------------------------
# get_model_info() — Lookup and Fallback
# ---------------------------------------------------------------------------
class TestGetModelInfo:
    """Tests for GetModelInfo."""

    def test_known_model_returns_exact_info(self) -> None:
        """Known model returns exact info."""
        info = get_model_info('gpt-4o')
        assert info.label == 'Microsoft Foundry - GPT-4o'

    def test_unknown_model_returns_default(self) -> None:
        """Unknown model returns default."""
        info = get_model_info('totally-new-model-xyz')
        assert info.label == 'Microsoft Foundry - totally-new-model-xyz'
        assert info.supports is not None
        assert info.supports.multiturn is True  # Default multimodal

    def test_unknown_model_is_multimodal_by_default(self) -> None:
        """Azure has 11,000+ models; unknown ones get generous defaults."""
        info = get_model_info('future-model-2030')
        assert info.supports is not None
        assert info.supports.media is True
        assert info.supports.tools is True


# ---------------------------------------------------------------------------
# Response Format Support
# ---------------------------------------------------------------------------
class TestResponseFormatSupport:
    """Tests for ResponseFormatSupport."""

    def test_gpt4o_in_response_format_list(self) -> None:
        """Gpt4o in response format list."""
        assert 'gpt-4o' in MODELS_SUPPORTING_RESPONSE_FORMAT

    def test_o1_in_response_format_list(self) -> None:
        """O1 in response format list."""
        assert 'o1' in MODELS_SUPPORTING_RESPONSE_FORMAT

    def test_no_duplicates(self) -> None:
        """No duplicates."""
        assert len(MODELS_SUPPORTING_RESPONSE_FORMAT) == len(set(MODELS_SUPPORTING_RESPONSE_FORMAT))


# ---------------------------------------------------------------------------
# Embedding Models
# ---------------------------------------------------------------------------
class TestEmbeddingModels:
    """Tests for EmbeddingModels."""

    def test_three_openai_embedding_models(self) -> None:
        """Three openai embedding models."""
        openai_embeds = [k for k in SUPPORTED_EMBEDDING_MODELS if k.startswith('text-embedding')]
        assert len(openai_embeds) == 3

    def test_cohere_embed_present(self) -> None:
        """Cohere embed present."""
        assert 'embed-v-4-0' in SUPPORTED_EMBEDDING_MODELS

    def test_embedding_dimensions(self) -> None:
        """Embedding dimensions."""
        assert SUPPORTED_EMBEDDING_MODELS['text-embedding-3-small']['dimensions'] == 1536
        assert SUPPORTED_EMBEDDING_MODELS['text-embedding-3-large']['dimensions'] == 3072
        assert SUPPORTED_EMBEDDING_MODELS['text-embedding-ada-002']['dimensions'] == 1536

    def test_all_embeddings_have_labels(self) -> None:
        """All embeddings have labels."""
        for name, info in SUPPORTED_EMBEDDING_MODELS.items():
            assert 'label' in info, f'Embedding model {name} is missing label'
            assert info['label'].startswith('Microsoft Foundry'), (
                f'Embedding {name} label should start with "Microsoft Foundry"'
            )
