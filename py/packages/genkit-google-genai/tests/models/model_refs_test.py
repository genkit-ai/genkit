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

"""Tests for the typed family ref constructors on GoogleAI / VertexAI."""

import importlib
from collections.abc import Callable
from typing import get_args

import genkit_google_genai
import pytest
from genkit_google_genai import (
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    GoogleAI,
    VertexAI,
)
from genkit_google_genai._models._gemini import (
    GEMINI_CATALOG_IDS,
    KnownGemini,
    KnownGeminiImage,
    KnownGeminiTts,
    KnownGemma,
    is_gemini_model,
    is_gemma_model,
    is_image_model,
    is_tts_model,
)
from genkit_google_genai._models._interactions_registry import KnownLyria, is_lyria_model_name
from genkit_google_genai._models._veo import KnownVeo, VeoConfig, is_veo_model

from genkit import GenkitError
from genkit.embedder import EmbedderRef
from genkit.model import ModelRef


class TestHappyPaths:
    """Each constructor mints its family's typed ref under its own namespace."""

    def test_gemini_model_both_plugins(self) -> None:
        """Same constructor name works on both plugins, each stamping its namespace."""
        googleai_ref = GoogleAI.gemini_model('gemini-2.5-flash')
        vertexai_ref = VertexAI.gemini_model('gemini-2.5-flash')

        assert isinstance(googleai_ref, ModelRef)
        assert googleai_ref.name == 'googleai/gemini-2.5-flash'
        assert googleai_ref.config_schema is GeminiConfig
        assert vertexai_ref.name == 'vertexai/gemini-2.5-flash'

    def test_unlisted_ids_still_mint_refs(self) -> None:
        """An id missing from the ``Known*`` Literal still works, so a new release needs no SDK bump."""
        assert GoogleAI.gemini_model('gemini-9.9-flash').name == 'googleai/gemini-9.9-flash'
        assert GoogleAI.veo_model('veo-9.9-generate-001').name == 'googleai/veo-9.9-generate-001'
        assert GoogleAI.lyria_model('lyria-9-preview').name == 'googleai/lyria-9-preview'

    def test_family_constructors_type_their_config(self) -> None:
        """Each family constructor carries its own config schema."""
        tts = GoogleAI.gemini_tts_model('gemini-2.5-flash-preview-tts')
        image = VertexAI.gemini_image_model('gemini-2.5-flash-image')
        gemma = GoogleAI.gemma_model('gemma-3-12b-it')
        veo = GoogleAI.veo_model('veo-3.1-fast-generate-preview')

        assert tts.config_schema is GeminiTtsConfig
        assert image.config_schema is GeminiImageConfig
        assert image.name == 'vertexai/gemini-2.5-flash-image'
        assert gemma.config_schema is GemmaConfig
        assert veo.config_schema is VeoConfig
        assert veo.name == 'googleai/veo-3.1-fast-generate-preview'

    def test_config_instance_rides_along(self) -> None:
        """A default config passed at construction survives into the ref."""
        config = GeminiConfig(temperature=0.3)
        ref = GoogleAI.gemini_model('gemini-2.5-flash', config=config)
        assert ref.config == config

    def test_unknown_id_allowed_on_gemini_model_and_embedding(self) -> None:
        """A brand-new release must work before this plugin learns its name."""
        assert GoogleAI.gemini_model('totally-new-model').name == 'googleai/totally-new-model'
        assert GoogleAI.embedding('totally-new-embedder').name == 'googleai/totally-new-embedder'

        with pytest.raises(GenkitError):
            GoogleAI.gemma_model('totally-new-model')


class TestStripThenPrefix:
    """Pasted prefixes are stripped so the constructor decides the namespace."""

    @pytest.mark.parametrize(
        'pasted',
        [
            'googleai/gemini-2.5-flash',
            'vertexai/gemini-2.5-flash',
            'model/gemini-2.5-flash',
            'models/gemini-2.5-flash',
            'models/googleai/gemini-2.5-flash',
        ],
    )
    def test_cross_plugin_paste_cannot_smuggle_namespace(self, pasted: str) -> None:
        """Every pasted prefix form lands on this plugin, not the pasted one."""
        assert GoogleAI.gemini_model(pasted).name == 'googleai/gemini-2.5-flash'
        assert VertexAI.gemini_model(pasted).name == 'vertexai/gemini-2.5-flash'

    def test_empty_after_strip_is_rejected(self) -> None:
        """A bare prefix with no id left is an invalid argument."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini_model('googleai/')
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'model name is required' in str(exc_info.value)

        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.embedding('googleai/')
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'embedder name is required' in str(exc_info.value)

        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.embedding('')
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'embedder name is required' in str(exc_info.value)

    def test_non_string_name_is_rejected(self) -> None:
        """A non-string must not become a name via str() (None → 'None')."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini_model(None)  # type: ignore[arg-type]
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'must be a string' in str(exc_info.value)

        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.embedding(123)  # type: ignore[arg-type]
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert 'must be a string' in str(exc_info.value)


class TestClosedRejectSet:
    """Ids whose action validates another schema must not mint this ref."""

    @pytest.mark.parametrize(
        'bad_id',
        [
            'veo-3.0-generate-001',  # wrong family: has its own constructor
            'lyria-002',  # no constructor in this plugin
            'googleai/lyria-002',  # prefix must not defeat the gate
            'deep-research-pro-preview',  # Interactions API family
            'antigravity-code-1',  # Interactions API family
            'imagegeneration@006',  # retired June 2026
            'virtual-try-on-001',  # predict shape not implemented
            'gemini-embedding-001',  # embedder, not a generate model
            'imagen-3.0-generate-002',  # not a supported model
            'imagen-4.0-generate-001',  # not a supported model
            'gemini-2.5-flash-preview-tts',  # wrong family: TTS
            'gemini-2.5-flash-image',  # wrong family: native image
            'gemma-3-12b-it',  # wrong family: Gemma
        ],
    )
    def test_gemini_model_rejects(self, bad_id: str) -> None:
        """gemini_model refuses every id whose action validates another schema."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini_model(bad_id)
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_error_points_at_the_right_constructor(self) -> None:
        """Rejections tell the caller which constructor to use instead."""
        with pytest.raises(GenkitError, match=r'gemini_tts_model'):
            GoogleAI.gemini_model('gemini-2.5-flash-preview-tts')
        with pytest.raises(GenkitError, match=r'VertexAI\.gemini_model'):
            VertexAI.gemma_model('gemini-2.5-flash')
        with pytest.raises(GenkitError, match=r'embedding'):
            GoogleAI.gemini_model('gemini-embedding-001')
        with pytest.raises(GenkitError, match=r'veo_model'):
            GoogleAI.gemini_model('veo-3.0-generate-001')
        with pytest.raises(GenkitError, match=r'lyria_model'):
            GoogleAI.gemini_model('lyria-002')
        with pytest.raises(GenkitError, match=r'deep_research_model'):
            GoogleAI.gemini_model('deep-research-pro-preview')
        with pytest.raises(GenkitError, match=r'antigravity_model'):
            GoogleAI.gemini_model('antigravity-code-1')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            VertexAI.gemini_model('lyria-002')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            VertexAI.gemini_model('deep-research-pro-preview')
        with pytest.raises(GenkitError, match=r'is not a supported model'):
            GoogleAI.gemini_model('imagegeneration@006')
        with pytest.raises(GenkitError, match=r'is not a supported model'):
            GoogleAI.gemini_model('virtual-try-on-001')
        with pytest.raises(GenkitError, match=r'GoogleAI\.gemini_image_model'):
            GoogleAI.gemini_model('imagen-4.0-generate-001')
        with pytest.raises(GenkitError, match=r'for image generation use VertexAI\.gemini_image_model\(\)'):
            VertexAI.gemini_model('imagen-3.0-generate-002')
        with pytest.raises(GenkitError, match=r'is not a gemma model'):
            GoogleAI.gemma_model('totally-new-model')

    def test_family_constructors_reject_other_families(self) -> None:
        """Non-gemini constructors take only their own family ids."""
        with pytest.raises(GenkitError):
            GoogleAI.gemini_tts_model('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            GoogleAI.gemma_model('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            VertexAI.gemini_image_model('imagen-3.0-generate-002')
        with pytest.raises(GenkitError):
            GoogleAI.veo_model('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            GoogleAI.veo_model('totally-new-model')
        with pytest.raises(GenkitError, match=r'model name must be a string'):
            GoogleAI.deep_research_model(None)  # type: ignore[arg-type]


class TestEmbeddingConstructor:
    """embedding() returns an EmbedderRef so it can't be passed as a model."""

    def test_embedding_ref(self) -> None:
        """embedding() returns an EmbedderRef under this plugin namespace."""
        ref = GoogleAI.embedding('gemini-embedding-001')
        assert isinstance(ref, EmbedderRef)
        assert not isinstance(ref, ModelRef)
        assert ref.name == 'googleai/gemini-embedding-001'

    def test_embedding_strips_and_prefixes(self) -> None:
        """Pasted embedder prefixes are stripped before namespacing."""
        ref = VertexAI.embedding('embedders/googleai/text-embedding-004', version='text-embedding-005')
        assert ref.name == 'vertexai/text-embedding-004'
        assert ref.version == 'text-embedding-005'

    def test_embedding_rejects_generate_models(self) -> None:
        """Generate-model ids cannot mint an EmbedderRef."""
        with pytest.raises(GenkitError, match=r'gemini_model'):
            GoogleAI.embedding('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            VertexAI.embedding('imagen-3.0-generate-002')


def _registered(is_family: Callable[[str], bool]) -> set[str]:
    """``_add_model`` names for one family."""
    return {name for name in GEMINI_CATALOG_IDS if is_family(name)}


class TestKnownIdLiterals:
    """Constructor name types are string Literals so quotes autocomplete."""

    def test_known_gemini_matches_catalog(self) -> None:
        """Quote autocomplete and the registered text catalog are the same set of ids."""
        known = set(get_args(KnownGemini))
        assert known == _registered(is_gemini_model)
        assert not any(is_tts_model(value) or is_image_model(value) or is_gemma_model(value) for value in known)

    def test_family_literals_match_their_catalog(self) -> None:
        """Every registered id autocompletes, and every autocomplete id routes to its family."""
        assert set(get_args(KnownGeminiTts)) == _registered(is_tts_model)
        assert set(get_args(KnownGeminiImage)) == _registered(is_image_model)
        # Gemma 3 ids resolve through the generic Gemma info, not _add_model.
        assert _registered(is_gemma_model) <= set(get_args(KnownGemma))
        assert all(is_gemma_model(value) for value in get_args(KnownGemma))
        assert all(is_veo_model(value) for value in get_args(KnownVeo))
        assert all(is_lyria_model_name(value) for value in get_args(KnownLyria))

    def test_latest_aliases_autocomplete_on_gemini_model(self) -> None:
        """Every rotating ``-latest`` alias is offered by ``gemini_model``."""
        assert {'gemini-pro-latest', 'gemini-flash-latest', 'gemini-flash-lite-latest'} <= set(get_args(KnownGemini))

    def test_infix_tts_id_autocompletes_on_the_tts_constructor(self) -> None:
        """A ``-tts`` infix belongs to the TTS constructor, not the text one."""
        assert 'gemini-3.1-flash-tts-preview' in get_args(KnownGeminiTts)
        assert 'gemini-3.1-flash-tts-preview' not in get_args(KnownGemini)

    def test_gemini_3_8_tts_ids_autocomplete_on_the_tts_constructor(self) -> None:
        """The 3.8 TTS ids are offered by ``gemini_tts_model``, not ``gemini_model``."""
        ids = {'gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'}
        assert ids <= set(get_args(KnownGeminiTts))
        assert not ids & set(get_args(KnownGemini))
        for name in ids:
            assert GoogleAI.gemini_tts_model(name).config_schema is GeminiTtsConfig

    def test_gemma_4_autocompletes_on_gemma_model(self) -> None:
        """The gemma-4 ids are offered by ``gemma_model``."""
        assert {'gemma-4-26b-a4b-it', 'gemma-4-31b-it'} <= set(get_args(KnownGemma))


class TestNoImagenSurface:
    """Imagen has no export or module, and imagen_model only raises."""

    def test_no_imagen_exports(self) -> None:
        """The package exposes no Imagen name."""
        assert not {name for name in genkit_google_genai.__all__ if 'Imagen' in name}

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_imagen_model_raises_with_the_image_hint(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """imagen_model raises INVALID_ARGUMENT pointing at gemini_image_model."""
        cls = plugin.__name__
        with pytest.raises(
            GenkitError, match=rf'{cls}\.imagen_model: .* use {cls}\.gemini_image_model\(\)'
        ) as exc_info:
            plugin.imagen_model('imagen-4.0-generate-001')
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_no_imagen_module(self) -> None:
        """The Imagen model module is gone."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('genkit_google_genai._models.imagen')

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_every_constructor_rejects_imagen_ids(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """An imagen- id is refused by every other family constructor with the image hint."""
        for method in ('gemini_model', 'gemini_tts_model', 'gemma_model'):
            with pytest.raises(GenkitError, match=r'for image generation use \w+\.gemini_image_model\(\)'):
                getattr(plugin, method)('imagen-4.0-generate-001')
        with pytest.raises(GenkitError, match=r'for image generation use \w+\.gemini_image_model\(\)'):
            plugin.embedding('imagen-4.0-generate-001')

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_gemini_image_model_does_not_suggest_itself(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """gemini_image_model refuses an imagen- id without pointing back at itself."""
        with pytest.raises(GenkitError) as exc_info:
            plugin.gemini_image_model('imagen-4.0-generate-001')
        assert 'is not a supported model' in str(exc_info.value)
        assert 'for image generation' not in str(exc_info.value)

    @pytest.mark.parametrize('bad_id', ['imagegeneration@006', 'imagetext@001', 'virtual-try-on-001'])
    def test_other_unsupported_ids_omit_the_image_hint(self, bad_id: str) -> None:
        """Only imagen- ids are redirected to image generation."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini_model(bad_id)
        assert 'is not a supported model' in str(exc_info.value)
        assert 'gemini_image_model' not in str(exc_info.value)
