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

from collections.abc import Callable
from enum import Enum
from typing import get_args

import pytest
from genkit_google_genai import (
    GoogleAI,
    KnownGemini,
    KnownGeminiImage,
    KnownGeminiTts,
    KnownGemma,
    KnownImagen,
    KnownVeo,
    VertexAI,
)
from genkit_google_genai.models.gemini import (
    GEMINI_CATALOG_IDS,
    GeminiConfigSchema,
    GeminiImageConfigSchema,
    GeminiTtsConfigSchema,
    GemmaConfigSchema,
    GoogleAIGeminiVersion,
    VertexAIGeminiVersion,
    is_gemini_model,
    is_gemma_model,
    is_image_model,
    is_tts_model,
)
from genkit_google_genai.models.imagen import (
    ImagenConfigSchema,
    ImagenVersion,
    is_imagen_model_name,
)
from genkit_google_genai.models.veo import VeoConfig, VeoVersion, is_veo_model

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
        assert googleai_ref.config_schema is GeminiConfigSchema
        assert vertexai_ref.name == 'vertexai/gemini-2.5-flash'

    def test_enum_names_still_work(self) -> None:
        """The existing version enums remain valid constructor input."""
        assert GoogleAI.gemini_model(GoogleAIGeminiVersion.GEMINI_2_5_FLASH).name == 'googleai/gemini-2.5-flash'
        assert GoogleAI.imagen_model(ImagenVersion.IMAGEN3).name == 'googleai/imagen-3.0-generate-002'
        assert GoogleAI.veo_model(VeoVersion.VEO_3_1_FAST_PREVIEW).name == 'googleai/veo-3.1-fast-generate-preview'
        assert VertexAI.veo_model(VeoVersion.VEO_3_1).name == 'vertexai/veo-3.1-generate-001'

    def test_family_constructors_type_their_config(self) -> None:
        """Each family constructor carries its own config schema."""
        tts = GoogleAI.gemini_tts_model('gemini-2.5-flash-preview-tts')
        image = VertexAI.gemini_image_model('gemini-2.5-flash-image')
        gemma = GoogleAI.gemma_model('gemma-3-12b-it')
        imagen = VertexAI.imagen_model('imagen-3.0-generate-002')
        veo = GoogleAI.veo_model('veo-3.1-fast-generate-preview')

        assert tts.config_schema is GeminiTtsConfigSchema
        assert image.config_schema is GeminiImageConfigSchema
        assert gemma.config_schema is GemmaConfigSchema
        assert imagen.config_schema is ImagenConfigSchema
        assert imagen.name == 'vertexai/imagen-3.0-generate-002'
        assert veo.config_schema is VeoConfig
        assert veo.name == 'googleai/veo-3.1-fast-generate-preview'

    def test_config_instance_rides_along(self) -> None:
        """A default config passed at construction survives into the ref."""
        config = GeminiConfigSchema(temperature=0.3)
        ref = GoogleAI.gemini_model('gemini-2.5-flash', config=config)
        assert ref.config == config

    def test_unknown_id_allowed_on_gemini_model_and_embedding(self) -> None:
        """A brand-new release must work before this plugin learns its name."""
        assert GoogleAI.gemini_model('totally-new-model').name == 'googleai/totally-new-model'
        assert GoogleAI.embedding('totally-new-embedder').name == 'googleai/totally-new-embedder'

        with pytest.raises(GenkitError):
            GoogleAI.imagen_model('totally-new-model')


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
            'imagen-3.0-generate-002',  # wrong family: has its own constructor
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
            VertexAI.imagen_model('gemini-2.5-flash')
        with pytest.raises(GenkitError, match=r'embedding'):
            GoogleAI.gemini_model('gemini-embedding-001')
        with pytest.raises(GenkitError, match=r'veo_model'):
            GoogleAI.gemini_model('veo-3.0-generate-001')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            GoogleAI.gemini_model('lyria-002')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            GoogleAI.gemini_model('deep-research-pro-preview')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            GoogleAI.gemini_model('antigravity-code-1')
        with pytest.raises(GenkitError, match=r'is not a supported model'):
            GoogleAI.gemini_model('imagegeneration@006')
        with pytest.raises(GenkitError, match=r'is not a supported model'):
            GoogleAI.gemini_model('virtual-try-on-001')
        with pytest.raises(GenkitError, match=r'is not a imagen model'):
            GoogleAI.imagen_model('totally-new-model')

    def test_family_constructors_reject_other_families(self) -> None:
        """Non-gemini constructors take only their own family ids."""
        with pytest.raises(GenkitError):
            GoogleAI.imagen_model('gemini-2.5-flash')
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


def _enum_ids(*enums: type[Enum]) -> set[str]:
    return {str(member.value) for enum in enums for member in enum}


def _family_catalog(is_family: Callable[[str], bool]) -> set[str]:
    """Version enums plus ``_add_model`` names for one family."""
    enum_ids = _enum_ids(GoogleAIGeminiVersion, VertexAIGeminiVersion)
    return {name for name in enum_ids | GEMINI_CATALOG_IDS if is_family(name)}


class TestKnownIdLiterals:
    """Constructor name types are string Literals so quotes autocomplete."""

    def test_known_gemini_matches_catalog(self) -> None:
        """Quote autocomplete and the text catalog are the same set of ids."""
        known = set(get_args(KnownGemini))
        assert known == _family_catalog(is_gemini_model)
        assert not any(is_tts_model(value) or is_image_model(value) or is_gemma_model(value) for value in known)

    def test_family_literals_match_their_catalog(self) -> None:
        """Sibling constructors autocomplete exactly their catalog ids."""
        assert set(get_args(KnownGeminiTts)) == _family_catalog(is_tts_model)
        assert set(get_args(KnownGeminiImage)) == _family_catalog(is_image_model)
        assert set(get_args(KnownGemma)) == _family_catalog(is_gemma_model)
        assert set(get_args(KnownImagen)) == {str(member.value) for member in ImagenVersion}
        assert all(is_imagen_model_name(value) for value in get_args(KnownImagen))
        assert set(get_args(KnownVeo)) == {str(member.value) for member in VeoVersion}
        assert all(is_veo_model(value) for value in get_args(KnownVeo))
