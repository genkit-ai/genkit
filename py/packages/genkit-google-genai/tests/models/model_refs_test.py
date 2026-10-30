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
import inspect
from collections.abc import Callable
from typing import Literal, get_args, get_origin, get_type_hints

import genkit_google_genai
import pytest
from genkit_google_genai import (
    EmbeddingConfig,
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    GoogleAI,
    VertexAI,
    VirtualTryOnConfig,
)
from genkit_google_genai._models._gemini import (
    GEMINI_CATALOG_IDS,
    KnownGeminiImage,
    KnownGeminiTts,
    KnownGemma,
    KnownGoogleAIGemini,
    KnownVertexAIGemini,
    is_gemini_model,
    is_gemma_model,
    is_image_model,
    is_tts_model,
)
from genkit_google_genai._models._interactions_registry import KnownLyria, is_lyria_model_name
from genkit_google_genai._models._veo import KnownGoogleAIVeo, KnownVertexAIVeo, VeoConfig, is_veo_model
from genkit_google_genai._models._virtual_try_on import KnownVirtualTryOn, is_virtual_try_on_model

from genkit import GenkitError
from genkit.embedder import EmbedderRef
from genkit.model import ModelRef

_SHARED_GEMINI = {
    'gemini-2.5-flash',
    'gemini-2.5-pro',
    'gemini-2.5-flash-lite',
    'gemini-3-flash-preview',
    'gemini-3.1-flash-lite',
    'gemini-3.1-pro-preview',
    'gemini-3.1-pro-preview-customtools',
    'gemini-3.5-flash',
    'gemini-3.6-flash',
    'gemini-3.7-flash',
}
_GOOGLEAI_LATEST = {'gemini-flash-latest', 'gemini-flash-lite-latest', 'gemini-pro-latest'}
_SHARED_TTS = {
    'gemini-2.5-flash-preview-tts',
    'gemini-2.5-pro-preview-tts',
    'gemini-3.1-flash-tts-preview',
    'gemini-3.8-flash-tts',
    'gemini-3.8-flash-lite-tts',
}
_SHARED_IMAGE = {'gemini-2.5-flash-image', 'gemini-3-pro-image', 'gemini-3.1-flash-image'}
# Registered in SUPPORTED_MODELS, but neither backend serves them, so they do not autocomplete.
_RETIRED_GEMINI = {
    'gemini-2.5-flash-preview-04-17',
    'gemini-2.5-pro-exp-03-25',
    'gemini-2.5-pro-preview-03-25',
    'gemini-2.5-pro-preview-05-06',
    'gemini-3-pro-preview',
    'gemini-3.1-flash-lite-preview',
}
_RETIRED_IMAGE = {'gemini-2.5-flash-image-preview', 'gemini-3-pro-image-preview', 'gemini-3.1-flash-image-preview'}

_BACKEND_IDS: dict[tuple[type, str], set[str] | None] = {
    (GoogleAI, 'gemini'): _SHARED_GEMINI | _GOOGLEAI_LATEST,
    (VertexAI, 'gemini'): _SHARED_GEMINI,
    (GoogleAI, 'gemini_tts'): _SHARED_TTS,
    (VertexAI, 'gemini_tts'): _SHARED_TTS,
    (GoogleAI, 'gemini_image'): _SHARED_IMAGE,
    (VertexAI, 'gemini_image'): _SHARED_IMAGE,
    (GoogleAI, 'gemma'): {
        'gemma-3-1b-it',
        'gemma-3-4b-it',
        'gemma-3-12b-it',
        'gemma-3-27b-it',
        'gemma-3n-e4b-it',
        'gemma-4-26b-a4b-it',
        'gemma-4-31b-it',
    },
    (VertexAI, 'gemma'): None,
    (GoogleAI, 'veo'): {
        'veo-3.1-generate-preview',
        'veo-3.1-fast-generate-preview',
        'veo-3.1-lite-generate-preview',
    },
    (VertexAI, 'veo'): {'veo-3.1-generate-001', 'veo-3.1-fast-generate-001', 'veo-3.1-lite-generate-001'},
}


class TestHappyPaths:
    """Each constructor mints its family's typed ref under its own namespace."""

    def test_gemini_both_plugins(self) -> None:
        """Same constructor name works on both plugins, each stamping its namespace."""
        googleai_ref = GoogleAI.gemini('gemini-2.5-flash')
        vertexai_ref = VertexAI.gemini('gemini-2.5-flash')

        assert isinstance(googleai_ref, ModelRef)
        assert googleai_ref.name == 'googleai/gemini-2.5-flash'
        assert googleai_ref.config_schema is GeminiConfig
        assert vertexai_ref.name == 'vertexai/gemini-2.5-flash'

    def test_unlisted_ids_still_mint_refs(self) -> None:
        """An id missing from the ``Known*`` Literal still works, so a new release needs no SDK bump."""
        assert GoogleAI.gemini('gemini-9.9-flash').name == 'googleai/gemini-9.9-flash'
        assert GoogleAI.veo('veo-9.9-generate-001').name == 'googleai/veo-9.9-generate-001'
        assert GoogleAI.lyria('lyria-9-preview').name == 'googleai/lyria-9-preview'

    def test_family_constructors_type_their_config(self) -> None:
        """Each family constructor carries its own config schema."""
        tts = GoogleAI.gemini_tts('gemini-2.5-flash-preview-tts')
        image = VertexAI.gemini_image('gemini-2.5-flash-image')
        gemma = GoogleAI.gemma('gemma-3-12b-it')
        veo = GoogleAI.veo('veo-3.1-fast-generate-preview')
        vertex_gemma = VertexAI.gemma('gemma-3-12b-it')
        vertex_veo = VertexAI.veo('veo-3.1-generate-001')

        assert tts.config_schema is GeminiTtsConfig
        assert image.config_schema is GeminiImageConfig
        assert image.name == 'vertexai/gemini-2.5-flash-image'
        assert gemma.config_schema is GemmaConfig
        assert veo.config_schema is VeoConfig
        assert veo.name == 'googleai/veo-3.1-fast-generate-preview'
        assert vertex_gemma.config_schema is GemmaConfig
        assert vertex_gemma.name == 'vertexai/gemma-3-12b-it'
        assert vertex_veo.config_schema is VeoConfig
        assert vertex_veo.name == 'vertexai/veo-3.1-generate-001'

    def test_config_instance_rides_along(self) -> None:
        """A default config passed at construction survives into the ref."""
        config = GeminiConfig(temperature=0.3)
        ref = GoogleAI.gemini('gemini-2.5-flash', config=config)
        assert ref.config == config

    def test_unknown_id_allowed_on_gemini_and_embedding(self) -> None:
        """A brand-new release must work before this plugin learns its name."""
        assert GoogleAI.gemini('totally-new-model').name == 'googleai/totally-new-model'
        assert GoogleAI.embedding('totally-new-embedder').name == 'googleai/totally-new-embedder'

        with pytest.raises(GenkitError):
            GoogleAI.gemma('totally-new-model')


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
        assert GoogleAI.gemini(pasted).name == 'googleai/gemini-2.5-flash'
        assert VertexAI.gemini(pasted).name == 'vertexai/gemini-2.5-flash'

    def test_empty_after_strip_is_rejected(self) -> None:
        """A bare prefix with no id left is an invalid argument."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini('googleai/')
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
            GoogleAI.gemini(None)  # type: ignore[arg-type]
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
            'veo-3.1-generate-preview',  # wrong family: has its own constructor
            'lyria-002',  # no constructor in this plugin
            'googleai/lyria-002',  # prefix must not defeat the gate
            'deep-research-pro-preview',  # Interactions API family
            'antigravity-code-1',  # Interactions API family
            'imagegeneration@006',  # retired June 2026
            'virtual-try-on-001',  # Vertex-only family
            'gemini-embedding-001',  # embedder, not a generate model
            'imagen-3.0-generate-002',  # not a supported model
            'imagen-4.0-generate-001',  # not a supported model
            'gemini-2.5-flash-preview-tts',  # wrong family: TTS
            'gemini-2.5-flash-image',  # wrong family: native image
            'gemma-3-12b-it',  # wrong family: Gemma
        ],
    )
    def test_gemini_rejects(self, bad_id: str) -> None:
        """gemini refuses every id whose action validates another schema."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini(bad_id)
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_error_points_at_the_right_constructor(self) -> None:
        """Rejections tell the caller which constructor to use instead."""
        with pytest.raises(GenkitError, match=r'gemini_tts\('):
            GoogleAI.gemini('gemini-2.5-flash-preview-tts')
        with pytest.raises(GenkitError, match=r'VertexAI\.gemini\('):
            VertexAI.gemma('gemini-2.5-flash')
        with pytest.raises(GenkitError, match=r'embedding'):
            GoogleAI.gemini('gemini-embedding-001')
        with pytest.raises(GenkitError, match=r'veo\('):
            GoogleAI.gemini('veo-3.1-generate-preview')
        with pytest.raises(GenkitError, match=r'lyria\('):
            GoogleAI.gemini('lyria-002')
        with pytest.raises(GenkitError, match=r'deep_research\('):
            GoogleAI.gemini('deep-research-pro-preview')
        with pytest.raises(GenkitError, match=r'antigravity\('):
            GoogleAI.gemini('antigravity-code-1')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            VertexAI.gemini('lyria-002')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            VertexAI.gemini('deep-research-pro-preview')
        with pytest.raises(GenkitError, match=r'is not a supported model'):
            GoogleAI.gemini('imagegeneration@006')
        with pytest.raises(GenkitError, match=r'has no ref constructor in this plugin'):
            GoogleAI.gemini('virtual-try-on-001')
        with pytest.raises(GenkitError, match=r'virtual_try_on\('):
            VertexAI.gemini('virtual-try-on-001')
        with pytest.raises(GenkitError, match=r'GoogleAI\.gemini_image\('):
            GoogleAI.gemini('imagen-4.0-generate-001')
        with pytest.raises(GenkitError, match=r'for image generation use VertexAI\.gemini_image\(\)'):
            VertexAI.gemini('imagen-3.0-generate-002')
        with pytest.raises(GenkitError, match=r'is not a gemma model'):
            GoogleAI.gemma('totally-new-model')

    def test_family_constructors_reject_other_families(self) -> None:
        """Non-gemini constructors take only their own family ids."""
        with pytest.raises(GenkitError):
            GoogleAI.gemini_tts('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            GoogleAI.gemma('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            VertexAI.gemini_image('imagen-3.0-generate-002')
        with pytest.raises(GenkitError):
            GoogleAI.veo('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            GoogleAI.veo('totally-new-model')
        with pytest.raises(GenkitError, match=r'model name must be a string'):
            GoogleAI.deep_research(None)  # type: ignore[arg-type]


class TestEmbeddingConstructor:
    """embedding() returns an EmbedderRef so it can't be passed as a model."""

    def test_embedding_ref(self) -> None:
        """embedding() returns an EmbedderRef under this plugin namespace."""
        ref = GoogleAI.embedding('gemini-embedding-001')
        assert isinstance(ref, EmbedderRef)
        assert not isinstance(ref, ModelRef)
        assert ref.name == 'googleai/gemini-embedding-001'

    def test_embedding_stores_a_typed_config_as_a_dict(self) -> None:
        """A typed EmbeddingConfig lands on the ref as the dict ai.embed() merges."""
        config = EmbeddingConfig(task_type='RETRIEVAL_QUERY', output_dimensionality=256)
        ref = GoogleAI.embedding('gemini-embedding-001', config=config)
        assert ref.config == {'task_type': 'RETRIEVAL_QUERY', 'output_dimensionality': 256}

    def test_embedding_normalizes_a_dict_config_to_snake_case(self) -> None:
        """A camelCase dict config is stored by field name on both plugins."""
        config: dict[str, object] = {'taskType': 'RETRIEVAL_DOCUMENT', 'mimeType': 'text/plain'}
        expected = {'task_type': 'RETRIEVAL_DOCUMENT', 'mime_type': 'text/plain'}
        assert GoogleAI.embedding('gemini-embedding-001', config=config).config == expected
        assert VertexAI.embedding('text-embedding-005', config=config).config == expected

    def test_embedding_keeps_the_camel_case_value_when_both_spellings_are_given(self) -> None:
        """With both spellings in one dict the camelCase value is the field and the only one stored."""
        config: dict[str, object] = {'taskType': 'CLUSTERING', 'task_type': 'RETRIEVAL_QUERY'}
        assert GoogleAI.embedding('gemini-embedding-001', config=config).config == {'task_type': 'CLUSTERING'}

    @pytest.mark.parametrize(
        ('config', 'key'),
        [({'outputDimensionality': 0}, 'outputDimensionality'), ({'someFutureFlag': True}, 'someFutureFlag')],
        ids=['invalid_value', 'unknown_key'],
    )
    def test_embedding_rejects_an_invalid_dict_config(self, config: dict[str, object], key: str) -> None:
        """A dict config that fails validation raises INVALID_ARGUMENT at construction, naming the key."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.embedding('gemini-embedding-001', config=config)
        assert exc_info.value.status == 'INVALID_ARGUMENT'
        assert exc_info.value.original_message == f'GoogleAI.embedding: invalid config field {key}'

    def test_embedding_strips_and_prefixes(self) -> None:
        """Pasted embedder prefixes are stripped before namespacing."""
        ref = VertexAI.embedding('embedders/googleai/text-embedding-004', version='text-embedding-005')
        assert ref.name == 'vertexai/text-embedding-004'
        assert ref.version == 'text-embedding-005'

    def test_embedding_rejects_generate_models(self) -> None:
        """Generate-model ids cannot mint an EmbedderRef."""
        with pytest.raises(GenkitError, match=r'gemini\('):
            GoogleAI.embedding('gemini-2.5-flash')
        with pytest.raises(GenkitError):
            VertexAI.embedding('imagen-3.0-generate-002')


def _registered(is_family: Callable[[str], bool]) -> set[str]:
    """``_add_model`` names for one family."""
    return {name for name in GEMINI_CATALOG_IDS if is_family(name)}


class TestKnownIdLiterals:
    """Constructor name types are string Literals so quotes autocomplete."""

    def test_known_gemini_matches_catalog(self) -> None:
        """Each backend autocompletes every registered text id it serves; only retired ids are hidden."""
        registered = _registered(is_gemini_model)
        googleai = set(get_args(KnownGoogleAIGemini))
        vertexai = set(get_args(KnownVertexAIGemini))
        assert googleai | vertexai <= registered
        assert registered - googleai == _RETIRED_GEMINI
        assert registered - vertexai == _RETIRED_GEMINI | _GOOGLEAI_LATEST
        assert not any(is_tts_model(value) or is_image_model(value) or is_gemma_model(value) for value in googleai)

    def test_known_gemini_image_matches_catalog(self) -> None:
        """Every registered image id autocompletes except the retired ``-preview`` ids."""
        known = set(get_args(KnownGeminiImage))
        registered = _registered(is_image_model)
        assert known <= registered
        assert registered - known == _RETIRED_IMAGE

    def test_family_literals_match_their_catalog(self) -> None:
        """Registered TTS and Gemma ids autocomplete, and every autocomplete id routes to its family."""
        assert set(get_args(KnownGeminiTts)) == _registered(is_tts_model)
        # Gemma 3 ids resolve through the generic Gemma info, not _add_model.
        assert _registered(is_gemma_model) <= set(get_args(KnownGemma))
        assert all(is_gemma_model(value) for value in get_args(KnownGemma))
        assert all(is_veo_model(value) for value in get_args(KnownGoogleAIVeo) + get_args(KnownVertexAIVeo))
        assert all(is_lyria_model_name(value) for value in get_args(KnownLyria))
        assert all(is_virtual_try_on_model(value) for value in get_args(KnownVirtualTryOn))

    def test_infix_tts_id_autocompletes_on_the_tts_constructor(self) -> None:
        """A ``-tts`` infix belongs to the TTS constructor, not the text one."""
        assert 'gemini-3.1-flash-tts-preview' in get_args(KnownGeminiTts)
        assert 'gemini-3.1-flash-tts-preview' not in get_args(KnownGoogleAIGemini)

    def test_gemini_3_8_tts_ids_autocomplete_on_the_tts_constructor(self) -> None:
        """The 3.8 TTS ids are offered by ``gemini_tts``, not ``gemini``."""
        ids = {'gemini-3.8-flash-tts', 'gemini-3.8-flash-lite-tts'}
        assert ids <= set(get_args(KnownGeminiTts))
        assert not ids & set(get_args(KnownGoogleAIGemini))
        for name in ids:
            assert GoogleAI.gemini_tts(name).config_schema is GeminiTtsConfig

    def test_gemma_4_autocompletes_on_gemma(self) -> None:
        """The gemma-4 ids are offered by ``gemma``."""
        assert {'gemma-4-26b-a4b-it', 'gemma-4-31b-it'} <= set(get_args(KnownGemma))


def _autocomplete_ids(method: Callable[..., object]) -> set[str] | None:
    """The Literal ids a constructor's ``name`` param offers, or None for plain ``str``."""
    hint = get_type_hints(method)['name']
    if hint is str:
        return None
    members = get_args(hint)
    assert str in members
    literals = [member for member in members if get_origin(member) is Literal]
    assert len(literals) == 1
    return set(get_args(literals[0]))


def _signature_without_name_type(fn: Callable[..., object]) -> inspect.Signature:
    sig = inspect.signature(fn)
    params = [p.replace(annotation=inspect.Parameter.empty) if p.name == 'name' else p for p in sig.parameters.values()]
    return sig.replace(parameters=params)


class TestPerBackendAutocomplete:
    """Each plugin class autocompletes only the ids its backend serves."""

    @pytest.mark.parametrize(
        ('plugin', 'method'),
        list(_BACKEND_IDS),
        ids=[f'{plugin.__name__}.{method}' for plugin, method in _BACKEND_IDS],
    )
    def test_name_param_offers_the_backend_ids(self, plugin: type, method: str) -> None:
        """The ``name`` annotation pins the exact id set for this backend and family."""
        assert _autocomplete_ids(getattr(plugin, method)) == _BACKEND_IDS[(plugin, method)]

    @pytest.mark.parametrize('method', ['gemini', 'gemma', 'veo'])
    def test_backends_share_the_signature_except_name(self, method: str) -> None:
        """Per-backend signatures differ only in the ``name`` annotation."""
        assert _signature_without_name_type(getattr(GoogleAI, method)) == _signature_without_name_type(
            getattr(VertexAI, method)
        )


class TestVirtualTryOnConstructor:
    """virtual_try_on lives on VertexAI only."""

    def test_mints_a_vertex_ref(self) -> None:
        """The ref carries the Vertex namespace and the try-on config schema."""
        ref = VertexAI.virtual_try_on('virtual-try-on-001')
        assert isinstance(ref, ModelRef)
        assert ref.name == 'vertexai/virtual-try-on-001'
        assert ref.config_schema is VirtualTryOnConfig

    @pytest.mark.parametrize('name', ['vertexai/virtual-try-on-001', 'models/virtual-try-on-001'])
    def test_strips_pasted_prefixes(self, name: str) -> None:
        """A pasted prefix resolves to the same ref as the bare id."""
        assert VertexAI.virtual_try_on(name).name == 'vertexai/virtual-try-on-001'

    @pytest.mark.parametrize('bad_id', ['gemini-2.5-flash', 'veo-3.0-generate-001', 'imagen-4.0-generate-001'])
    def test_rejects_other_families(self, bad_id: str) -> None:
        """Only virtual-try-on ids mint this ref."""
        with pytest.raises(GenkitError) as exc_info:
            VertexAI.virtual_try_on(bad_id)
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_not_offered_on_googleai(self) -> None:
        """Virtual Try-On has no Google AI backend, so it has no constructor there."""
        assert not hasattr(GoogleAI, 'virtual_try_on')


class TestNoImagenSurface:
    """Imagen has no export or module, and imagen only raises."""

    def test_no_imagen_exports(self) -> None:
        """The package exposes no Imagen name."""
        assert not {name for name in genkit_google_genai.__all__ if 'Imagen' in name}

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_imagen_raises_with_the_image_hint(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """imagen raises INVALID_ARGUMENT pointing at gemini_image."""
        cls = plugin.__name__
        with pytest.raises(GenkitError, match=rf'{cls}\.imagen: .* use {cls}\.gemini_image\(\)') as exc_info:
            plugin.imagen('imagen-4.0-generate-001')
        assert exc_info.value.status == 'INVALID_ARGUMENT'

    def test_no_imagen_module(self) -> None:
        """The Imagen model module is gone."""
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('genkit_google_genai._models.imagen')

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_every_constructor_rejects_imagen_ids(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """An imagen- id is refused by every other family constructor with the image hint."""
        for method in ('gemini', 'gemini_tts', 'gemma'):
            with pytest.raises(GenkitError, match=r'for image generation use \w+\.gemini_image\(\)'):
                getattr(plugin, method)('imagen-4.0-generate-001')
        with pytest.raises(GenkitError, match=r'for image generation use \w+\.gemini_image\(\)'):
            plugin.embedding('imagen-4.0-generate-001')

    @pytest.mark.parametrize('plugin', [GoogleAI, VertexAI])
    def test_gemini_image_does_not_suggest_itself(self, plugin: type[GoogleAI] | type[VertexAI]) -> None:
        """gemini_image refuses an imagen- id without pointing back at itself."""
        with pytest.raises(GenkitError) as exc_info:
            plugin.gemini_image('imagen-4.0-generate-001')
        assert 'is not a supported model' in str(exc_info.value)
        assert 'for image generation' not in str(exc_info.value)

    @pytest.mark.parametrize('bad_id', ['imagegeneration@006', 'imagetext@001'])
    def test_other_unsupported_ids_omit_the_image_hint(self, bad_id: str) -> None:
        """Only imagen- ids are redirected to image generation."""
        with pytest.raises(GenkitError) as exc_info:
            GoogleAI.gemini(bad_id)
        assert 'is not a supported model' in str(exc_info.value)
        assert 'gemini_image' not in str(exc_info.value)
