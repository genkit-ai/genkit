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

"""Tests for Google GenAI family model ref helpers."""

import pytest
from genkit_google_genai import (
    GeminiConfig,
    GeminiImageConfig,
    GeminiTtsConfig,
    GemmaConfig,
    ImagenConfig,
    LyriaConfig,
    VeoConfig,
    gemini_image_model,
    gemini_model,
    gemini_tts_model,
    gemma_model,
    imagen_model,
    lyria_model,
    veo_model,
)
from genkit_google_genai.models.imagen import is_imagen_model


@pytest.mark.parametrize(
    ('helper', 'unknown_name', 'schema'),
    [
        (gemini_model, 'gemini-flash-pro-whatever-99', GeminiConfig),
        (gemini_tts_model, 'gemini-9.9-flash-preview-tts', GeminiTtsConfig),
        (gemini_image_model, 'gemini-9.9-flash-image-preview', GeminiImageConfig),
        (gemma_model, 'gemma-9-99b-it', GemmaConfig),
        (imagen_model, 'imagen-99.0-generate-001', ImagenConfig),
        (veo_model, 'veo-9.9-generate-001', VeoConfig),
        (lyria_model, 'lyria-999', LyriaConfig),
    ],
)
def test_family_helper_unknown_version_still_typed_schema(
    helper: object,
    unknown_name: str,
    schema: type,
) -> None:
    """Unknown version strings still stamp the family config schema."""
    ref = helper(unknown_name)  # type: ignore[operator]
    assert ref.name == f'googleai/{unknown_name}'
    assert ref.config_schema is schema


def test_gemini_model_stamps_default_config() -> None:
    """Default config on the ref is preserved for later generate calls."""
    config = GeminiConfig.model_validate({'temperature': 0.2})
    ref = gemini_model('gemini-flash-latest', config=config)
    assert ref.config == config
    assert ref.config_schema is GeminiConfig


def test_namespace_idempotent_when_already_prefixed() -> None:
    """Namespace is not doubled when the name is already qualified."""
    ref = gemini_model('googleai/gemini-2.5-flash')
    assert ref.name == 'googleai/gemini-2.5-flash'


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('imagen-3.0-generate-002', True),
        ('imagen-4.0-generate-001', True),
        ('imagegeneration@006', True),
        ('gemini-2.5-flash-image', False),
        ('gemini-3-pro-image', False),
        ('gemini-2.5-flash', False),
    ],
)
def test_is_imagen_model(name: str, expected: bool) -> None:
    """Imagen detection uses imagen- / imagegeneration@ prefixes, not gemini- image models."""
    assert is_imagen_model(name) is expected


def test_gemini_image_models_are_not_imagen() -> None:
    """Gemini native image helpers stay separate from Imagen."""
    assert not is_imagen_model('gemini-2.5-flash-image')
    assert not is_imagen_model('gemini-3-pro-image')


def test_is_imagen_model_path_prefixes() -> None:
    """Verify is_imagen_model handles fully qualified path prefixes."""
    assert is_imagen_model('models/imagen-3.0-generate-002')
    assert is_imagen_model('googleai/imagen-3.0-generate-002')
    assert is_imagen_model('vertexai/imagegeneration@006')
