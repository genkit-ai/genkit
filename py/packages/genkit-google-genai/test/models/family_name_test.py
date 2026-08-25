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

"""Family name checks for Google model routing."""

import pytest
from genkit_google_genai.models._routing import classify_family, is_unroutable_model_id
from genkit_google_genai.models.gemini import (
    is_gemini_model,
    is_gemma_model,
    is_image_model,
    is_tts_model,
)
from genkit_google_genai.models.imagen import (
    is_imagen_model_name,
    is_unsupported_image_model_name,
)
from genkit_google_genai.models.lyria import is_lyria_model
from genkit_google_genai.models.veo import is_veo_model


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('imagen-3.0-generate-002', True),
        ('IMAGEN-3.0-generate-002', True),
        ('googleai/imagen-3.0-generate-002', True),
        ('vertexai/imagen-3.0-generate-002', True),
        ('models/imagen-3.0-generate-002', True),
        ('gemini-2.5-flash-image', False),
        ('gemini-2.5-flash-image-preview', False),
        ('imagegeneration@006', False),
        ('virtual-try-on-001', False),
        ('veo-3.0-generate-001', False),
    ],
)
def test_is_imagen_model_name(name: str, expected: bool) -> None:
    """Imagen is the ``imagen-`` prefix only, on both plugins."""
    assert is_imagen_model_name(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('imagegeneration@006', True),
        ('imagegeneration@005', True),
        ('vertexai/imagegeneration@006', True),
        ('imagetext@001', True),
        ('vertexai/imagetext@001', True),
        ('virtual-try-on-001', True),
        ('vertexai/virtual-try-on-001', True),
        ('imagen-3.0-generate-002', False),
        ('gemini-2.5-flash-image', False),
        ('gemini-2.5-flash', False),
    ],
)
def test_is_unsupported_image_model_name(name: str, expected: bool) -> None:
    """Retired / unimplemented image ids fail closed instead of routing to Gemini."""
    assert is_unsupported_image_model_name(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gemini-2.5-flash-preview-tts', True),
        ('GEMINI-2.5-FLASH-PREVIEW-TTS', True),
        ('googleai/gemini-2.5-pro-preview-tts', True),
        ('gemini-2.5-flash', False),
        ('gemini-2.5-flash-image', False),
    ],
)
def test_is_tts_model(name: str, expected: bool) -> None:
    """TTS is ``gemini-`` plus ``-tts`` on the local name."""
    assert is_tts_model(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gemini-2.5-flash-image', True),
        ('GEMINI-2.5-FLASH-IMAGE', True),
        ('gemini-2.0-flash-preview-image-generation', True),
        ('googleai/gemini-3-pro-image-preview', True),
        ('imagen-3.0-generate-002', False),
        ('gemini-2.5-flash', False),
        ('gemini-2.5-flash-preview-tts', False),
    ],
)
def test_is_image_model(name: str, expected: bool) -> None:
    """Gemini native image is ``gemini-`` plus ``-image``."""
    assert is_image_model(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gemma-2-27b-it', True),
        ('GEMMA-3-12B-IT', True),
        ('googleai/gemma-3-12b-it', True),
        ('gemini-2.5-flash', False),
    ],
)
def test_is_gemma_model(name: str, expected: bool) -> None:
    """Gemma is the ``gemma-`` prefix on the local name."""
    assert is_gemma_model(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('gemini-2.0-flash-001', True),
        ('GEMINI-2.0-FLASH-001', True),
        ('googleai/gemini-2.5-flash', True),
        ('gemini-2.5-flash-preview-tts', False),
        ('gemini-2.5-flash-image', False),
        ('gemma-2-27b-it', False),
    ],
)
def test_is_gemini_model(name: str, expected: bool) -> None:
    """Standard Gemini excludes TTS and native image variants."""
    assert is_gemini_model(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('veo-3.0-generate-001', True),
        ('googleai/veo-3.1-generate-preview', True),
        ('VEO-2.0-generate-001', True),
        ('gemini-2.0-flash', False),
        ('devotional-hymn', False),
    ],
)
def test_is_veo_model_local_and_namespaced(name: str, expected: bool) -> None:
    """Veo is the ``veo-`` prefix after stripping a namespace."""
    assert is_veo_model(name) is expected


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('lyria-002', True),
        ('LYRIA-002', True),
        ('vertexai/lyria-002', True),
        ('gemini-2.5-flash', False),
    ],
)
def test_is_lyria_model(name: str, expected: bool) -> None:
    """Lyria is the ``lyria-`` prefix on the local name."""
    assert is_lyria_model(name) is expected


@pytest.mark.parametrize(
    'name',
    [
        'lyria-002',
        'LYRIA-002',
        'deep-research-pro-preview',
        'antigravity-code-1',
        'gemini-embedding-001',
        'imagegeneration@006',
        'imagetext@001',
        'virtual-try-on-001',
        'veo-3.0-generate-001',
        'googleai/lyria-002',
        'models/deep-research-pro-preview',
        'googleai/deep-research-pro-preview',
        'publishers/google/models/deep-research-pro-preview',
        'publishers/google/models/antigravity-preview-05-2026',
        'publishers/google/models/imagetext@001',
    ],
)
def test_unroutable_ids_fail_closed(name: str) -> None:
    """Ids with no generate path here must not default to Gemini."""
    assert is_unroutable_model_id(name) is True


@pytest.mark.parametrize(
    ('name', 'family'),
    [
        ('deep-research-pro-preview', 'deep-research'),
        ('models/deep-research-pro-preview', 'deep-research'),
        ('googleai/deep-research-pro-preview', 'deep-research'),
        ('publishers/google/models/deep-research-pro-preview', 'deep-research'),
        ('antigravity-preview-05-2026', 'antigravity'),
        ('publishers/google/models/antigravity-preview-05-2026', 'antigravity'),
        ('lyria-002', 'lyria'),
        ('imagetext@001', 'unsupported'),
    ],
)
def test_classify_family_uses_leaf_name(name: str, family: str) -> None:
    """Deep Research and Antigravity are separate buckets, keyed off the leaf."""
    assert classify_family(name) == family


@pytest.mark.parametrize(
    'name',
    [
        'gemini-2.5-flash',
        'GEMINI-2.5-FLASH-PREVIEW-TTS',
        'imagen-3.0-generate-002',
        'gemma-3-12b-it',
    ],
)
def test_routable_ids_are_not_unroutable(name: str) -> None:
    """Generate families still resolve as MODEL."""
    assert is_unroutable_model_id(name) is False
