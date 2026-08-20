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

"""Bucket a model id so resolve(MODEL) does not default it to Gemini.

Families with no generate path here resolve to nothing: Veo is
background-only, embedders are a different action kind, and retired or
unimplemented image ids must not fall through. Lyria, Deep Research, and
Antigravity fail closed until those families have a generate action here.
"""

from genkit_google_genai.models.gemini import (
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

# Prefixes people paste from action keys, Dev UI traces, or another plugin's
# samples. Stripping them first means the constructor decides the namespace,
# so GoogleAI.gemini_model('vertexai/gemini-2.5-flash') cannot smuggle a
# cross-plugin name into the ref.
STRIP_PREFIXES = (
    'background-model/',
    'model/',
    'models/',
    'embedders/',
    'googleai/',
    'vertexai/',
)

# Families with no MODEL generate path on this plugin. Constructor
# refusal is a different table — TTS, native image, Gemma, and Imagen
# still resolve as MODEL.
UNROUTABLE_FAMILIES = frozenset({
    'embedder',
    'unsupported',
    'veo',
    'lyria',
    'deep-research',
    'antigravity',
})


def strip_ref_prefixes(name: str) -> str:
    """Reduce a pasted name to the bare model id."""
    local = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in STRIP_PREFIXES:
            if local.startswith(prefix):
                local = local[len(prefix) :]
                changed = True
    return local


def classify_family(name: str) -> str:
    """Bucket a model id by the last path segment.

    The embedding check runs first because embedder ids can carry a family
    prefix (``gemini-embedding-001``) and must never mint a generate ref.
    """
    leaf = name.split('/')[-1].lower()
    if 'embedding' in leaf:
        return 'embedder'
    if is_unsupported_image_model_name(leaf):
        return 'unsupported'
    if is_veo_model(leaf):
        return 'veo'
    if is_lyria_model(leaf):
        return 'lyria'
    if leaf.startswith('deep-research-'):
        return 'deep-research'
    if leaf.startswith('antigravity-'):
        return 'antigravity'
    if is_imagen_model_name(leaf):
        return 'imagen'
    if is_tts_model(leaf):
        return 'tts'
    if is_image_model(leaf):
        return 'image'
    if is_gemma_model(leaf):
        return 'gemma'
    if leaf.startswith('gemini-'):
        return 'gemini'
    return 'unknown'


def is_unroutable_model_id(name: str) -> bool:
    """True for ids that resolve(MODEL) must refuse instead of defaulting to Gemini."""
    return classify_family(name) in UNROUTABLE_FAMILIES
