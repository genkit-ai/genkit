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

"""Mistral AI model information and metadata.

This module contains metadata for all supported Mistral AI models,
including capabilities and version information.

Model catalog and capabilities are sourced from:
- https://docs.mistral.ai/getting-started/models/
- https://docs.mistral.ai/getting-started/models/compare
"""

from genkit.types import ModelInfo, Supports

__all__ = ['SUPPORTED_MISTRAL_MODELS', 'get_default_model_info']

# Standard text model capabilities (no vision).
_TEXT_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

# Vision model capabilities (Pixtral, Mistral Large 3, Medium 3.1, Small 3.2, Ministral 3).
_VISION_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=True,
    system_role=True,
    output=['text', 'json'],
)

# Code model capabilities (Codestral, Devstral).
_CODE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

# Embedding model capabilities (mistral-embed, codestral-embed).
_EMBEDDING_SUPPORTS = Supports(
    multiturn=False,
    tools=False,
    media=False,
    system_role=False,
    output=['text'],
)

# Audio model capabilities (Voxtral — chat with audio input).
_AUDIO_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=False,
    media=True,
    system_role=True,
    output=['text'],
)

SUPPORTED_MISTRAL_MODELS: dict[str, ModelInfo] = {
    # Mistral Large 3 — state-of-the-art, open-weight, multimodal (vision).
    # https://docs.mistral.ai/models/mistral-large-3-25-12
    'mistral-large-latest': ModelInfo(
        label='Mistral AI - Large 3 (Latest)',
        versions=['mistral-large-latest', 'mistral-large-2512'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Mistral Medium 3.1 — frontier-class multimodal (vision).
    # https://docs.mistral.ai/models/mistral-medium-3-1-25-08
    'mistral-medium-latest': ModelInfo(
        label='Mistral AI - Medium 3.1 (Latest)',
        versions=['mistral-medium-latest', 'mistral-medium-2508'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Mistral Small 3.2 — compact multimodal with vision.
    # https://docs.mistral.ai/models/mistral-small-3-2-25-06
    'mistral-small-latest': ModelInfo(
        label='Mistral AI - Small 3.2 (Latest)',
        versions=['mistral-small-latest', 'mistral-small-2506'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Ministral 3 14B — best-in-class text and vision capabilities.
    # https://docs.mistral.ai/models/ministral-3-14b-25-12
    'ministral-14b-latest': ModelInfo(
        label='Mistral AI - Ministral 3 14B',
        versions=['ministral-14b-latest', 'ministral-14b-2512'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Ministral 3 8B — powerful and efficient text and vision.
    # https://docs.mistral.ai/models/ministral-3-8b-25-12
    'ministral-8b-latest': ModelInfo(
        label='Mistral AI - Ministral 3 8B',
        versions=['ministral-8b-latest', 'ministral-8b-2512'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Ministral 3 3B — tiny and efficient text and vision.
    # https://docs.mistral.ai/models/ministral-3-3b-25-12
    'ministral-3b-latest': ModelInfo(
        label='Mistral AI - Ministral 3 3B',
        versions=['ministral-3b-latest', 'ministral-3b-2512'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Magistral Medium 1.2 — frontier-class multimodal reasoning.
    # https://docs.mistral.ai/models/magistral-medium-1-2-25-09
    'magistral-medium-latest': ModelInfo(
        label='Mistral AI - Magistral Medium 1.2 (Reasoning)',
        versions=['magistral-medium-latest', 'magistral-medium-2509'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Magistral Small 1.2 — small multimodal reasoning.
    # https://docs.mistral.ai/models/magistral-small-1-2-25-09
    'magistral-small-latest': ModelInfo(
        label='Mistral AI - Magistral Small 1.2 (Reasoning)',
        versions=['magistral-small-latest', 'magistral-small-2509'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Codestral — cutting-edge code completion.
    # https://docs.mistral.ai/models/codestral-25-08
    'codestral-latest': ModelInfo(
        label='Mistral AI - Codestral (Latest)',
        versions=['codestral-latest', 'codestral-2508'],
        supports=_CODE_MODEL_SUPPORTS,
    ),
    # Devstral 2 — frontier code agent model for SWE tasks (123B dense).
    # https://docs.mistral.ai/models/devstral-2-25-12
    'devstral-latest': ModelInfo(
        label='Mistral AI - Devstral 2 (Code Agent)',
        versions=['devstral-latest', 'devstral-2512'],
        supports=_CODE_MODEL_SUPPORTS,
    ),
    # Devstral Small 2 — smaller code agent model (24B, Labs).
    'devstral-small-latest': ModelInfo(
        label='Mistral AI - Devstral Small 2 (Code Agent)',
        versions=['devstral-small-latest', 'devstral-small-2512'],
        supports=_CODE_MODEL_SUPPORTS,
    ),
    # Voxtral Small — audio input for chat use cases.
    # https://docs.mistral.ai/models/voxtral-small-25-07
    'voxtral-small-latest': ModelInfo(
        label='Mistral AI - Voxtral Small (Audio)',
        versions=['voxtral-small-latest'],
        supports=_AUDIO_MODEL_SUPPORTS,
    ),
    # Voxtral Mini — mini audio input model for chat.
    # https://docs.mistral.ai/models/voxtral-mini-25-07
    'voxtral-mini-latest': ModelInfo(
        label='Mistral AI - Voxtral Mini (Audio)',
        versions=['voxtral-mini-latest'],
        supports=_AUDIO_MODEL_SUPPORTS,
    ),
    # Mistral Embed — 1024-dimensional text embeddings for RAG and search.
    # https://docs.mistral.ai/models/mistral-embed-23-12
    'mistral-embed': ModelInfo(
        label='Mistral AI - Embed',
        versions=['mistral-embed'],
        supports=_EMBEDDING_SUPPORTS,
    ),
    # Codestral Embed — semantic code embeddings.
    # No -latest alias; use the dated version directly.
    # https://docs.mistral.ai/models/codestral-embed-25-05
    'codestral-embed-2505': ModelInfo(
        label='Mistral AI - Codestral Embed',
        versions=['codestral-embed-2505'],
        supports=_EMBEDDING_SUPPORTS,
    ),
    # Mistral Small Creative — creative writing and character interaction.
    'mistral-small-creative-latest': ModelInfo(
        label='Mistral AI - Small Creative',
        versions=['mistral-small-creative-latest'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Mistral Saba — regional / research model.
    'mistral-saba-latest': ModelInfo(
        label='Mistral AI - Saba (Latest)',
        versions=['mistral-saba-latest'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Pixtral Large — legacy vision model (superseded by Large 3 / Medium 3.1).
    'pixtral-large-latest': ModelInfo(
        label='Mistral AI - Pixtral Large (Vision)',
        versions=['pixtral-large-latest', 'pixtral-large-2411'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    'pixtral-12b-latest': ModelInfo(
        label='Mistral AI - Pixtral 12B (Vision)',
        versions=['pixtral-12b-latest', 'pixtral-12b-2409'],
        supports=_VISION_MODEL_SUPPORTS,
    ),
    # Legacy open-weight models.
    'open-mistral-nemo': ModelInfo(
        label='Mistral AI - Nemo 12B (Open)',
        versions=['open-mistral-nemo', 'open-mistral-nemo-2407'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'open-codestral-mamba': ModelInfo(
        label='Mistral AI - Codestral Mamba (Open)',
        versions=['open-codestral-mamba'],
        supports=_CODE_MODEL_SUPPORTS,
    ),
}

# Names of models that are used for OCR (separate API endpoint).
OCR_MODEL_NAMES: frozenset[str] = frozenset({
    'mistral-ocr-latest',
    'mistral-ocr-2505',
    'mistral-ocr-2512',
})

# Names of models that are used for audio transcription (separate API endpoint).
TRANSCRIPTION_MODEL_NAMES: frozenset[str] = frozenset({
    'voxtral-mini-latest',
    'voxtral-mini-2507',
    'voxtral-mini-2602',
})


def get_default_model_info(name: str) -> ModelInfo:
    """Get default model information for unknown Mistral models.

    Args:
        name: Model name.

    Returns:
        Default ModelInfo with standard Mistral capabilities.
    """
    return ModelInfo(
        label=f'Mistral AI - {name}',
        supports=_TEXT_MODEL_SUPPORTS,
    )
