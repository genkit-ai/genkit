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

"""Model metadata for Cohere AI models.

This module defines the supported Cohere chat and embedding models
along with their capabilities and metadata.

Cross-checked against Cohere's official model documentation:

    - https://docs.cohere.com/docs/models
    - https://docs.cohere.com/docs/command-a
    - https://docs.cohere.com/docs/command-a-reasoning
    - https://docs.cohere.com/docs/command-a-translate
    - https://docs.cohere.com/docs/command-a-vision
    - https://docs.cohere.com/docs/command-r7b
    - https://docs.cohere.com/docs/command-r-plus

Note: ``command`` and ``command-light`` were removed by Cohere on
September 15, 2025 and are intentionally excluded.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelSupports(BaseModel):
    """Feature flags for a model."""

    multiturn: bool = False
    tools: bool = False
    media: bool = False
    system_role: bool = False
    output: list[str] | None = None


class ModelInfo(BaseModel):
    """Model metadata."""

    label: str
    supports: ModelSupports


# Cohere Command models — chat/generation.
# https://docs.cohere.com/docs/models#command
SUPPORTED_COHERE_MODELS: dict[str, ModelInfo] = {
    'command-a-03-2025': ModelInfo(
        label='Cohere - Command A',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-a-reasoning-08-2025': ModelInfo(
        label='Cohere - Command A Reasoning',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-a-translate-08-2025': ModelInfo(
        label='Cohere - Command A Translate',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-a-vision-07-2025': ModelInfo(
        label='Cohere - Command A Vision',
        supports=ModelSupports(
            multiturn=True,
            tools=False,
            media=True,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r7b-12-2024': ModelInfo(
        label='Cohere - Command R7B',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r-plus-08-2024': ModelInfo(
        label='Cohere - Command R+ (08-2024)',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r-plus-04-2024': ModelInfo(
        label='Cohere - Command R+ (04-2024)',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r-plus': ModelInfo(
        label='Cohere - Command R+',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r-08-2024': ModelInfo(
        label='Cohere - Command R (08-2024)',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r-03-2024': ModelInfo(
        label='Cohere - Command R (03-2024)',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
    'command-r': ModelInfo(
        label='Cohere - Command R',
        supports=ModelSupports(
            multiturn=True,
            tools=True,
            media=False,
            system_role=True,
            output=['text', 'json'],
        ),
    ),
}

# Cohere Embed models — text embeddings.
# https://docs.cohere.com/docs/models#embed
SUPPORTED_EMBEDDING_MODELS: dict[str, dict[str, object]] = {
    'embed-v4.0': {
        'label': 'Cohere - Embed v4.0',
        'dimensions': 1024,
    },
    'embed-english-v3.0': {
        'label': 'Cohere - Embed English v3.0',
        'dimensions': 1024,
    },
    'embed-english-light-v3.0': {
        'label': 'Cohere - Embed English Light v3.0',
        'dimensions': 384,
    },
    'embed-multilingual-v3.0': {
        'label': 'Cohere - Embed Multilingual v3.0',
        'dimensions': 1024,
    },
    'embed-multilingual-light-v3.0': {
        'label': 'Cohere - Embed Multilingual Light v3.0',
        'dimensions': 384,
    },
}


def get_default_model_info(name: str) -> ModelInfo:
    """Return a default ModelInfo for unrecognised model names.

    Args:
        name: The model name.

    Returns:
        A ModelInfo with conservative capability flags.
    """
    return ModelInfo(
        label=f'Cohere - {name}',
        supports=ModelSupports(
            multiturn=True,
            tools=False,
            media=False,
            system_role=True,
            output=['text'],
        ),
    )
