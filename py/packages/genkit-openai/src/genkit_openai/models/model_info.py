# Copyright 2025 Google LLC
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


"""OpenAI Compatible Models for Genkit."""

import sys
from typing import Literal, TypeAlias

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from genkit import (
    ModelInfo,
    Supports,
)
from genkit_openai.typing import SupportedOutputFormat

OPENAI = 'openai'
MODEL_GARDEN = 'model-garden'


class PluginSource(StrEnum):
    """Source of the plugin (OpenAI or Model Garden)."""

    OPENAI = 'openai'
    MODEL_GARDEN = 'model-garden'


MULTIMODAL_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.STRUCTURED_OUTPUTS, SupportedOutputFormat.TEXT],
)

GPT_4_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.TEXT],
)

GPT_35_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
)

O_SERIES_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=False,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
)

GPT_5_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
)

GPT_OSS_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
)

LLAMA_3_1 = 'meta/llama-3.1-405b-instruct-maas'
LLAMA_3_2 = 'meta/llama-3.2-90b-vision-instruct-maas'

# Quote autocomplete needs a Literal. The catalog below is what you edit when
# a chat model ships; a test requires these members and the dict keys to be the
# same set, and the dict is typed so a new key that is not in the Literal
# does not type-check. Image, TTS, STT, and embedding ids live in other
# catalogs — they do not take OpenAIConfig.
KnownGpt: TypeAlias = Literal[
    'gpt-4o',
    'gpt-4o-2024-05-13',
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18',
    'gpt-4.5-preview',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4-turbo',
    'gpt-4-turbo-2024-04-09',
    'gpt-4-turbo-preview',
    'gpt-4-0125-preview',
    'gpt-4-1106-preview',
    'gpt-4',
    'gpt-4-0613',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0125',
    'gpt-3.5-turbo-1106',
    'o1',
    'o3',
    'o3-mini',
    'o4-mini',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-chat-latest',
    'gpt-5.1',
    'gpt-5.2',
    'gpt-5.2-chat',
    'gpt-oss-120b',
    'gpt-oss-20b',
]


# Source: https://platform.openai.com/docs/models
# Codex and -pro ids (gpt-5.x-codex, o3-pro, gpt-5.x-pro) are served only by
# the Responses API; this plugin speaks Chat Completions, so they are absent.
SUPPORTED_OPENAI_MODELS: dict[KnownGpt, ModelInfo] = {
    # --- GPT-4o series ---
    'gpt-4o': ModelInfo(label='OpenAI - gpt-4o', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4o-2024-05-13': ModelInfo(label='OpenAI - gpt-4o-2024-05-13', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4o-mini': ModelInfo(label='OpenAI - gpt-4o-mini', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4o-mini-2024-07-18': ModelInfo(label='OpenAI - gpt-4o-mini-2024-07-18', supports=MULTIMODAL_MODEL_SUPPORTS),
    # --- GPT-4.x series ---
    'gpt-4.5-preview': ModelInfo(label='OpenAI - gpt-4.5-preview', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4.1': ModelInfo(label='OpenAI - gpt-4.1', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4.1-mini': ModelInfo(label='OpenAI - gpt-4.1-mini', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4-turbo': ModelInfo(label='OpenAI - gpt-4-turbo', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4-turbo-2024-04-09': ModelInfo(label='OpenAI - gpt-4-turbo-2024-04-09', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4-turbo-preview': ModelInfo(label='OpenAI - gpt-4-turbo-preview', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4-0125-preview': ModelInfo(label='OpenAI - gpt-4-0125-preview', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4-1106-preview': ModelInfo(label='OpenAI - gpt-4-1106-preview', supports=MULTIMODAL_MODEL_SUPPORTS),
    'gpt-4': ModelInfo(label='OpenAI - gpt-4', supports=GPT_4_MODEL_SUPPORTS),
    'gpt-4-0613': ModelInfo(label='OpenAI - gpt-4-0613', supports=GPT_4_MODEL_SUPPORTS),
    # --- GPT-3.5 series ---
    'gpt-3.5-turbo': ModelInfo(label='OpenAI - gpt-3.5-turbo', supports=GPT_35_MODEL_SUPPORTS),
    'gpt-3.5-turbo-0125': ModelInfo(label='OpenAI - gpt-3.5-turbo-0125', supports=GPT_35_MODEL_SUPPORTS),
    'gpt-3.5-turbo-1106': ModelInfo(label='OpenAI - gpt-3.5-turbo-1106', supports=GPT_35_MODEL_SUPPORTS),
    # --- O-series (reasoning) ---
    'o1': ModelInfo(label='OpenAI - o1', supports=O_SERIES_MODEL_SUPPORTS),
    'o3': ModelInfo(label='OpenAI - o3', supports=O_SERIES_MODEL_SUPPORTS),
    'o3-mini': ModelInfo(
        label='OpenAI - o3-mini',
        supports=Supports(
            multiturn=True,
            media=False,
            tools=True,
            system_role=False,
            output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
        ),
    ),
    'o4-mini': ModelInfo(label='OpenAI - o4-mini', supports=O_SERIES_MODEL_SUPPORTS),
    # --- GPT-5 series ---
    'gpt-5': ModelInfo(label='OpenAI - gpt-5', supports=GPT_5_MODEL_SUPPORTS),
    'gpt-5-mini': ModelInfo(label='OpenAI - gpt-5-mini', supports=GPT_5_MODEL_SUPPORTS),
    'gpt-5-nano': ModelInfo(label='OpenAI - gpt-5-nano', supports=GPT_5_MODEL_SUPPORTS),
    'gpt-5-chat-latest': ModelInfo(
        label='OpenAI - gpt-5-chat-latest',
        supports=Supports(
            multiturn=True,
            media=True,
            tools=False,
            system_role=True,
            output=[SupportedOutputFormat.TEXT],
        ),
    ),
    'gpt-5.1': ModelInfo(label='OpenAI - gpt-5.1', supports=GPT_5_MODEL_SUPPORTS),
    'gpt-5.2': ModelInfo(label='OpenAI - gpt-5.2', supports=GPT_5_MODEL_SUPPORTS),
    'gpt-5.2-chat': ModelInfo(label='OpenAI - gpt-5.2-chat', supports=GPT_5_MODEL_SUPPORTS),
    # --- OSS models (hosted) ---
    'gpt-oss-120b': ModelInfo(label='OpenAI - gpt-oss-120b', supports=GPT_OSS_MODEL_SUPPORTS),
    'gpt-oss-20b': ModelInfo(label='OpenAI - gpt-oss-20b', supports=GPT_OSS_MODEL_SUPPORTS),
}

SUPPORTED_EMBEDDING_MODELS: dict[str, dict] = {
    'text-embedding-3-small': {
        'label': 'OpenAI - text-embedding-3-small',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
    'text-embedding-3-large': {
        'label': 'OpenAI - text-embedding-3-large',
        'dimensions': 3072,
        'supports': {'input': ['text']},
    },
    'text-embedding-ada-002': {
        'label': 'OpenAI - text-embedding-ada-002',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
}

SUPPORTED_OPENAI_COMPAT_MODELS: dict[str, ModelInfo] = {
    LLAMA_3_1: ModelInfo(
        label='ModelGarden - Meta - llama-3.1',
        supports=Supports(
            multiturn=True,
            media=False,
            tools=True,
            system_role=True,
            long_running=False,
            output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
        ),
    ),
    LLAMA_3_2: ModelInfo(
        label='ModelGarden - Meta - llama-3.2',
        supports=Supports(
            multiturn=True,
            media=True,
            tools=True,
            system_role=True,
            output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
        ),
    ),
}


DEFAULT_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
)


def get_default_model_info(name: str) -> ModelInfo:
    """Gets the default model info given a name."""
    return ModelInfo(
        label=f'ModelGarden - {name}',
        supports=DEFAULT_SUPPORTS,
    )


def get_default_openai_model_info(name: str) -> ModelInfo:
    """Gets the default model info given a name."""
    return ModelInfo(label=f'OpenAI - {name}', supports=Supports(multiturn=True))
