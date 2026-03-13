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

"""Model information for Cloudflare Workers AI models.

This module contains metadata for all supported Cloudflare AI models,
including capabilities and version information.

See: https://developers.cloudflare.com/workers-ai/models/
"""

from genkit.core.typing import ModelInfo, Supports

# Default model capabilities for text generation
_DEFAULT_TEXT_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,  # Some Cloudflare models support tool_calls
    system_role=True,
    output=['text', 'json'],
)

# Multimodal model capabilities (e.g., Llama 4 Scout)
_MULTIMODAL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Embedding model capabilities
_EMBEDDING_SUPPORTS = Supports(
    multiturn=False,
    media=False,
    tools=False,
    system_role=False,
    output=['text'],
)


SUPPORTED_CF_MODELS: dict[str, ModelInfo] = {
    # Meta Llama 3.3 Models
    '@cf/meta/llama-3.3-70b-instruct-fp8-fast': ModelInfo(
        label='Meta - Llama 3.3 70B Instruct FP8 Fast',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3.3-70b-instruct-fp8-fast'],
    ),
    # Meta Llama 3.1 Models
    '@cf/meta/llama-3.1-8b-instruct': ModelInfo(
        label='Meta - Llama 3.1 8B Instruct',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3.1-8b-instruct'],
    ),
    '@cf/meta/llama-3.1-8b-instruct-fast': ModelInfo(
        label='Meta - Llama 3.1 8B Instruct Fast',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3.1-8b-instruct-fast'],
    ),
    '@cf/meta/llama-3.1-8b-instruct-fp8': ModelInfo(
        label='Meta - Llama 3.1 8B Instruct FP8',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3.1-8b-instruct-fp8'],
    ),
    '@cf/meta/llama-3.1-70b-instruct': ModelInfo(
        label='Meta - Llama 3.1 70B Instruct',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3.1-70b-instruct'],
    ),
    # Meta Llama 4 Models (Multimodal)
    '@cf/meta/llama-4-scout-17b-16e-instruct': ModelInfo(
        label='Meta - Llama 4 Scout 17B Instruct (Multimodal)',
        supports=_MULTIMODAL_SUPPORTS,
        versions=['@cf/meta/llama-4-scout-17b-16e-instruct'],
    ),
    # Meta Llama 3 Models
    '@cf/meta/llama-3-8b-instruct': ModelInfo(
        label='Meta - Llama 3 8B Instruct',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/meta/llama-3-8b-instruct'],
    ),
    # Mistral Models
    # Note: Mistral 7B v0.2 uses @hf/ prefix (Hugging Face hosted),
    # while newer Mistral models like Small 3.1 use @cf/ prefix
    '@hf/mistral/mistral-7b-instruct-v0.2': ModelInfo(
        label='Mistral AI - Mistral 7B Instruct v0.2',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@hf/mistral/mistral-7b-instruct-v0.2'],
    ),
    '@cf/mistral/mistral-small-3.1-24b-instruct': ModelInfo(
        label='Mistral AI - Mistral Small 3.1 24B Instruct',
        supports=_MULTIMODAL_SUPPORTS,  # Has vision capabilities
        versions=['@cf/mistral/mistral-small-3.1-24b-instruct'],
    ),
    # Qwen Models
    '@cf/qwen/qwen1.5-14b-chat-awq': ModelInfo(
        label='Qwen - Qwen 1.5 14B Chat AWQ',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/qwen/qwen1.5-14b-chat-awq'],
    ),
    '@cf/qwen/qwen1.5-7b-chat-awq': ModelInfo(
        label='Qwen - Qwen 1.5 7B Chat AWQ',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/qwen/qwen1.5-7b-chat-awq'],
    ),
    '@cf/qwen/qwen3-30b-a3b-fp8': ModelInfo(
        label='Qwen - Qwen 3 30B A3B FP8',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/qwen/qwen3-30b-a3b-fp8'],
    ),
    '@cf/qwen/qwq-32b': ModelInfo(
        label='Qwen - QwQ 32B (Reasoning)',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/qwen/qwq-32b'],
    ),
    '@cf/qwen/qwen2.5-coder-32b-instruct': ModelInfo(
        label='Qwen - Qwen 2.5 Coder 32B Instruct',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/qwen/qwen2.5-coder-32b-instruct'],
    ),
    # Google Gemma Models
    # Note: Gemma 7B uses @hf/ prefix (Hugging Face hosted),
    # while Gemma 3 uses @cf/ prefix
    '@cf/google/gemma-3-12b-it': ModelInfo(
        label='Google - Gemma 3 12B Instruct (Multimodal)',
        supports=_MULTIMODAL_SUPPORTS,  # Gemma 3 supports images
        versions=['@cf/google/gemma-3-12b-it'],
    ),
    '@hf/google/gemma-7b-it': ModelInfo(
        label='Google - Gemma 7B Instruct',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@hf/google/gemma-7b-it'],
    ),
    # Microsoft Phi Models
    '@cf/microsoft/phi-2': ModelInfo(
        label='Microsoft - Phi 2',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/microsoft/phi-2'],
    ),
    # DeepSeek Models
    '@cf/deepseek-ai/deepseek-r1-distill-qwen-32b': ModelInfo(
        label='DeepSeek - R1 Distill Qwen 32B',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=['@cf/deepseek-ai/deepseek-r1-distill-qwen-32b'],
    ),
}


SUPPORTED_EMBEDDING_MODELS: dict[str, dict[str, object]] = {
    '@cf/baai/bge-base-en-v1.5': {
        'label': 'BAAI - BGE Base EN v1.5',
        'dimensions': 768,
        'supports': {'input': ['text']},
    },
    '@cf/baai/bge-large-en-v1.5': {
        'label': 'BAAI - BGE Large EN v1.5',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
    '@cf/baai/bge-small-en-v1.5': {
        'label': 'BAAI - BGE Small EN v1.5',
        'dimensions': 384,
        'supports': {'input': ['text']},
    },
    '@cf/google/embeddinggemma-300m': {
        'label': 'Google - EmbeddingGemma 300M',
        'dimensions': 768,
        'supports': {'input': ['text']},
    },
    '@cf/qwen/qwen3-embedding-0.6b': {
        'label': 'Qwen - Qwen 3 Embedding 0.6B',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
}


def get_model_info(model_id: str) -> ModelInfo:
    """Get model information for a given model ID.

    Args:
        model_id: The Cloudflare model ID (e.g., '@cf/meta/llama-3.1-8b-instruct').

    Returns:
        ModelInfo with capabilities, or a default ModelInfo if not found.
    """
    if model_id in SUPPORTED_CF_MODELS:
        return SUPPORTED_CF_MODELS[model_id]

    # Default for unknown models
    return ModelInfo(
        label=f'Cloudflare - {model_id}',
        supports=_DEFAULT_TEXT_SUPPORTS,
        versions=[model_id],
    )


__all__ = [
    'SUPPORTED_CF_MODELS',
    'SUPPORTED_EMBEDDING_MODELS',
    'get_model_info',
]
