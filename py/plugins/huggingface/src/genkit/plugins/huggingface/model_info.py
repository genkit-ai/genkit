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

"""Hugging Face model information and metadata.

This module contains metadata for popular Hugging Face models.
Note: HF hosts 1M+ models, so this is just a curated list of popular ones.
Any model ID from huggingface.co can be used with this plugin.

See: https://huggingface.co/models
"""

from genkit.types import ModelInfo, Supports

__all__ = ['POPULAR_HUGGINGFACE_MODELS', 'get_default_model_info']

# Standard text model capabilities
# Note: Tool support depends on the specific model. Many popular models support tools.
_TEXT_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

# Vision-language model capabilities
_VISION_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=True,
    system_role=True,
    output=['text', 'json'],
)

# Code model capabilities
_CODE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

# Embedding model capabilities
_EMBEDDING_SUPPORTS = Supports(
    multiturn=False,
    tools=False,
    media=False,
    system_role=False,
    output=['text'],
)

# Popular models curated list - users can use any HF model ID
POPULAR_HUGGINGFACE_MODELS: dict[str, ModelInfo] = {
    # Meta Llama Models
    'meta-llama/Llama-3.3-70B-Instruct': ModelInfo(
        label='Meta - Llama 3.3 70B Instruct',
        versions=['meta-llama/Llama-3.3-70B-Instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'meta-llama/Llama-3.1-8B-Instruct': ModelInfo(
        label='Meta - Llama 3.1 8B Instruct',
        versions=['meta-llama/Llama-3.1-8B-Instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'meta-llama/Llama-3.1-70B-Instruct': ModelInfo(
        label='Meta - Llama 3.1 70B Instruct',
        versions=['meta-llama/Llama-3.1-70B-Instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Mistral Models (on HF)
    'mistralai/Mistral-Small-24B-Instruct-2501': ModelInfo(
        label='Mistral AI - Small 24B Instruct',
        versions=['mistralai/Mistral-Small-24B-Instruct-2501'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'mistralai/Mistral-7B-Instruct-v0.3': ModelInfo(
        label='Mistral AI - 7B Instruct v0.3',
        versions=['mistralai/Mistral-7B-Instruct-v0.3'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'mistralai/Mixtral-8x7B-Instruct-v0.1': ModelInfo(
        label='Mistral AI - Mixtral 8x7B Instruct',
        versions=['mistralai/Mixtral-8x7B-Instruct-v0.1'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Qwen Models
    'Qwen/Qwen2.5-72B-Instruct': ModelInfo(
        label='Qwen - 2.5 72B Instruct',
        versions=['Qwen/Qwen2.5-72B-Instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'Qwen/Qwen2.5-7B-Instruct': ModelInfo(
        label='Qwen - 2.5 7B Instruct',
        versions=['Qwen/Qwen2.5-7B-Instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'Qwen/Qwen2.5-Coder-32B-Instruct': ModelInfo(
        label='Qwen - 2.5 Coder 32B Instruct',
        versions=['Qwen/Qwen2.5-Coder-32B-Instruct'],
        supports=_CODE_MODEL_SUPPORTS,
    ),
    # DeepSeek Models (on HF)
    'deepseek-ai/DeepSeek-R1': ModelInfo(
        label='DeepSeek - R1 (Reasoning)',
        versions=['deepseek-ai/DeepSeek-R1'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'deepseek-ai/DeepSeek-V3': ModelInfo(
        label='DeepSeek - V3',
        versions=['deepseek-ai/DeepSeek-V3'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Google Models (on HF)
    'google/gemma-2-27b-it': ModelInfo(
        label='Google - Gemma 2 27B Instruct',
        versions=['google/gemma-2-27b-it'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    'google/gemma-2-9b-it': ModelInfo(
        label='Google - Gemma 2 9B Instruct',
        versions=['google/gemma-2-9b-it'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Microsoft Models
    'microsoft/Phi-3.5-mini-instruct': ModelInfo(
        label='Microsoft - Phi 3.5 Mini Instruct',
        versions=['microsoft/Phi-3.5-mini-instruct'],
        supports=_TEXT_MODEL_SUPPORTS,
    ),
    # Embedding Models
    'sentence-transformers/all-MiniLM-L6-v2': ModelInfo(
        label='Sentence Transformers - MiniLM L6 v2',
        versions=['sentence-transformers/all-MiniLM-L6-v2'],
        supports=_EMBEDDING_SUPPORTS,
    ),
    'BAAI/bge-large-en-v1.5': ModelInfo(
        label='BAAI - BGE Large EN v1.5',
        versions=['BAAI/bge-large-en-v1.5'],
        supports=_EMBEDDING_SUPPORTS,
    ),
}


def get_default_model_info(name: str) -> ModelInfo:
    """Get default model information for unknown Hugging Face models.

    Args:
        name: Model name/ID.

    Returns:
        Default ModelInfo with standard capabilities.
    """
    # Extract a readable label from the model ID
    parts = name.split('/')
    label = parts[-1] if len(parts) > 1 else name

    return ModelInfo(
        label=f'Hugging Face - {label}',
        supports=_TEXT_MODEL_SUPPORTS,
    )
