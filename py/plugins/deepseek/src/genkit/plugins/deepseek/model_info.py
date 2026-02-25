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

"""DeepSeek model information and metadata.

Defines supported models and their capabilities. Reasoning models
(deepseek-reasoner, deepseek-r1) have tools disabled because the
DeepSeek API ignores tool parameters for these models.
"""

from genkit.types import ModelInfo, Supports

__all__ = [
    'SUPPORTED_DEEPSEEK_MODELS',
    'get_default_model_info',
    'is_reasoning_model',
]

# Chat model capabilities (supports tools, structured output).
_DEEPSEEK_CHAT_SUPPORTS = Supports(
    multiturn=True,
    tools=True,
    media=False,
    system_role=True,
    output=['text', 'json'],
)

# Reasoning model capabilities (no tools — R1 ignores them silently).
_DEEPSEEK_REASONING_SUPPORTS = Supports(
    multiturn=True,
    tools=False,
    media=False,
    system_role=True,
    output=['text'],
)

# Names of reasoning models that return reasoning_content and
# silently ignore temperature, top_p, and tool parameters.
_REASONING_MODEL_NAMES: frozenset[str] = frozenset({
    'deepseek-reasoner',
    'deepseek-r1',
})

SUPPORTED_DEEPSEEK_MODELS: dict[str, ModelInfo] = {
    'deepseek-reasoner': ModelInfo(
        label='DeepSeek - Reasoner',
        versions=['deepseek-reasoner'],
        supports=_DEEPSEEK_REASONING_SUPPORTS,
    ),
    'deepseek-chat': ModelInfo(
        label='DeepSeek - Chat',
        versions=['deepseek-chat'],
        supports=_DEEPSEEK_CHAT_SUPPORTS,
    ),
    'deepseek-v3': ModelInfo(
        label='DeepSeek - V3',
        versions=['deepseek-v3'],
        supports=_DEEPSEEK_CHAT_SUPPORTS,
    ),
    'deepseek-v4': ModelInfo(
        label='DeepSeek - V4',
        versions=['deepseek-v4'],
        supports=_DEEPSEEK_CHAT_SUPPORTS,
    ),
    'deepseek-r1': ModelInfo(
        label='DeepSeek - R1',
        versions=['deepseek-r1'],
        supports=_DEEPSEEK_REASONING_SUPPORTS,
    ),
}


def is_reasoning_model(name: str) -> bool:
    """Check if the model is a reasoning model.

    Reasoning models (R1, reasoner) return chain-of-thought in a
    separate ``reasoning_content`` field and silently ignore parameters
    like ``temperature``, ``top_p``, and ``tools``.

    Args:
        name: The model name (with or without the plugin prefix).

    Returns:
        True if the model is a reasoning model.
    """
    # Strip plugin prefix if present.
    clean = name.split('/', 1)[-1] if '/' in name else name
    return clean in _REASONING_MODEL_NAMES


def get_default_model_info(name: str) -> ModelInfo:
    """Get default model information for unknown DeepSeek models.

    Args:
        name: Model name.

    Returns:
        Default ModelInfo with standard DeepSeek capabilities.
    """
    supports = _DEEPSEEK_REASONING_SUPPORTS if is_reasoning_model(name) else _DEEPSEEK_CHAT_SUPPORTS
    return ModelInfo(
        label=f'DeepSeek - {name}',
        supports=supports,
    )
