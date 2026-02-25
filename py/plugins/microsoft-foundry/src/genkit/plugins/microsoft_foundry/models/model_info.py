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

"""Microsoft Foundry model info.

This module defines the supported models for the Microsoft Foundry plugin.
Microsoft Foundry provides access to a comprehensive catalog of AI models
from multiple providers.

See: https://ai.azure.com/catalog/models

Supported Model Categories
==========================
+----------------+--------------------------------------------------+
| Category       | Models                                           |
+----------------+--------------------------------------------------+
| GPT Series     | gpt-4, gpt-4o, gpt-4.1, gpt-5, gpt-5.1, gpt-5.2, gpt-5.3 |
| O-Series       | o1, o3, o3-mini, o4-mini                         |
| Claude         | claude-opus-4-6, claude-opus-4-5, claude-sonnet-4-5 |
| DeepSeek       | DeepSeek-V3.2, DeepSeek-R1-0528                  |
| Grok           | grok-4, grok-3, grok-3-mini                      |
| Llama          | Llama-4-Maverick-17B-128E-Instruct-FP8           |
| Mistral        | Mistral-Large-3                                  |
| Embeddings     | text-embedding-3-small/large, embed-v-4-0        |
+----------------+--------------------------------------------------+

Note: This is a subset of the 11,000+ models available in the catalog.
Any model can be used dynamically via the microsoft_foundry_model() function.
"""

from genkit.types import ModelInfo, Supports

# Model capability definitions
MULTIMODAL_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

TEXT_ONLY_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# O-series and reasoning models typically output text only
REASONING_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text'],
)

REASONING_MODEL_MINI_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text'],
)

# Claude models support (via Azure AI Foundry's Anthropic integration)
CLAUDE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# DeepSeek models support
DEEPSEEK_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Grok models support
GROK_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Llama models support
LLAMA_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Mistral models support
MISTRAL_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

SUPPORTED_MICROSOFT_FOUNDRY_MODELS: dict[str, ModelInfo] = {
    # =========================================================================
    # OpenAI GPT Series
    # =========================================================================
    # GPT-4 Series
    'gpt-4o': ModelInfo(
        label='Microsoft Foundry - GPT-4o',
        versions=['gpt-4o'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4o-mini': ModelInfo(
        label='Microsoft Foundry - GPT-4o Mini',
        versions=['gpt-4o-mini'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4': ModelInfo(
        label='Microsoft Foundry - GPT-4',
        versions=['gpt-4', 'gpt-4-32k'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4.5': ModelInfo(
        label='Microsoft Foundry - GPT-4.5',
        versions=['gpt-4.5-preview'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4.1': ModelInfo(
        label='Microsoft Foundry - GPT-4.1',
        versions=['gpt-4.1'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4.1-mini': ModelInfo(
        label='Microsoft Foundry - GPT-4.1 Mini',
        versions=['gpt-4.1-mini'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-4.1-nano': ModelInfo(
        label='Microsoft Foundry - GPT-4.1 Nano',
        versions=['gpt-4.1-nano'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    # GPT-3.5 Series
    'gpt-3.5-turbo': ModelInfo(
        label='Microsoft Foundry - GPT-3.5 Turbo',
        versions=['gpt-3.5-turbo', 'gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-16k'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # OpenAI O-Series (Reasoning models)
    # =========================================================================
    'o1': ModelInfo(
        label='Microsoft Foundry - o1',
        versions=['o1'],
        supports=Supports(multiturn=True, media=True, tools=False, system_role=True, output=['text']),
    ),
    'o1-mini': ModelInfo(
        label='Microsoft Foundry - o1 Mini',
        versions=['o1-mini'],
        supports=REASONING_MODEL_MINI_SUPPORTS,
    ),
    'o1-preview': ModelInfo(
        label='Microsoft Foundry - o1 Preview',
        versions=['o1-preview'],
        supports=REASONING_MODEL_SUPPORTS,
    ),
    'o3': ModelInfo(
        label='Microsoft Foundry - o3',
        versions=['o3'],
        supports=REASONING_MODEL_SUPPORTS,
    ),
    'o3-mini': ModelInfo(
        label='Microsoft Foundry - o3 Mini',
        versions=['o3-mini'],
        supports=REASONING_MODEL_MINI_SUPPORTS,
    ),
    'o3-pro': ModelInfo(
        label='Microsoft Foundry - o3 Pro',
        versions=['o3-pro'],
        supports=REASONING_MODEL_SUPPORTS,
    ),
    'o4-mini': ModelInfo(
        label='Microsoft Foundry - o4 Mini',
        versions=['o4-mini'],
        supports=REASONING_MODEL_SUPPORTS,
    ),
    'codex-mini': ModelInfo(
        label='Microsoft Foundry - Codex Mini',
        versions=['codex-mini'],
        supports=REASONING_MODEL_MINI_SUPPORTS,
    ),
    # =========================================================================
    # OpenAI GPT-5 Series
    # =========================================================================
    'gpt-5': ModelInfo(
        label='Microsoft Foundry - GPT-5',
        versions=['gpt-5'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5-mini': ModelInfo(
        label='Microsoft Foundry - GPT-5 Mini',
        versions=['gpt-5-mini'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5-nano': ModelInfo(
        label='Microsoft Foundry - GPT-5 Nano',
        versions=['gpt-5-nano'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5-chat': ModelInfo(
        label='Microsoft Foundry - GPT-5 Chat',
        versions=['gpt-5-chat'],
        supports=Supports(multiturn=True, media=True, tools=True, system_role=True, output=['text']),
    ),
    'gpt-5-codex': ModelInfo(
        label='Microsoft Foundry - GPT-5 Codex',
        versions=['gpt-5-codex'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5-pro': ModelInfo(
        label='Microsoft Foundry - GPT-5 Pro',
        versions=['gpt-5-pro'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.1': ModelInfo(
        label='Microsoft Foundry - GPT-5.1',
        versions=['gpt-5.1'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.1-chat': ModelInfo(
        label='Microsoft Foundry - GPT-5.1 Chat',
        versions=['gpt-5.1-chat'],
        supports=Supports(multiturn=True, media=False, tools=True, system_role=True, output=['text', 'json']),
    ),
    'gpt-5.1-codex': ModelInfo(
        label='Microsoft Foundry - GPT-5.1 Codex',
        versions=['gpt-5.1-codex'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.1-codex-mini': ModelInfo(
        label='Microsoft Foundry - GPT-5.1 Codex Mini',
        versions=['gpt-5.1-codex-mini'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.1-codex-max': ModelInfo(
        label='Microsoft Foundry - GPT-5.1 Codex Max',
        versions=['gpt-5.1-codex-max'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.2': ModelInfo(
        label='Microsoft Foundry - GPT-5.2',
        versions=['gpt-5.2'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.2-chat': ModelInfo(
        label='Microsoft Foundry - GPT-5.2 Chat',
        versions=['gpt-5.2-chat'],
        supports=Supports(multiturn=True, media=False, tools=True, system_role=True, output=['text', 'json']),
    ),
    'gpt-5.2-codex': ModelInfo(
        label='Microsoft Foundry - GPT-5.2 Codex',
        versions=['gpt-5.2-codex'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.2-pro': ModelInfo(
        label='Microsoft Foundry - GPT-5.2 Pro',
        versions=['gpt-5.2-pro'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-5.3-codex': ModelInfo(
        label='Microsoft Foundry - GPT-5.3 Codex',
        versions=['gpt-5.3-codex'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'gpt-oss-120B': ModelInfo(
        label='Microsoft Foundry - GPT-OSS 120B',
        versions=['gpt-oss-120B'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'gpt-oss-20b': ModelInfo(
        label='Microsoft Foundry - GPT-OSS 20B',
        versions=['gpt-oss-20b'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Anthropic Claude Models (via Microsoft Foundry)
    # =========================================================================
    'claude-opus-4-5': ModelInfo(
        label='Microsoft Foundry - Claude Opus 4.5',
        versions=['claude-opus-4-5'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'claude-sonnet-4-5': ModelInfo(
        label='Microsoft Foundry - Claude Sonnet 4.5',
        versions=['claude-sonnet-4-5'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'claude-haiku-4-5': ModelInfo(
        label='Microsoft Foundry - Claude Haiku 4.5',
        versions=['claude-haiku-4-5'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'claude-opus-4-1': ModelInfo(
        label='Microsoft Foundry - Claude Opus 4.1',
        versions=['claude-opus-4-1'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'claude-opus-4-6': ModelInfo(
        label='Microsoft Foundry - Claude Opus 4.6',
        versions=['claude-opus-4-6'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # DeepSeek Models
    # =========================================================================
    'DeepSeek-V3.2': ModelInfo(
        label='Microsoft Foundry - DeepSeek V3.2',
        versions=['DeepSeek-V3.2'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'DeepSeek-V3.2-Speciale': ModelInfo(
        label='Microsoft Foundry - DeepSeek V3.2 Speciale',
        versions=['DeepSeek-V3.2-Speciale'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'DeepSeek-V3.1': ModelInfo(
        label='Microsoft Foundry - DeepSeek V3.1',
        versions=['DeepSeek-V3.1'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'DeepSeek-V3-0324': ModelInfo(
        label='Microsoft Foundry - DeepSeek V3 0324',
        versions=['DeepSeek-V3-0324'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'DeepSeek-R1-0528': ModelInfo(
        label='Microsoft Foundry - DeepSeek R1 0528',
        versions=['DeepSeek-R1-0528'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'MAI-DS-R1': ModelInfo(
        label='Microsoft Foundry - MAI-DS-R1',
        versions=['MAI-DS-R1'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # xAI Grok Models
    # =========================================================================
    'grok-4': ModelInfo(
        label='Microsoft Foundry - Grok 4',
        versions=['grok-4'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    'grok-4-fast-reasoning': ModelInfo(
        label='Microsoft Foundry - Grok 4 Fast Reasoning',
        versions=['grok-4-fast-reasoning'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    'grok-4-fast-non-reasoning': ModelInfo(
        label='Microsoft Foundry - Grok 4 Fast Non-Reasoning',
        versions=['grok-4-fast-non-reasoning'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    'grok-3': ModelInfo(
        label='Microsoft Foundry - Grok 3',
        versions=['grok-3'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    'grok-3-mini': ModelInfo(
        label='Microsoft Foundry - Grok 3 Mini',
        versions=['grok-3-mini'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    'grok-code-fast-1': ModelInfo(
        label='Microsoft Foundry - Grok Code Fast 1',
        versions=['grok-code-fast-1'],
        supports=GROK_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Meta Llama Models
    # =========================================================================
    'Llama-4-Maverick-17B-128E-Instruct-FP8': ModelInfo(
        label='Microsoft Foundry - Llama 4 Maverick 17B',
        versions=['Llama-4-Maverick-17B-128E-Instruct-FP8'],
        supports=LLAMA_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Mistral Models
    # =========================================================================
    'Mistral-Large-3': ModelInfo(
        label='Microsoft Foundry - Mistral Large 3',
        versions=['Mistral-Large-3'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral-document-ai-2505': ModelInfo(
        label='Microsoft Foundry - Mistral Document AI',
        versions=['mistral-document-ai-2505'],
        supports=Supports(
            multiturn=False,
            media=True,
            tools=False,
            system_role=False,
            output=['text'],
        ),
    ),
    # =========================================================================
    # Other Models
    # =========================================================================
    'Kimi-K2-Thinking': ModelInfo(
        label='Microsoft Foundry - Kimi K2 Thinking',
        versions=['Kimi-K2-Thinking'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'model-router': ModelInfo(
        label='Microsoft Foundry - Model Router',
        versions=['model-router'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
}

# Models that support the OpenAI response_format parameter
# Note: Duplicates removed, typos fixed (gpt-o1 -> o1, gpt-oss-120b -> gpt-oss-120B)
MODELS_SUPPORTING_RESPONSE_FORMAT = [
    # GPT-3.5 series
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-instruct',
    'gpt-3.5-turbo-16k',
    # GPT-4 series
    'gpt-4',
    'gpt-4-32k',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4.1',
    'gpt-4.1-mini',
    'gpt-4.1-nano',
    'gpt-4.5',
    # GPT-5 series
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-chat',
    'gpt-5-codex',
    'gpt-5-pro',
    'gpt-5.1',
    'gpt-5.1-chat',
    'gpt-5.1-codex',
    'gpt-5.1-codex-mini',
    'gpt-5.1-codex-max',
    'gpt-5.2',
    'gpt-5.2-chat',
    'gpt-5.2-codex',
    # OSS models
    'gpt-oss-20b',
    'gpt-oss-120B',
    # o-series reasoning models
    'o1',
    'o1-mini',
    'o1-preview',
    'o3',
    'o3-mini',
    'o3-pro',
    'o4-mini',
    # Codex
    'codex-mini',
]

# Embedding models
# See: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/models#embeddings
SUPPORTED_EMBEDDING_MODELS: dict[str, dict] = {
    'text-embedding-3-small': {
        'label': 'Microsoft Foundry - Text Embedding 3 Small',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
    'text-embedding-3-large': {
        'label': 'Microsoft Foundry - Text Embedding 3 Large',
        'dimensions': 3072,
        'supports': {'input': ['text']},
    },
    'text-embedding-ada-002': {
        'label': 'Microsoft Foundry - Text Embedding ADA 002',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
    # Cohere embedding model
    # See: https://ai.azure.com/explore/models/embed-v-4-0/version/4/registry/azureml-cohere
    'embed-v-4-0': {
        'label': 'Microsoft Foundry - Cohere Embed v4.0',
        'dimensions': 1536,  # Supports 256, 512, 1024, 1536
        'supports': {'input': ['text', 'image']},
    },
}


def get_model_info(name: str) -> ModelInfo:
    """Get model info for a given model name.

    For the full model catalog, see:
    https://ai.azure.com/catalog/models

    Args:
        name: The name of the model.

    Returns:
        ModelInfo for the model.
    """
    if name in SUPPORTED_MICROSOFT_FOUNDRY_MODELS:
        return SUPPORTED_MICROSOFT_FOUNDRY_MODELS[name]
    # Default info for unknown models - assume multimodal capable
    # This allows users to use any model from the 11,000+ catalog dynamically
    return ModelInfo(
        label=f'Microsoft Foundry - {name}',
        supports=MULTIMODAL_MODEL_SUPPORTS,
    )
