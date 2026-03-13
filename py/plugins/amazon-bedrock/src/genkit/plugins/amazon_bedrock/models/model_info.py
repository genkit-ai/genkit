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

"""AWS Bedrock model info.

This module defines the supported models for the AWS Bedrock plugin.
AWS Bedrock provides access to foundation models from multiple providers
through a unified API.

See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

Supported Model Categories
==========================
+------------------+--------------------------------------------------+
| Provider         | Models                                           |
+------------------+--------------------------------------------------+
| Amazon           | Nova Pro, Nova Lite, Nova Micro, Titan           |
| Anthropic        | Claude Opus, Sonnet, Haiku (4.x, 3.x series)     |
| AI21 Labs        | Jamba 1.5 Large, Mini                            |
| Cohere           | Command R, Command R+, Embed, Rerank             |
| DeepSeek         | R1, V3.1                                         |
| Google           | Gemma 3 (4B, 12B, 27B)                           |
| Meta             | Llama 3.x, Llama 4 Maverick/Scout                |
| MiniMax          | M2                                               |
| Mistral AI       | Large 3, Pixtral, Magistral, Ministral           |
| Moonshot AI      | Kimi K2 Thinking                                 |
| NVIDIA           | Nemotron Nano                                    |
| OpenAI           | GPT-OSS 120B, 20B                                |
| Qwen             | Qwen3 32B, 235B, Coder, VL                       |
| Writer           | Palmyra X4, X5                                   |
+------------------+--------------------------------------------------+

Model ID Format:
    Bedrock model IDs follow the pattern: `{provider}.{model-name}-{version}`
    Examples:
        - anthropic.claude-sonnet-4-5-20250929-v1:0
        - meta.llama3-3-70b-instruct-v1:0
        - amazon.nova-pro-v1:0

Cross-Region Inference:
    For cross-region inference profiles, use the region-prefixed ID:
        - us.anthropic.claude-sonnet-4-5-20250929-v1:0
        - us.deepseek.r1-v1:0

Trademark Notice:
    Model names are trademarks of their respective owners. This plugin is
    developed independently and is not affiliated with or endorsed by the
    model providers.
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

TEXT_ONLY_NO_TOOLS_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=False,
    system_role=True,
    output=['text'],
)

# Reasoning models typically output text only
REASONING_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=False,
    system_role=True,
    output=['text'],
)

# Anthropic Claude model supports
CLAUDE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Amazon Nova model supports (text, image, video input)
NOVA_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

NOVA_MICRO_SUPPORTS = Supports(
    multiturn=True,
    media=False,  # Micro is text-only
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Meta Llama model supports
LLAMA_TEXT_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

LLAMA_MULTIMODAL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Mistral model supports
MISTRAL_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

MISTRAL_MULTIMODAL_SUPPORTS = Supports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# Cohere model supports
COHERE_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=True,
    system_role=True,
    output=['text', 'json'],
)

# DeepSeek model supports (reasoning)
DEEPSEEK_MODEL_SUPPORTS = Supports(
    multiturn=True,
    media=False,
    tools=False,  # DeepSeek-R1 doesn't support tools in Bedrock
    system_role=True,
    output=['text'],
)

SUPPORTED_BEDROCK_MODELS: dict[str, ModelInfo] = {
    # =========================================================================
    # Amazon Nova Models
    # See: https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html
    # =========================================================================
    'amazon.nova-pro-v1:0': ModelInfo(
        label='Amazon Nova Pro',
        versions=['amazon.nova-pro-v1:0'],
        supports=NOVA_MODEL_SUPPORTS,
    ),
    'amazon.nova-lite-v1:0': ModelInfo(
        label='Amazon Nova Lite',
        versions=['amazon.nova-lite-v1:0'],
        supports=NOVA_MODEL_SUPPORTS,
    ),
    'amazon.nova-micro-v1:0': ModelInfo(
        label='Amazon Nova Micro',
        versions=['amazon.nova-micro-v1:0'],
        supports=NOVA_MICRO_SUPPORTS,
    ),
    'amazon.nova-premier-v1:0': ModelInfo(
        label='Amazon Nova Premier',
        versions=['amazon.nova-premier-v1:0'],
        supports=NOVA_MODEL_SUPPORTS,
    ),
    'amazon.nova-2-lite-v1:0': ModelInfo(
        label='Amazon Nova 2 Lite',
        versions=['amazon.nova-2-lite-v1:0'],
        supports=NOVA_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Anthropic Claude Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
    # =========================================================================
    'anthropic.claude-sonnet-4-5-20250929-v1:0': ModelInfo(
        label='Claude Sonnet 4.5',
        versions=['anthropic.claude-sonnet-4-5-20250929-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-sonnet-4-20250514-v1:0': ModelInfo(
        label='Claude Sonnet 4',
        versions=['anthropic.claude-sonnet-4-20250514-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-opus-4-5-20251101-v1:0': ModelInfo(
        label='Claude Opus 4.5',
        versions=['anthropic.claude-opus-4-5-20251101-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-opus-4-1-20250805-v1:0': ModelInfo(
        label='Claude Opus 4.1',
        versions=['anthropic.claude-opus-4-1-20250805-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-opus-4-6-20260205-v1:0': ModelInfo(
        label='Claude Opus 4.6',
        versions=['anthropic.claude-opus-4-6-20260205-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-haiku-4-5-20251001-v1:0': ModelInfo(
        label='Claude Haiku 4.5',
        versions=['anthropic.claude-haiku-4-5-20251001-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-3-5-haiku-20241022-v1:0': ModelInfo(
        label='Claude 3.5 Haiku',
        versions=['anthropic.claude-3-5-haiku-20241022-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    'anthropic.claude-3-haiku-20240307-v1:0': ModelInfo(
        label='Claude 3 Haiku',
        versions=['anthropic.claude-3-haiku-20240307-v1:0'],
        supports=CLAUDE_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # AI21 Labs Jamba Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
    # =========================================================================
    'ai21.jamba-1-5-large-v1:0': ModelInfo(
        label='Jamba 1.5 Large',
        versions=['ai21.jamba-1-5-large-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'ai21.jamba-1-5-mini-v1:0': ModelInfo(
        label='Jamba 1.5 Mini',
        versions=['ai21.jamba-1-5-mini-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Cohere Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
    # =========================================================================
    'cohere.command-r-plus-v1:0': ModelInfo(
        label='Command R+',
        versions=['cohere.command-r-plus-v1:0'],
        supports=COHERE_MODEL_SUPPORTS,
    ),
    'cohere.command-r-v1:0': ModelInfo(
        label='Command R',
        versions=['cohere.command-r-v1:0'],
        supports=COHERE_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # DeepSeek Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
    # =========================================================================
    'deepseek.r1-v1:0': ModelInfo(
        label='DeepSeek R1',
        versions=['deepseek.r1-v1:0'],
        supports=DEEPSEEK_MODEL_SUPPORTS,
    ),
    'deepseek.v3-v1:0': ModelInfo(
        label='DeepSeek V3.1',
        versions=['deepseek.v3-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Google Gemma Models
    # =========================================================================
    'google.gemma-3-4b-it': ModelInfo(
        label='Gemma 3 4B IT',
        versions=['google.gemma-3-4b-it'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'google.gemma-3-12b-it': ModelInfo(
        label='Gemma 3 12B IT',
        versions=['google.gemma-3-12b-it'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    'google.gemma-3-27b-it': ModelInfo(
        label='Gemma 3 27B IT',
        versions=['google.gemma-3-27b-it'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Meta Llama Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
    # =========================================================================
    'meta.llama3-3-70b-instruct-v1:0': ModelInfo(
        label='Llama 3.3 70B Instruct',
        versions=['meta.llama3-3-70b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-1-405b-instruct-v1:0': ModelInfo(
        label='Llama 3.1 405B Instruct',
        versions=['meta.llama3-1-405b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-1-70b-instruct-v1:0': ModelInfo(
        label='Llama 3.1 70B Instruct',
        versions=['meta.llama3-1-70b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-1-8b-instruct-v1:0': ModelInfo(
        label='Llama 3.1 8B Instruct',
        versions=['meta.llama3-1-8b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-70b-instruct-v1:0': ModelInfo(
        label='Llama 3 70B Instruct',
        versions=['meta.llama3-70b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-8b-instruct-v1:0': ModelInfo(
        label='Llama 3 8B Instruct',
        versions=['meta.llama3-8b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-2-90b-instruct-v1:0': ModelInfo(
        label='Llama 3.2 90B Instruct',
        versions=['meta.llama3-2-90b-instruct-v1:0'],
        supports=LLAMA_MULTIMODAL_SUPPORTS,
    ),
    'meta.llama3-2-11b-instruct-v1:0': ModelInfo(
        label='Llama 3.2 11B Instruct',
        versions=['meta.llama3-2-11b-instruct-v1:0'],
        supports=LLAMA_MULTIMODAL_SUPPORTS,
    ),
    'meta.llama3-2-3b-instruct-v1:0': ModelInfo(
        label='Llama 3.2 3B Instruct',
        versions=['meta.llama3-2-3b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama3-2-1b-instruct-v1:0': ModelInfo(
        label='Llama 3.2 1B Instruct',
        versions=['meta.llama3-2-1b-instruct-v1:0'],
        supports=LLAMA_TEXT_SUPPORTS,
    ),
    'meta.llama4-maverick-17b-instruct-v1:0': ModelInfo(
        label='Llama 4 Maverick 17B Instruct',
        versions=['meta.llama4-maverick-17b-instruct-v1:0'],
        supports=LLAMA_MULTIMODAL_SUPPORTS,
    ),
    'meta.llama4-scout-17b-instruct-v1:0': ModelInfo(
        label='Llama 4 Scout 17B Instruct',
        versions=['meta.llama4-scout-17b-instruct-v1:0'],
        supports=LLAMA_MULTIMODAL_SUPPORTS,
    ),
    # =========================================================================
    # MiniMax Models
    # =========================================================================
    'minimax.minimax-m2': ModelInfo(
        label='MiniMax M2',
        versions=['minimax.minimax-m2'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Mistral AI Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
    # =========================================================================
    'mistral.mistral-large-3-675b-instruct': ModelInfo(
        label='Mistral Large 3',
        versions=['mistral.mistral-large-3-675b-instruct'],
        supports=MISTRAL_MULTIMODAL_SUPPORTS,
    ),
    'mistral.mistral-large-2407-v1:0': ModelInfo(
        label='Mistral Large (24.07)',
        versions=['mistral.mistral-large-2407-v1:0'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.mistral-large-2402-v1:0': ModelInfo(
        label='Mistral Large (24.02)',
        versions=['mistral.mistral-large-2402-v1:0'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.mistral-small-2402-v1:0': ModelInfo(
        label='Mistral Small (24.02)',
        versions=['mistral.mistral-small-2402-v1:0'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.mistral-7b-instruct-v0:2': ModelInfo(
        label='Mistral 7B Instruct',
        versions=['mistral.mistral-7b-instruct-v0:2'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.mixtral-8x7b-instruct-v0:1': ModelInfo(
        label='Mixtral 8x7B Instruct',
        versions=['mistral.mixtral-8x7b-instruct-v0:1'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.pixtral-large-2502-v1:0': ModelInfo(
        label='Pixtral Large (25.02)',
        versions=['mistral.pixtral-large-2502-v1:0'],
        supports=MISTRAL_MULTIMODAL_SUPPORTS,
    ),
    'mistral.magistral-small-2509': ModelInfo(
        label='Magistral Small',
        versions=['mistral.magistral-small-2509'],
        supports=MISTRAL_MULTIMODAL_SUPPORTS,
    ),
    'mistral.ministral-3-3b-instruct': ModelInfo(
        label='Ministral 3B',
        versions=['mistral.ministral-3-3b-instruct'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.ministral-3-8b-instruct': ModelInfo(
        label='Ministral 8B',
        versions=['mistral.ministral-3-8b-instruct'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    'mistral.ministral-3-14b-instruct': ModelInfo(
        label='Ministral 14B',
        versions=['mistral.ministral-3-14b-instruct'],
        supports=MISTRAL_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Moonshot AI Models
    # =========================================================================
    'moonshot.kimi-k2-thinking': ModelInfo(
        label='Kimi K2 Thinking',
        versions=['moonshot.kimi-k2-thinking'],
        supports=REASONING_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # NVIDIA Models
    # =========================================================================
    'nvidia.nemotron-nano-9b-v2': ModelInfo(
        label='Nemotron Nano 9B',
        versions=['nvidia.nemotron-nano-9b-v2'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'nvidia.nemotron-nano-12b-v2': ModelInfo(
        label='Nemotron Nano 12B VL',
        versions=['nvidia.nemotron-nano-12b-v2'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # OpenAI Models (GPT-OSS on Bedrock)
    # =========================================================================
    'openai.gpt-oss-120b-1:0': ModelInfo(
        label='GPT-OSS 120B',
        versions=['openai.gpt-oss-120b-1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'openai.gpt-oss-20b-1:0': ModelInfo(
        label='GPT-OSS 20B',
        versions=['openai.gpt-oss-20b-1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'openai.gpt-oss-safeguard-120b': ModelInfo(
        label='GPT-OSS Safeguard 120B',
        versions=['openai.gpt-oss-safeguard-120b'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'openai.gpt-oss-safeguard-20b': ModelInfo(
        label='GPT-OSS Safeguard 20B',
        versions=['openai.gpt-oss-safeguard-20b'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Qwen Models
    # =========================================================================
    'qwen.qwen3-32b-v1:0': ModelInfo(
        label='Qwen3 32B',
        versions=['qwen.qwen3-32b-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'qwen.qwen3-235b-a22b-2507-v1:0': ModelInfo(
        label='Qwen3 235B A22B',
        versions=['qwen.qwen3-235b-a22b-2507-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'qwen.qwen3-coder-30b-a3b-v1:0': ModelInfo(
        label='Qwen3 Coder 30B',
        versions=['qwen.qwen3-coder-30b-a3b-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'qwen.qwen3-coder-480b-a35b-v1:0': ModelInfo(
        label='Qwen3 Coder 480B',
        versions=['qwen.qwen3-coder-480b-a35b-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'qwen.qwen3-next-80b-a3b': ModelInfo(
        label='Qwen3 Next 80B',
        versions=['qwen.qwen3-next-80b-a3b'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'qwen.qwen3-vl-235b-a22b': ModelInfo(
        label='Qwen3 VL 235B',
        versions=['qwen.qwen3-vl-235b-a22b'],
        supports=MULTIMODAL_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Writer Models
    # See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-writer-palmyra.html
    # =========================================================================
    'writer.palmyra-x4-v1:0': ModelInfo(
        label='Palmyra X4',
        versions=['writer.palmyra-x4-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    'writer.palmyra-x5-v1:0': ModelInfo(
        label='Palmyra X5',
        versions=['writer.palmyra-x5-v1:0'],
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    ),
    # =========================================================================
    # Amazon Titan Text Models
    # =========================================================================
    'amazon.titan-tg1-large': ModelInfo(
        label='Titan Text Large',
        versions=['amazon.titan-tg1-large'],
        supports=TEXT_ONLY_NO_TOOLS_SUPPORTS,
    ),
}

# Embedding models
# See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
SUPPORTED_EMBEDDING_MODELS: dict[str, dict] = {
    'amazon.titan-embed-text-v2:0': {
        'label': 'Titan Text Embeddings V2',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
    'amazon.titan-embed-text-v1': {
        'label': 'Titan Embeddings G1 - Text',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
    'amazon.titan-embed-image-v1': {
        'label': 'Titan Multimodal Embeddings G1',
        'dimensions': 1024,
        'supports': {'input': ['text', 'image']},
    },
    'amazon.nova-2-multimodal-embeddings-v1:0': {
        'label': 'Nova Multimodal Embeddings',
        'dimensions': 1024,
        'supports': {'input': ['text', 'image', 'audio', 'video']},
    },
    'cohere.embed-english-v3': {
        'label': 'Cohere Embed English',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
    'cohere.embed-multilingual-v3': {
        'label': 'Cohere Embed Multilingual',
        'dimensions': 1024,
        'supports': {'input': ['text']},
    },
    'cohere.embed-v4:0': {
        'label': 'Cohere Embed v4',
        'dimensions': 1024,
        'supports': {'input': ['text', 'image']},
    },
}


def get_model_info(name: str) -> ModelInfo:
    """Get model info for a given model name.

    For the full model catalog, see:
    https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

    Args:
        name: The name of the model (e.g., 'anthropic.claude-sonnet-4-5-20250929-v1:0').

    Returns:
        ModelInfo for the model.
    """
    if name in SUPPORTED_BEDROCK_MODELS:
        return SUPPORTED_BEDROCK_MODELS[name]

    # Default info for unknown models - assume text-only capable
    # This allows users to use any model dynamically
    return ModelInfo(
        label=f'Bedrock - {name}',
        supports=TEXT_ONLY_MODEL_SUPPORTS,
    )
