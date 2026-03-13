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

"""AWS Bedrock plugin for Genkit.

This plugin provides access to AWS Bedrock models through the Genkit framework.
AWS Bedrock is a fully managed service that provides access to foundation models
from multiple providers through a unified API.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ AWS Bedrock         │ Amazon's AI model marketplace. One place to       │
    │                     │ access Claude, Llama, Titan, and more.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Converse API        │ A unified way to talk to ANY Bedrock model.       │
    │                     │ Same code works for Claude, Llama, Nova, etc.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Model ID            │ The name of a specific model. Like                │
    │                     │ "anthropic.claude-3-sonnet-20240229-v1:0".        │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inference Profile   │ A cross-region alias for a model. Required        │
    │                     │ when using API keys instead of IAM roles.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Region              │ Which AWS data center to use. Pick one near       │
    │                     │ you (us-east-1, eu-west-1, ap-northeast-1).       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ IAM Role            │ AWS's way of granting permissions. Like a         │
    │                     │ badge that lets your code access models.          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Nova                │ Amazon's own AI models (Pro, Lite, Micro).        │
    │                     │ Good balance of cost and performance.             │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Inline Bytes        │ Bedrock needs actual image data, not URLs.        │
    │                     │ We fetch images for you automatically.            │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW AWS BEDROCK PROCESSES YOUR REQUEST                 │
    │                                                                         │
    │    Your Code                                                            │
    │    ai.generate(prompt="Describe this image", media=[image_url])         │
    │         │                                                               │
    │         │  (1) Request goes to AmazonBedrock plugin                        │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  AmazonBedrock     │   • Adds AWS credentials                         │
    │    │  Plugin         │   • Converts model ID → inference profile        │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Fetch image bytes (Bedrock needs inline data)        │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  BedrockModel   │   • Downloads image from URL                     │
    │    │  (httpx async)  │   • Converts to Converse API format              │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) Bedrock Converse API call                            │
    │             ▼                                                           │
    │    ════════════════════════════════════════════════════                 │
    │             │  Internet (HTTPS to bedrock.{region}.amazonaws.com)       │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  AWS Bedrock    │   Routes to the right provider                   │
    │    │                 │   (Claude, Llama, Nova, etc.)                    │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) Streaming response                                   │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your App       │   response.text = "I see a cute kitten..."       │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Architecture Overview::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         AWS Bedrock Plugin                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  Plugin Entry Point (__init__.py)                                       │
    │  ├── AmazonBedrock - Plugin class                                          │
    │  ├── bedrock_model() / inference_profile() - Helper functions           │
    │  └── Pre-defined models (claude_sonnet_4_5, deepseek_r1, nova_pro, ...) │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  plugin.py - Plugin Implementation                                      │
    │  ├── AmazonBedrock class (registers models/embedders)                      │
    │  └── Configuration and boto3 client initialization                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  typing.py - Type-Safe Configuration Classes                            │
    │  ├── BedrockConfig (common base)                                        │
    │  ├── Provider-specific configs (AnthropicConfig, MetaLlamaConfig, ...)  │
    │  └── Enums (CohereSafetyMode, CohereToolChoice, StabilityMode, ...)     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models/model.py - Model Implementation                                 │
    │  ├── BedrockModel (Converse API integration)                            │
    │  ├── Automatic inference profile conversion for API key auth            │
    │  ├── Async media URL fetching (httpx) - Bedrock requires inline bytes   │
    │  └── JSON mode via prompt engineering (no native support)               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  models/model_info.py - Model Registry                                  │
    │  ├── SUPPORTED_BEDROCK_MODELS (12+ providers)                           │
    │  └── SUPPORTED_EMBEDDING_MODELS                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Supported Model Providers:
    - Amazon (Nova Pro, Nova Lite, Nova Micro, Titan)
    - Anthropic (Claude Opus, Sonnet, Haiku)
    - AI21 Labs (Jamba 1.5)
    - Cohere (Command R, Command R+, Embed)
    - DeepSeek (R1, V3)
    - Google (Gemma 3)
    - Meta (Llama 3.x, Llama 4)
    - MiniMax (M2)
    - Mistral AI (Large 3, Pixtral, Ministral)
    - Moonshot AI (Kimi K2)
    - NVIDIA (Nemotron)
    - OpenAI (GPT-OSS)
    - Qwen (Qwen3)
    - Writer (Palmyra)

Documentation Links:
    - AWS Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html
    - Supported Models: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    - Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html

Example::

    from genkit import Genkit
    from genkit.plugins.amazon_bedrock import AmazonBedrock, claude_sonnet_4_5

    ai = Genkit(
        plugins=[AmazonBedrock(region='us-east-1')],
        model=claude_sonnet_4_5,
    )

    response = await ai.generate(prompt='Tell me a joke.')
    print(response.text)

Trademark Notice:
    This is a community plugin and is not officially supported by Amazon Web Services.
    "Amazon", "AWS", "Amazon Bedrock", and related marks are trademarks of
    Amazon.com, Inc. or its affiliates. Model names are trademarks of their
    respective owners.
"""

from .plugin import (
    AMAZON_BEDROCK_PLUGIN_NAME,
    AmazonBedrock,
    bedrock_model,
    bedrock_name,
    # Pre-defined model references - Anthropic Claude
    claude_3_5_haiku,
    claude_3_haiku,
    claude_haiku_4_5,
    claude_opus_4_1,
    claude_opus_4_5,
    claude_opus_4_6,
    claude_sonnet_4,
    claude_sonnet_4_5,
    # Pre-defined model references - Cohere
    command_r,
    command_r_plus,
    # Pre-defined model references - DeepSeek
    deepseek_r1,
    deepseek_v3,
    get_config_schema_for_model,
    # Inference profile helpers (for API key authentication)
    get_inference_profile_prefix,
    inference_profile,
    # Pre-defined model references - AI21 Jamba
    jamba_large,
    jamba_mini,
    # Pre-defined model references - Meta Llama
    llama_3_1_70b,
    llama_3_1_405b,
    llama_3_3_70b,
    llama_4_maverick,
    llama_4_scout,
    # Pre-defined model references - Mistral
    mistral_large,
    mistral_large_3,
    # Pre-defined model references - Amazon Nova
    nova_lite,
    nova_micro,
    nova_premier,
    nova_pro,
    pixtral_large,
)
from .telemetry import add_aws_telemetry
from .typing import (
    # Model-Specific Configs
    AI21JambaConfig,
    AmazonNovaConfig,
    AnthropicConfig,
    # Base/Common Configs
    BedrockConfig,
    CohereConfig,
    # Enums
    CohereSafetyMode,
    CohereToolChoice,
    DeepSeekConfig,
    # Mixin
    GenkitCommonConfigMixin,
    GoogleGemmaConfig,
    MetaLlamaConfig,
    MiniMaxConfig,
    MistralConfig,
    MoonshotConfig,
    NvidiaConfig,
    OpenAIConfig,
    QwenConfig,
    StabilityConfig,
    # Embedding Config
    TextEmbeddingConfig,
    TitanConfig,
    WriterConfig,
)

__all__ = [
    'AMAZON_BEDROCK_PLUGIN_NAME',
    'AI21JambaConfig',
    # Plugin
    'AmazonBedrock',
    # Model-Specific Configs (16 providers)
    'AmazonNovaConfig',
    'AnthropicConfig',
    # Base/Common Configs
    'BedrockConfig',
    'CohereConfig',
    # Enums
    'CohereSafetyMode',
    'CohereToolChoice',
    'DeepSeekConfig',
    # Mixin
    'GenkitCommonConfigMixin',
    'GoogleGemmaConfig',
    'MetaLlamaConfig',
    'MiniMaxConfig',
    'MistralConfig',
    'MoonshotConfig',
    'NvidiaConfig',
    'OpenAIConfig',
    'QwenConfig',
    'StabilityConfig',
    # Embedding Config
    'TextEmbeddingConfig',
    'TitanConfig',
    'WriterConfig',
    # Helper functions
    'bedrock_model',
    'bedrock_name',
    'claude_3_5_haiku',
    'claude_3_haiku',
    'claude_haiku_4_5',
    'claude_opus_4_1',
    'claude_opus_4_5',
    'claude_opus_4_6',
    'claude_sonnet_4',
    # Pre-defined model references - Anthropic Claude
    'claude_sonnet_4_5',
    'command_r',
    # Pre-defined model references - Cohere
    'command_r_plus',
    # Pre-defined model references - DeepSeek
    'deepseek_r1',
    'deepseek_v3',
    'get_config_schema_for_model',
    # Inference profile helpers (for API key authentication)
    'get_inference_profile_prefix',
    'inference_profile',
    # Pre-defined model references - AI21 Jamba
    'jamba_large',
    'jamba_mini',
    'llama_3_1_70b',
    'llama_3_1_405b',
    # Pre-defined model references - Meta Llama
    'llama_3_3_70b',
    'llama_4_maverick',
    'llama_4_scout',
    'mistral_large',
    # Pre-defined model references - Mistral
    'mistral_large_3',
    'nova_lite',
    'nova_micro',
    'nova_premier',
    # Pre-defined model references - Amazon Nova
    'nova_pro',
    'pixtral_large',
    # Telemetry
    'add_aws_telemetry',
]
