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

"""AWS Bedrock configuration types for Genkit.

This module defines configuration schemas that align with the AWS Bedrock
Converse API and model-specific parameters for each supported provider.

Architecture Overview::

    +------------------+       +----------------------+       +------------------+
    |   Genkit App     |       |  AWS Bedrock Plugin  |       |  AWS Bedrock     |
    |                  |       |                      |       |  Service         |
    |  +------------+  |       |  +----------------+  |       |                  |
    |  | ai.generate|------->|  | BedrockModel   |------->|  Converse API    |
    |  +------------+  |       |  +----------------+  |       |  ConverseStream  |
    |                  |       |                      |       |                  |
    |  +------------+  |       |  +----------------+  |       |  +------------+  |
    |  | ai.embed   |------->|  | EmbedAction    |------->|  | Embeddings |  |
    |  +------------+  |       |  +----------------+  |       |  +------------+  |
    +------------------+       +----------------------+       +------------------+
                                        |
                                        v
                               +------------------+
                               | Model Configs    |
                               | (this module)    |
                               |                  |
                               | - BedrockConfig  |
                               | - AnthropicConfig|
                               | - MetaLlamaConfig|
                               | - CohereConfig   |
                               | - MistralConfig  |
                               | - etc.           |
                               +------------------+

Design Rationale:
    We use static configuration classes with `extra='allow'` rather than dynamic
    parameter discovery for several reasons:

    1. **Type Safety**: Static configs provide IDE autocompletion, type checking,
       and validation for known parameters with documented constraints.

    2. **Forward Compatibility**: The `extra='allow'` Pydantic setting allows any
       additional parameters to pass through, supporting new/undocumented params.

    3. **DevUI Compatibility**: Model-specific configs enable the Genkit DevUI to
       show relevant configuration options for each model family.

Common Parameters NOT in Converse API:
    When using models through the Converse API, some native model parameters
    may need to be passed via `additionalModelRequestFields`:

    +-------------------+------------------+--------------------------------------+
    | Parameter         | Native API       | Converse Alternative                 |
    +-------------------+------------------+--------------------------------------+
    | top_k             | Anthropic,       | Pass via additionalModelRequestFields|
    |                   | Cohere           | or use top_p instead.                |
    +-------------------+------------------+--------------------------------------+
    | random_seed       | Mistral          | Pass via additionalModelRequestFields|
    +-------------------+------------------+--------------------------------------+
    | safe_prompt       | Mistral          | Pass via additionalModelRequestFields|
    +-------------------+------------------+--------------------------------------+
    | documents         | Cohere           | Pass via additionalModelRequestFields|
    +-------------------+------------------+--------------------------------------+
    | thinking          | Anthropic        | Pass via additionalModelRequestFields|
    +-------------------+------------------+--------------------------------------+

Supported Model Providers:
    This plugin supports foundation models available through the Bedrock service:

    +------------------+--------------------------------------------------+
    | Provider         | Models                                           |
    +------------------+--------------------------------------------------+
    | Amazon           | Nova Pro, Nova Lite, Nova Micro, Nova Premier,   |
    |                  | Nova Canvas, Nova Reel, Nova Sonic, Titan        |
    | Anthropic        | Claude Sonnet 4.5, Claude Opus 4.5, Claude 4,    |
    |                  | Claude Haiku 4.5, Claude 3.5 Haiku, Claude 3     |
    | AI21 Labs        | Jamba 1.5 Large, Jamba 1.5 Mini                  |
    | Cohere           | Command R, Command R+, Embed v4, Rerank 3.5      |
    | DeepSeek         | DeepSeek-R1, DeepSeek-V3.1                       |
    | Google           | Gemma 3 4B IT, Gemma 3 12B IT, Gemma 3 27B IT    |
    | Luma AI          | Ray v2 (video generation)                        |
    | Meta             | Llama 4 Maverick, Llama 4 Scout, Llama 3.3,      |
    |                  | Llama 3.2, Llama 3.1, Llama 3                    |
    | MiniMax          | MiniMax M2                                       |
    | Mistral AI       | Mistral Large 3, Pixtral Large, Magistral Small, |
    |                  | Ministral, Mixtral, Voxtral                      |
    | Moonshot AI      | Kimi K2 Thinking                                 |
    | NVIDIA           | Nemotron Nano 9B v2, Nemotron Nano 12B v2        |
    | OpenAI           | GPT OSS 120B, GPT OSS 20B, GPT OSS Safeguard     |
    | Qwen             | Qwen3 32B, Qwen3 235B, Qwen3 Coder, Qwen3 VL     |
    | Stability AI     | Stable Diffusion 3.5, Stable Image Core/Ultra    |
    | TwelveLabs       | Pegasus, Marengo Embed (video understanding)     |
    | Writer           | Palmyra X4, Palmyra X5                           |
    +------------------+--------------------------------------------------+

See Also:
    AWS Bedrock Documentation:
        - Model Parameters: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
        - Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        - InferenceConfiguration: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html

    Model-Specific Documentation:
        - Anthropic: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
        - Meta Llama: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        - Mistral: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
        - Cohere: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
        - DeepSeek: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
        - Amazon Nova: https://docs.aws.amazon.com/nova/latest/userguide/getting-started-schema.html

Trademark Notice:
    This is a community plugin and is not officially supported by Amazon Web Services.
    "Amazon", "AWS", "Amazon Bedrock", and related marks are trademarks of
    Amazon.com, Inc. or its affiliates. Model names (Claude, Llama, Mistral, etc.)
    are trademarks of their respective owners.
"""

import sys
from typing import Any, ClassVar, Literal

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CohereSafetyMode(StrEnum):
    """Safety mode for Cohere models.

    Controls the safety instruction inserted into the prompt.

    See: https://docs.cohere.com/v2/docs/safety-modes
    """

    CONTEXTUAL = 'CONTEXTUAL'
    STRICT = 'STRICT'
    OFF = 'OFF'


class CohereToolChoice(StrEnum):
    """Tool choice for Cohere models.

    Controls whether the model is forced to use a tool.

    See: https://docs.cohere.com/v2/reference/chat
    """

    REQUIRED = 'REQUIRED'
    NONE = 'NONE'


class GenkitCommonConfigMixin(BaseModel):
    """Genkit common configuration parameters mixin.

    These parameters match the Genkit core GenerationCommonConfigSchema and are
    expected by the Genkit DevUI for proper rendering of the config pane.

    Reference:
        - JS Schema: js/ai/src/model-types.ts (GenerationCommonConfigSchema)
        - Python Schema: genkit/core/typing.py (GenerationCommonConfig)

    When creating model configs, inherit from this mixin (via BedrockConfig)
    to ensure DevUI compatibility.

    Parameters:
        version: A specific version of the model family.
        temperature: Controls randomness in token selection (0.0-1.0).
        max_output_tokens: Maximum number of tokens to generate.
        top_k: Maximum number of tokens to consider when sampling.
        top_p: Nucleus sampling probability mass (0.0-1.0).
        stop_sequences: Strings that will stop output generation.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    version: str | None = Field(
        default=None,
        description='A specific version of the model family.',
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Controls randomness in token selection (0.0-1.0).',
    )
    max_output_tokens: int | None = Field(
        default=None,
        description='Maximum number of tokens to generate.',
    )
    top_k: int | None = Field(
        default=None,
        description='Maximum number of tokens to consider when sampling.',
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Nucleus sampling probability mass (0.0-1.0).',
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description='Strings that will stop output generation.',
    )


class BedrockConfig(GenkitCommonConfigMixin):
    """Base AWS Bedrock configuration for Genkit.

    Combines:
    - **GenkitCommonConfigMixin**: Standard Genkit parameters for DevUI compatibility
    - **Bedrock Converse API parameters**: For AWS Bedrock API compatibility

    Use model-specific configs (AnthropicConfig, MetaLlamaConfig, etc.) for additional
    model-specific parameters. All model configs inherit from this base.

    Official Documentation:
        - Converse API: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        - InferenceConfiguration: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
    """

    # Bedrock Converse API standard parameters
    # Note: temperature, top_p, stop_sequences are inherited from GenkitCommonConfigMixin
    max_tokens: int | None = Field(
        default=None,
        description='Maximum tokens (Bedrock-style). Use max_output_tokens for Genkit compatibility.',
    )


class AmazonNovaConfig(BedrockConfig):
    """Configuration for Amazon Nova models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports:
        - Nova Pro (text, image, video input)
        - Nova Lite (text, image, video input)
        - Nova Micro (text only)
        - Nova Premier (text, image, video input)
        - Nova Canvas (image generation)
        - Nova Reel (video generation)
        - Nova Sonic / Nova 2 Sonic (speech)

    Official Documentation:
        - Amazon Nova: https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html
        - Request Schema: https://docs.aws.amazon.com/nova/latest/userguide/getting-started-schema.html

    Note: The timeout period for inference calls to Amazon Nova is 60 minutes.
    """

    pass  # Nova uses standard Converse API parameters from BedrockConfig


class AnthropicToolChoice(StrEnum):
    """Tool choice mode for Anthropic models.

    Controls how the model uses tools.

    See: https://docs.anthropic.com/en/api/messages
    """

    AUTO = 'auto'
    ANY = 'any'
    TOOL = 'tool'


class AnthropicEffort(StrEnum):
    """Effort level for Claude Opus 4.5 (beta).

    Controls how liberally Claude spends tokens for the best result.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html
    """

    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class AnthropicConfig(BedrockConfig):
    """Configuration for Anthropic Claude models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Anthropic-specific params.

    Supports:
        - Claude Sonnet 4.5 (anthropic.claude-sonnet-4-5-20250929-v1:0)
        - Claude Opus 4.5 (anthropic.claude-opus-4-5-20251101-v1:0)
        - Claude Opus 4.1 (anthropic.claude-opus-4-1-20250805-v1:0)
        - Claude Sonnet 4 (anthropic.claude-sonnet-4-20250514-v1:0)
        - Claude Haiku 4.5 (anthropic.claude-haiku-4-5-20251001-v1:0)
        - Claude 3.5 Haiku (anthropic.claude-3-5-haiku-20241022-v1:0)
        - Claude 3 Haiku (anthropic.claude-3-haiku-20240307-v1:0)

    Official Documentation:
        - Anthropic on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html
        - Messages API: https://docs.anthropic.com/en/api/messages
        - Request/Response: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html

    Note:
        - Claude Sonnet 4.5 and Haiku 4.5 support specifying either temperature OR top_p, not both.
        - The timeout period for Claude 4 models is 60 minutes.
    """

    # Anthropic-specific parameters
    anthropic_version: str | None = Field(
        default=None,
        description='Anthropic API version. Use "bedrock-2023-05-31" for Messages API.',
    )
    anthropic_beta: list[str] | None = Field(
        default=None,
        description='Beta headers for opt-in features (e.g., "computer-use-2025-01-24", "effort-2025-11-24").',
    )
    system: str | None = Field(
        default=None,
        description='System prompt for context and instructions. Requires Claude 2.1+.',
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description='Configuration for enabling Claude extended thinking capability.',
    )
    effort: AnthropicEffort | None = Field(
        default=None,
        description='Effort level for Claude Opus 4.5 (beta). Requires "effort-2025-11-24" beta header.',
    )
    tool_choice: dict[str, str] | None = Field(
        default=None,
        description='How to use tools: {"type": "auto"|"any"|"tool", "name": "tool_name"}.',
    )


class AI21JambaConfig(BedrockConfig):
    """Configuration for AI21 Labs Jamba models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Jamba-specific params.

    Supports: Jamba 1.5 Large, Jamba 1.5 Mini (256K context window)

    Official Documentation:
        - AI21 on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
        - AI21 Jamba API: https://docs.ai21.com/reference/jamba-1-6-api-ref

    Features:
        - 256K token context window
        - Structured JSON output
        - Function calling support
        - Multilingual (EN, ES, FR, PT, IT, NL, DE, AR, HE)

    Default Values:
        - temperature: 1.0 (range: 0.0-2.0, note: higher max than most models)
        - top_p: 1.0 (range: 0.01-1.0)
    """

    # Note: Jamba supports temperature up to 2.0, unlike most models
    # The base BedrockConfig limits to 1.0, but extra='allow' lets higher values through

    n: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description='Number of chat responses to generate. Default: 1, Max: 16. Note: n > 1 requires temperature > 0.',
    )
    frequency_penalty: float | None = Field(
        default=None,
        description='Reduce frequency of repeated words. Higher values produce fewer repeated words.',
    )
    presence_penalty: float | None = Field(
        default=None,
        description='Reduce repetition of any repeated tokens, applied equally regardless of frequency.',
    )
    stop: list[str] | None = Field(
        default=None,
        description='Stop sequences (up to 64K each). Supports newlines as \\n.',
    )


class CohereConfig(BedrockConfig):
    """Configuration for Cohere models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Cohere-specific params.

    Supports:
        - Command R+ (cohere.command-r-plus-v1:0)
        - Command R (cohere.command-r-v1:0)
        - Embed English (cohere.embed-english-v3)
        - Embed Multilingual (cohere.embed-multilingual-v3)
        - Embed v4 (cohere.embed-v4:0) - text and image
        - Rerank 3.5 (cohere.rerank-v3-5:0)

    Official Documentation:
        - Cohere on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command-r-plus.html
        - Cohere Chat API: https://docs.cohere.com/v2/reference/chat
    """

    k: int | None = Field(
        default=None,
        ge=0,
        le=500,
        description='Top-k sampling (Cohere-specific). When k=0, k-sampling is disabled. Default: 0.',
    )
    p: float | None = Field(
        default=None,
        ge=0.01,
        le=0.99,
        description='Nucleus sampling probability mass (Cohere-specific). Default: 0.75.',
    )
    safety_mode: CohereSafetyMode | None = Field(
        default=None,
        description='Safety instruction mode: CONTEXTUAL, STRICT, or OFF.',
    )
    tool_choice: CohereToolChoice | None = Field(
        default=None,
        description='Force tool use: REQUIRED or NONE.',
    )
    documents: list[str | dict[str, Any]] | None = Field(
        default=None,
        description='Documents for RAG-based generation with citations.',
    )
    search_queries_only: bool | None = Field(
        default=None,
        description='Only return search queries without model response.',
    )
    preamble: str | None = Field(
        default=None,
        description='Override default preamble for search query generation.',
    )
    prompt_truncation: Literal['OFF', 'AUTO_PRESERVE_ORDER'] | None = Field(
        default=None,
        description='How to handle prompt truncation when exceeding context.',
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Reduce repetitiveness proportional to frequency.',
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Reduce repetitiveness for any repeated tokens.',
    )
    seed: int | None = Field(
        default=None,
        description='Seed for deterministic sampling.',
    )
    return_prompt: bool | None = Field(
        default=None,
        description='Return full prompt in response.',
    )
    raw_prompting: bool | None = Field(
        default=None,
        description='Send message without preprocessing.',
    )


class DeepSeekConfig(BedrockConfig):
    """Configuration for DeepSeek models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus DeepSeek-specific params.

    Supports: DeepSeek-R1, DeepSeek-V3.1

    Official Documentation:
        - DeepSeek on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-deepseek.html
        - DeepSeek API: https://api-docs.deepseek.com/api/create-chat-completion
        - Reasoning Guide: https://api-docs.deepseek.com/guides/reasoning_model.html

    Important Notes:
        - For optimal response quality with DeepSeek-R1, limit max_tokens to 8,192 or fewer.
        - While API accepts up to 32,768 tokens, quality degrades above 8,192.
        - Response includes reasoning_content for reasoning models (Converse API).
        - Must use cross-region inference profile ID (e.g., us.deepseek.r1-v1:0).

    Response Stop Reasons:
        - 'stop': Model finished generating
        - 'length': Hit max_tokens limit (increase max_tokens)
    """

    stop: list[str] | None = Field(
        default=None,
        max_length=10,
        description='Stop sequences (max 10 items).',
    )


class GoogleGemmaConfig(BedrockConfig):
    """Configuration for Google Gemma models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports:
        - Gemma 3 4B IT (google.gemma-3-4b-it) - text + image
        - Gemma 3 12B IT (google.gemma-3-12b-it) - text + image
        - Gemma 3 27B IT (google.gemma-3-27b-it) - text + image

    Official Documentation:
        - Gemma on Bedrock: Available in model catalog
        - Google AI: https://ai.google.dev/gemma

    Note: "IT" = Instruction Tuned variants.
    """

    pass  # Gemma uses standard parameters from BedrockConfig


class MetaLlamaConfig(BedrockConfig):
    """Configuration for Meta Llama models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Llama-specific params.

    Supports:
        - Llama 4 Maverick 17B Instruct (meta.llama4-maverick-17b-instruct-v1:0)
        - Llama 4 Scout 17B Instruct (meta.llama4-scout-17b-instruct-v1:0)
        - Llama 3.3 70B Instruct (meta.llama3-3-70b-instruct-v1:0)
        - Llama 3.2 90B/11B/3B/1B Instruct (multimodal for 90B/11B)
        - Llama 3.1 405B/70B/8B Instruct
        - Llama 3 70B/8B Instruct

    Official Documentation:
        - Meta on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        - Llama Guide: https://ai.meta.com/llama/get-started/#prompting

    Note:
        - Llama 3.2 Instruct and Llama 3.3 Instruct models use geofencing.
        - Llama 4 models support streaming.

    Default Values:
        - temperature: 0.5 (range: 0-1)
        - top_p: 0.9 (range: 0-1)
        - max_gen_len: 512 (range: 1-2048)
    """

    max_gen_len: int | None = Field(
        default=None,
        ge=1,
        le=2048,
        description='Maximum tokens to generate (Llama naming). Default: 512, Max: 2048.',
    )
    images: list[str] | None = Field(
        default=None,
        description='List of base64-encoded images for Llama 3.2+ multimodal models.',
    )


class MiniMaxConfig(BedrockConfig):
    """Configuration for MiniMax models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports: MiniMax M2

    Official Documentation:
        - MiniMax on Bedrock: Available in model catalog
    """

    pass  # MiniMax uses standard parameters from BedrockConfig


class MistralToolChoice(StrEnum):
    """Tool choice mode for Mistral models.

    Controls how the model uses tools.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-chat-completion.html
    """

    AUTO = 'auto'
    ANY = 'any'
    NONE = 'none'


class MistralConfig(BedrockConfig):
    """Configuration for Mistral AI models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Mistral-specific params.

    Supports:
        - Mistral Large 3 (mistral.mistral-large-3-675b-instruct)
        - Pixtral Large 25.02 (mistral.pixtral-large-2502-v1:0) - multimodal
        - Magistral Small 2509 (mistral.magistral-small-2509) - multimodal
        - Ministral 3B/8B/14B (mistral.ministral-3-*-instruct)
        - Mixtral 8x7B Instruct (mistral.mixtral-8x7b-instruct-v0:1)
        - Mistral 7B Instruct (mistral.mistral-7b-instruct-v0:2)
        - Voxtral Mini/Small (mistral.voxtral-*) - speech models

    Official Documentation:
        - Mistral on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
        - Chat Completion: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral-chat-completion.html
        - Mistral API: https://docs.mistral.ai/capabilities/completion/

    Note: Mistral models support both text completion and chat completion formats.

    Default Values (Mistral Large):
        - temperature: 0.7 (range: 0-1)
        - top_p: 1.0 (range: 0-1)
        - max_tokens: 8192
    """

    random_seed: int | None = Field(
        default=None,
        description='Seed for deterministic sampling. Mistral uses random_seed instead of seed.',
    )
    safe_prompt: bool | None = Field(
        default=None,
        description='Enable safety additions to reduce risky outputs.',
    )
    tool_choice: MistralToolChoice | None = Field(
        default=None,
        description='How to use tools: auto (model decides), any (forced), none (disabled).',
    )


class MoonshotConfig(BedrockConfig):
    """Configuration for Moonshot AI models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports: Kimi K2 Thinking

    Official Documentation:
        - Moonshot on Bedrock: Available in model catalog
    """

    pass  # Moonshot uses standard parameters from BedrockConfig


class NvidiaConfig(BedrockConfig):
    """Configuration for NVIDIA models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus NVIDIA-specific params.

    Supports:
        - NVIDIA Nemotron Nano 9B v2 (nvidia.nemotron-nano-9b-v2) - text
        - NVIDIA Nemotron Nano 12B v2 VL BF16 (nvidia.nemotron-nano-12b-v2) - text + image

    Official Documentation:
        - NVIDIA NIM API: https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
        - Thinking Budget: https://docs.nvidia.com/nim/large-language-models/latest/thinking-budget-control.html

    Note: NVIDIA NIM API is compatible with OpenAI's API format.
    """

    max_thinking_tokens: int | None = Field(
        default=None,
        description='Max reasoning tokens before final answer. Controls thinking budget for reflection models.',
    )


class OpenAIConfig(BedrockConfig):
    """Configuration for OpenAI models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports:
        - GPT OSS 120B (openai.gpt-oss-120b-1:0)
        - GPT OSS 20B (openai.gpt-oss-20b-1:0)
        - GPT OSS Safeguard 120B (openai.gpt-oss-safeguard-120b)
        - GPT OSS Safeguard 20B (openai.gpt-oss-safeguard-20b)

    Official Documentation:
        - OpenAI on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-openai.html

    Note: These are open-source models, not OpenAI's proprietary GPT-4/GPT-3.5.
    """

    pass  # OpenAI uses standard parameters from BedrockConfig


class QwenConfig(BedrockConfig):
    """Configuration for Alibaba Qwen models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Qwen-specific params.

    Supports:
        - Qwen3 32B dense (qwen.qwen3-32b-v1:0)
        - Qwen3 235B A22B 2507 (qwen.qwen3-235b-a22b-2507-v1:0)
        - Qwen3 Next 80B A3B (qwen.qwen3-next-80b-a3b)
        - Qwen3 Coder 480B A35B (qwen.qwen3-coder-480b-a35b-v1:0)
        - Qwen3 Coder 30B A3B (qwen.qwen3-coder-30b-a3b-v1:0)
        - Qwen3 VL 235B A22B (qwen.qwen3-vl-235b-a22b) - text + image

    Official Documentation:
        - Qwen on Bedrock: Available in model catalog
        - Qwen API: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference
        - Thinking Mode: https://www.alibabacloud.com/help/en/model-studio/use-qwen-by-calling-api

    Key Features:
        - OpenAI-compatible API format
        - Advanced reasoning with Thinking mode (Qwen3)
        - Multimodal support (text + images for VL models)

    Note: Qwen API is OpenAI-compatible, so standard parameters work.
    """

    enable_thinking: bool | None = Field(
        default=None,
        description='Enable Thinking mode for enhanced reasoning (Qwen3 models).',
    )


class WriterConfig(BedrockConfig):
    """Configuration for Writer AI models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports: Palmyra X4 (128K context), Palmyra X5 (1M context)

    Official Documentation:
        - Writer on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-writer-palmyra.html
        - Writer API: https://dev.writer.com/api-guides/chat-completion
        - Bedrock Integration: https://dev.writer.com/providers/aws-bedrock

    Key Features:
        - Advanced reasoning and multi-step tool-calling
        - Code generation and structured outputs
        - Built-in RAG (Retrieval-Augmented Generation)
        - Multilingual support (30+ languages)
        - Adaptive reasoning for context-based strategy adjustment

    Note: Writer API uses standard OpenAI-compatible parameters.
    Temperature defaults to 1 and controls response randomness.
    """


class StabilityAspectRatio(StrEnum):
    """Aspect ratio options for Stability AI image generation.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
    """

    RATIO_16_9 = '16:9'
    RATIO_1_1 = '1:1'
    RATIO_21_9 = '21:9'
    RATIO_2_3 = '2:3'
    RATIO_3_2 = '3:2'
    RATIO_4_5 = '4:5'
    RATIO_5_4 = '5:4'
    RATIO_9_16 = '9:16'
    RATIO_9_21 = '9:21'


class StabilityOutputFormat(StrEnum):
    """Output format options for Stability AI image generation.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
    """

    JPEG = 'jpeg'
    PNG = 'png'
    WEBP = 'webp'


class StabilityMode(StrEnum):
    """Generation mode for Stability AI models.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
    """

    TEXT_TO_IMAGE = 'text-to-image'
    IMAGE_TO_IMAGE = 'image-to-image'


class StabilityConfig(BedrockConfig):
    """Configuration for Stability AI models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig.

    Supports:
        - Stable Diffusion 3.5 Large (stability.sd3-5-large-v1:0)
        - Stable Image Core 1.0 (stability.stable-image-core-v1:1)
        - Stable Image Ultra 1.0 (stability.stable-image-ultra-v1:1)
        - Stable Image Control (Sketch, Structure)
        - Stable Image Editing (Inpaint, Outpaint, Erase, Search and Replace/Recolor)
        - Stable Image Upscale (Fast, Creative, Conservative)

    Official Documentation:
        - Stability on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html
        - SD 3.5 Large: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
        - Stable Image Services: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html

    Note: These are image generation/editing models, not text generation.

    Generation Modes:
        - text-to-image: Requires only prompt
        - image-to-image: Requires prompt, image, and strength
    """

    # Stable Diffusion 3.5 Large parameters
    seed: int | None = Field(
        default=None,
        ge=0,
        le=4294967294,
        description='Random seed for reproducible generation. 0 = random seed.',
    )
    aspect_ratio: StabilityAspectRatio | None = Field(
        default=None,
        description='Aspect ratio for text-to-image. Default: 1:1.',
    )
    mode: StabilityMode | None = Field(
        default=None,
        description='Generation mode: text-to-image or image-to-image.',
    )
    negative_prompt: str | None = Field(
        default=None,
        max_length=10000,
        description='Text describing elements to exclude from the output image.',
    )
    output_format: StabilityOutputFormat | None = Field(
        default=None,
        description='Output image format: jpeg, png, or webp. Default: png.',
    )
    image: str | None = Field(
        default=None,
        description='Base64-encoded input image for image-to-image mode. Min 64px per side.',
    )
    strength: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Influence of input image (image-to-image). 0 = preserve input, 1 = ignore input.',
    )

    # Legacy parameters for older Stability models (Stable Diffusion XL, etc.)
    cfg_scale: float | None = Field(
        default=None,
        description='(Legacy) How strongly the image should conform to prompt. Range: 0-35.',
    )
    steps: int | None = Field(
        default=None,
        description='(Legacy) Number of diffusion steps. Range: 10-150.',
    )
    style_preset: str | None = Field(
        default=None,
        description='(Legacy) Style preset for image generation.',
    )


class TitanConfig(BedrockConfig):
    r"""Configuration for Amazon Titan models on AWS Bedrock.

    Inherits all Genkit common parameters from BedrockConfig plus Titan-specific params.

    Supports:
        - Titan Text Large (amazon.titan-tg1-large)
        - Titan Embeddings G1 - Text (amazon.titan-embed-text-v1)
        - Titan Text Embeddings V2 (amazon.titan-embed-text-v2:0)
        - Titan Multimodal Embeddings G1 (amazon.titan-embed-image-v1)
        - Titan Image Generator G1 v2 (amazon.titan-image-generator-v2:0)

    Official Documentation:
        - Titan on Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html

    Default Values:
        - temperature: 0.7 (range: 0.0-1.0)
        - top_p: 0.9 (range: 0.0-1.0)
        - max_token_count: 512

    Prompt Format:
        For conversational responses, use: "User: <prompt>\nBot:"
    """

    max_token_count: int | None = Field(
        default=None,
        description='Maximum tokens to generate (Titan naming). Default: 512.',
    )


class TextEmbeddingConfig(BaseModel):
    """Configuration for text embedding requests.

    See: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html

    Attributes:
        dimensions: Output embedding dimensions (model-dependent).
        normalize: Whether to normalize the output vector.
        input_type: Type of input for Cohere models: search_document, search_query, etc.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    dimensions: int | None = Field(
        default=None,
        description='Output embedding dimensions.',
    )
    normalize: bool | None = Field(
        default=None,
        description='Whether to normalize the output vector.',
    )
    input_type: str | None = Field(
        default=None,
        description='Input type for Cohere: search_document, search_query, classification, clustering.',
    )


__all__ = [
    'AI21JambaConfig',
    # Model-Specific Configs (16 providers)
    'AmazonNovaConfig',
    'AnthropicConfig',
    # Enums
    'AnthropicEffort',
    'AnthropicToolChoice',
    # Base Config
    'BedrockConfig',
    'CohereConfig',
    'CohereSafetyMode',
    'CohereToolChoice',
    'DeepSeekConfig',
    # Mixins
    'GenkitCommonConfigMixin',
    'GoogleGemmaConfig',
    'MetaLlamaConfig',
    'MiniMaxConfig',
    'MistralConfig',
    'MistralToolChoice',
    'MoonshotConfig',
    'NvidiaConfig',
    'OpenAIConfig',
    'QwenConfig',
    'StabilityAspectRatio',
    'StabilityConfig',
    'StabilityMode',
    'StabilityOutputFormat',
    # Embedding Config
    'TextEmbeddingConfig',
    'TitanConfig',
    'WriterConfig',
]
