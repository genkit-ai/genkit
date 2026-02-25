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

"""Microsoft Foundry configuration types for Genkit.

This module defines configuration schemas that align with the Microsoft Foundry/
Azure AI Foundry Model Inference API and model-specific parameters.

Design Rationale:
    We use static configuration classes with `extra='allow'` rather than dynamic
    parameter discovery for several reasons:

    1. **API Limitation**: Azure AI Foundry's Model Info API (`GET /info`) only
       returns basic metadata (model_name, model_type, model_provider_name) and
       does NOT expose supported parameters, capabilities, or parameter constraints.

    2. **Type Safety**: Static configs provide IDE autocompletion, type checking,
       and validation for known parameters with documented constraints.

    3. **Forward Compatibility**: The `extra='allow'` Pydantic setting allows any
       additional parameters to pass through, supporting new/undocumented params.

    4. **Runtime Flexibility**: Azure's `extra-parameters` header controls how
       unknown parameters are handled: 'error', 'drop', or 'pass-through'.

    Users can pass any parameter - known ones get validation, unknown ones pass
    through to the backend model for acceptance or rejection.

Common Parameters NOT in OpenAI-Compatible API:
    When using models through Azure AI Foundry's OpenAI-compatible interface
    (MicrosoftFoundryConfig), some native model parameters are NOT available:

    +-------------------+------------------+--------------------------------------+
    | Parameter         | Native API       | OpenAI Alternative                   |
    +-------------------+------------------+--------------------------------------+
    | top_k             | Anthropic,       | Use `top_p` (nucleus sampling)       |
    |                   | Google, Llama    | instead. Similar effect.             |
    +-------------------+------------------+--------------------------------------+
    | repetition_penalty| HuggingFace TGI, | Use `frequency_penalty` and/or       |
    |                   | Llama, Falcon    | `presence_penalty` instead.          |
    +-------------------+------------------+--------------------------------------+
    | do_sample         | HuggingFace TGI  | Set `temperature > 0` for sampling,  |
    |                   |                  | `temperature = 0` for greedy.        |
    +-------------------+------------------+--------------------------------------+
    | typical_p         | HuggingFace TGI  | No direct equivalent. Use `top_p`.   |
    +-------------------+------------------+--------------------------------------+
    | best_of           | Some models      | Use `n` to generate multiple and     |
    |                   |                  | select the best client-side.         |
    +-------------------+------------------+--------------------------------------+
    | max_new_tokens    | HuggingFace TGI  | Use `max_tokens` instead.            |
    +-------------------+------------------+--------------------------------------+
    | random_seed       | Mistral          | Use `seed` instead.                  |
    +-------------------+------------------+--------------------------------------+
    | safe_prompt       | Mistral          | Not available via OpenAI interface.  |
    +-------------------+------------------+--------------------------------------+
    | thinking          | Anthropic,       | Use `reasoning_effort` for OpenAI    |
    |                   | DeepSeek         | o-series, or pass through directly.  |
    +-------------------+------------------+--------------------------------------+

    To use model-specific parameters, either:
    1. Use the model-specific config class (e.g., AnthropicConfig, LlamaConfig)
    2. Pass parameters directly - they'll flow through via `extra='allow'`
    3. Access the model through its native API instead of Azure AI Foundry

Supported Model Families (Top 30):
    1. OpenAI (GPT-4o, o1, o3, o4 series)
    2. Mistral AI (Mistral Large, Small, Mixtral, Codestral)
    3. Meta Llama (Llama 3.1, 3.2, 3.3, 4)
    4. Cohere (Command R, Command R+, Command A)
    5. DeepSeek (DeepSeek V3, DeepSeek Reasoner)
    6. Microsoft Phi (Phi-3, Phi-3.5, Phi-4)
    7. Anthropic Claude (Opus, Sonnet, Haiku)
    8. AI21 Labs Jamba (Jamba Large, Mini)
    9. xAI Grok (Grok 3, Grok 4)
    10. NVIDIA NIM (Nemotron, various)
    11. Google Gemma (Gemma 2, Gemma 3)
    12. Alibaba Qwen (Qwen 2.5, Qwen 3)
    13. Databricks DBRX
    14. TII Falcon (Falcon 3, Falcon 2)
    15. IBM Granite (Granite 3, Granite Code)
    16. G42 Jais (Arabic LLM)
    17. BigCode StarCoder (StarCoder 2, StarChat)
    18. Stability AI StableLM
    19. MosaicML MPT
    20. TimesFM / Chronos (Time Series)
    21. 01.AI Yi (Yi-1.5, Yi-34B)
    22. Zhipu AI GLM (GLM-4, ChatGLM)
    23. Baichuan (Baichuan 2)
    24. Shanghai AI Lab InternLM
    25. Snowflake Arctic
    26. Writer Palmyra
    27. Reka (Reka Core, Flash, Edge)
    28. OpenBMB MiniCPM
    29. Inflection Pi
    30. Salesforce XGen / CodeGen

See Also:
    Azure AI Foundry Documentation:
        - Microsoft Foundry Docs: https://learn.microsoft.com/en-us/azure/ai-foundry/
        - Model Catalog: https://ai.azure.com/catalog/models
        - SDK Overview: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/sdk-overview
        - Azure AI Model Inference API: https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/
        - Chat Completions API: https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/
        - Model Info API: https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-model-info/

    Model-Specific Documentation:
        - OpenAI: https://platform.openai.com/docs/api-reference/chat/create
        - Mistral: https://docs.mistral.ai/capabilities/completion/
        - Cohere: https://docs.cohere.com/v2/reference/chat
        - DeepSeek: https://api-docs.deepseek.com/api/create-chat-completion
        - Llama: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-models-llama
        - Anthropic: https://docs.anthropic.com/en/api/messages
        - AI21 Jamba: https://docs.ai21.com/reference/jamba-1-6-api-ref
        - xAI Grok: https://docs.x.ai/docs/api-reference
        - NVIDIA NIM: https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
        - Qwen: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference
        - Falcon: https://ai.azure.com/catalog/models/tiiuae-falcon3-1b-instruct
        - Databricks DBRX: https://docs.databricks.com/en/machine-learning/foundation-model-apis/api-reference.html
        - Yi: https://platform.01.ai/docs
        - GLM: https://open.bigmodel.cn/dev/api
        - Reka: https://docs.reka.ai/
        - Writer: https://dev.writer.com/api-guides/chat-completion
"""

import sys
from typing import Any, ClassVar, Literal

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class VisualDetailLevel(StrEnum):
    """Visual detail level for image processing.

    Controls the level of visual detail when processing image embeddings.
    Lower detail levels decrease token usage.
    """

    AUTO = 'auto'
    LOW = 'low'
    HIGH = 'high'


class ReasoningEffort(StrEnum):
    """Reasoning effort level for reasoning models (OpenAI o1, o3, o4 series).

    Controls how much effort the model spends on reasoning before responding.
    Higher values produce more thorough reasoning but use more tokens.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort
    """

    NONE = 'none'
    MINIMAL = 'minimal'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    XHIGH = 'xhigh'


class Verbosity(StrEnum):
    """Verbosity level for model responses.

    Controls how verbose the model's response will be.
    Lower values produce more concise responses.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-verbosity
    """

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


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


class DeepSeekThinkingType(StrEnum):
    """Thinking mode for DeepSeek Reasoner models.

    Controls whether reasoning/thinking is enabled.

    See: https://api-docs.deepseek.com/api/create-chat-completion
    """

    ENABLED = 'enabled'
    DISABLED = 'disabled'


class AnthropicServiceTier(StrEnum):
    """Service tier for Anthropic Claude models.

    Controls whether to use priority or standard capacity.

    See: https://docs.anthropic.com/en/api/service-tiers
    """

    STANDARD = 'standard'
    PRIORITY = 'priority'


class GenkitCommonConfigMixin(BaseModel):
    """Genkit common configuration parameters mixin.

    These parameters match the Genkit core GenerationCommonConfigSchema and are
    expected by the Genkit DevUI for proper rendering of the config pane.

    Reference:
        - JS Schema: js/ai/src/model-types.ts (GenerationCommonConfigSchema)
        - Python Schema: genkit/core/typing.py (GenerationCommonConfig)

    When creating model configs, inherit from this mixin (via MicrosoftFoundryConfig)
    to ensure DevUI compatibility.

    Parameters:
        version: A specific version of the model family (e.g., 'gemini-2.0-flash').
        temperature: Controls randomness in token selection (0.0-2.0).
        max_output_tokens: Maximum number of tokens to generate.
        top_k: Maximum number of tokens to consider when sampling.
        top_p: Nucleus sampling probability mass (0.0-1.0).
        stop_sequences: Up to 5 strings that will stop output generation.
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
        le=2.0,
        description='Controls randomness in token selection.',
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
        description='Nucleus sampling probability mass.',
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        description='Up to 5 strings that will stop output generation.',
    )


class MicrosoftFoundryConfig(GenkitCommonConfigMixin):
    """Base Microsoft Foundry configuration for Genkit.

    Combines:
    - **GenkitCommonConfigMixin**: Standard Genkit parameters for DevUI compatibility
    - **OpenAI-style parameters**: For Azure AI Foundry API compatibility

    Use model-specific configs (MistralConfig, LlamaConfig, etc.) for additional
    model-specific parameters. All model configs inherit from this base.

    Official Documentation:
        - Azure AI Foundry: https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-chat-completions/
        - OpenAI: https://platform.openai.com/docs/api-reference/chat/create
    """

    # OpenAI/Azure AI compatible parameters
    # Note: temperature, top_p, etc. are inherited from GenkitCommonConfigMixin
    model: str | None = None
    max_tokens: int | None = Field(
        default=None,
        description='Maximum tokens (OpenAI-style). Use max_output_tokens for Genkit compatibility.',
    )
    max_completion_tokens: int | None = None
    n: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = Field(
        default=None,
        description='Stop sequences (OpenAI-style). Use stop_sequences for Genkit compatibility.',
    )
    stream: bool | None = None

    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    seed: int | None = None
    user: str | None = None

    response_format: dict[str, Any] | None = None
    modalities: list[Literal['text', 'audio']] | None = None

    visual_detail_level: VisualDetailLevel | None = None

    reasoning_effort: ReasoningEffort | None = None
    verbosity: Verbosity | None = None

    parallel_tool_calls: bool | None = None


OpenAIConfig = MicrosoftFoundryConfig
"""OpenAI model configuration. Alias for MicrosoftFoundryConfig."""


class MistralConfig(MicrosoftFoundryConfig):
    """Configuration for Mistral AI models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Mistral-specific params.

    Supports: Mistral Large, Mistral Small, Mistral 7B, Mixtral, Codestral, etc.

    Official Documentation:
        - Mistral API: https://docs.mistral.ai/capabilities/completion/
        - Sampling Parameters: https://docs.mistral.ai/capabilities/completion/sampling

    Note: `mistral-large-2512` does not support the `n` parameter.
    """

    random_seed: int | None = None
    """Mistral uses random_seed instead of seed."""

    safe_prompt: bool | None = None
    """Enable provider safety additions to reduce risky outputs."""


class LlamaConfig(MicrosoftFoundryConfig):
    """Configuration for Meta Llama models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Llama-specific params.

    Supports: Llama 3.1, Llama 3.2, Llama 3.3, Llama 4, etc.

    Official Documentation:
        - Azure Llama: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-models-llama
        - Model Catalog: https://ai.azure.com/catalog/models/meta-llama-meta-llama-3-8b-instruct
    """

    max_new_tokens: int | None = Field(default=None)
    """Maximum tokens to generate. Llama uses max_new_tokens instead of max_tokens."""

    repetition_penalty: float | None = None
    """Controls likelihood of repetition. Values > 1 reduce repetition."""

    do_sample: bool | None = None
    """Whether to use sampling vs greedy decoding. Default: false."""

    best_of: int | None = None
    """Generate multiple sequences and return the best one."""

    typical_p: float | None = None
    """Typical probability of a token for locally typical sampling."""

    truncate: bool | None = None
    """Truncate input to max model length. Default: true."""

    return_full_text: bool | None = None
    """Return full text or only generated part. Default: false."""

    details: bool | None = None
    """Return generation details. Default: false."""

    watermark: bool | None = None
    """Add watermark to generation. Default: false."""


class CohereConfig(MicrosoftFoundryConfig):
    """Configuration for Cohere models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Cohere-specific params.

    Supports: Command R, Command R+, Command A, etc.

    Official Documentation:
        - Cohere Chat API: https://docs.cohere.com/v2/reference/chat
        - Safety Modes: https://docs.cohere.com/v2/docs/safety-modes
    """

    k: int | None = Field(default=None, ge=0, le=500)
    """Top-k sampling (Cohere-specific). When k=0, k-sampling is disabled. Default: 0."""

    p: float | None = Field(default=None, ge=0.01, le=0.99)
    """Nucleus sampling probability mass (Cohere-specific). Default: 0.75."""

    safety_mode: CohereSafetyMode | None = None
    """Safety instruction mode: CONTEXTUAL, STRICT, or OFF."""

    tool_choice: CohereToolChoice | None = None
    """Force tool use: REQUIRED or NONE."""

    strict_tools: bool | None = None
    """Force tool calls to follow tool definition strictly (Beta)."""

    documents: list[str | dict[str, Any]] | None = None
    """Documents for RAG-based generation with citations."""

    citation_options: dict[str, Any] | None = None
    """Options for controlling citation generation."""

    thinking: dict[str, Any] | None = None
    """Configuration for reasoning features."""

    priority: int | None = Field(default=None, ge=0, le=999)
    """Request priority. Lower = higher priority. Default: 0."""


class DeepSeekConfig(MicrosoftFoundryConfig):
    """Configuration for DeepSeek models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus DeepSeek-specific params.

    Supports: DeepSeek V3, DeepSeek Reasoner, DeepSeek Chat, etc.

    Official Documentation:
        - DeepSeek API: https://api-docs.deepseek.com/api/create-chat-completion
        - Chat Prefix Completion: https://api-docs.deepseek.com/guides/chat_prefix_completion

    Note: The API is compatible with OpenAI SDKs.
    """

    thinking: dict[str, Any] | None = None
    """Controls thinking/reasoning mode for 'deepseek-reasoner'."""

    prefix: bool | None = None
    """Force model to start with supplied assistant message content (Beta)."""


class PhiConfig(MicrosoftFoundryConfig):
    """Configuration for Microsoft Phi models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Phi-3, Phi-3.5, Phi-4, etc.

    Microsoft Phi models generally follow the OpenAI-compatible interface.
    """

    pass  # Phi uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class AnthropicConfig(MicrosoftFoundryConfig):
    """Configuration for Anthropic Claude models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Anthropic-specific params.

    Supports: Claude Opus, Claude Sonnet, Claude Haiku (claude-3.5, claude-3.7, claude-4)

    Official Documentation:
        - Anthropic Messages API: https://docs.anthropic.com/en/api/messages
        - Create Message: https://docs.anthropic.com/en/api/messages/create

    Note: Anthropic uses a different API structure than OpenAI. Azure AI Foundry
    may provide OpenAI-compatible endpoints for Claude models.
    """

    thinking: dict[str, Any] | None = None
    """Configuration for enabling Claude's extended thinking capability."""

    metadata: dict[str, Any] | None = None
    """Object describing metadata about the request."""

    service_tier: AnthropicServiceTier | None = None
    """Determines whether to use priority or standard capacity."""


class AI21JambaConfig(MicrosoftFoundryConfig):
    """Configuration for AI21 Labs Jamba models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Jamba Large, Jamba Mini, Jamba 1.5, Jamba 1.6

    Official Documentation:
        - AI21 Jamba API: https://docs.ai21.com/reference/jamba-1-6-api-ref
        - AWS Bedrock Jamba: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html
    """

    pass  # Jamba uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class GrokConfig(MicrosoftFoundryConfig):
    """Configuration for xAI Grok models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Grok 3, Grok 4, etc.

    Official Documentation:
        - xAI API Reference: https://docs.x.ai/docs/api-reference
        - Reasoning: https://docs.x.ai/docs/guides/reasoning

    Note: Reasoning models (like Grok 4) do NOT support presence_penalty,
    frequency_penalty, or stop parameters.
    """

    pass  # Grok uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class NvidiaConfig(MicrosoftFoundryConfig):
    """Configuration for NVIDIA NIM models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus NVIDIA-specific params.

    Supports: Nemotron, various NVIDIA-optimized models

    Official Documentation:
        - NVIDIA NIM API: https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html
        - Sampling Parameters: https://docs.nvidia.com/nim/vision-language-models/latest/sampling-params.html
    """

    repetition_penalty: float | None = None
    """Penalizes repeated tokens. Values > 1 reduce repetition."""


class GemmaConfig(MicrosoftFoundryConfig):
    """Configuration for Google Gemma models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Gemma 2, Gemma 3, etc.

    Official Documentation:
        - Vertex AI GenerationConfig: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GenerationConfig
        - Azure Gemma: https://ai.azure.com/catalog/models (search for Gemma)
    """

    pass  # Gemma uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class QwenConfig(MicrosoftFoundryConfig):
    """Configuration for Alibaba Qwen models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Qwen-specific params.

    Supports: Qwen 2.5, Qwen 3, Qwen-VL, etc.

    Official Documentation:
        - Qwen API: https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference
        - Qwen Chat: https://qwen-ai.chat/docs/api/

    Note: Qwen uses an OpenAI-compatible interface.
    """

    repetition_penalty: float | None = None
    """Penalizes repeated tokens. Values > 1.0 reduce repetition."""


class DbrxConfig(MicrosoftFoundryConfig):
    """Configuration for Databricks DBRX model on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    DBRX is a mixture-of-experts model with 132B total parameters (36B active).

    Official Documentation:
        - Databricks Foundation Model APIs: https://docs.databricks.com/en/machine-learning/foundation-model-apis/api-reference.html
        - DBRX Model Card: https://github.com/databricks/dbrx

    Note: DBRX supports a maximum context length of 32,768 tokens.
    """

    pass  # DBRX uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class FalconConfig(MicrosoftFoundryConfig):
    """Configuration for TII Falcon models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Falcon-specific params.

    Supports: Falcon 3, Falcon 2, Falcon 40B, Falcon 180B, etc.

    Official Documentation:
        - Azure Falcon: https://ai.azure.com/catalog/models/tiiuae-falcon3-1b-instruct
        - NVIDIA NIM Falcon: https://docs.api.nvidia.com/nim/reference/tiiuae-falcon3-7b-instruct

    Note: Falcon models use Text Generation Inference (TGI) API with OpenAI compatibility.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    do_sample: bool | None = None
    """Activate logits sampling. Default: false (greedy decoding)."""

    repetition_penalty: float | None = None
    """Penalize repeated tokens. Values > 1 reduce repetition."""


class GraniteConfig(MicrosoftFoundryConfig):
    """Configuration for IBM Granite models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Granite-specific params.

    Supports: Granite 3, Granite Code, Granite Guardian, etc.

    Official Documentation:
        - IBM Granite: https://www.ibm.com/granite
        - Azure Granite: https://ai.azure.com/catalog/models (search for Granite)

    Note: Granite models generally follow the OpenAI-compatible interface.
    """

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""


class JaisConfig(MicrosoftFoundryConfig):
    """Configuration for G42 Jais models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Jais-specific params.

    Jais is an Arabic-English bilingual large language model developed by G42.

    Supports: Jais-30b, Jais-13b, Jais-7b, etc.

    Official Documentation:
        - Azure Jais: https://ai.azure.com/catalog/models (search for Jais)
        - G42 Jais: https://www.g42.ai/research/jais

    Note: Jais models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    do_sample: bool | None = None
    """Activate logits sampling."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""


class StarCoderConfig(MicrosoftFoundryConfig):
    """Configuration for BigCode StarCoder models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus StarCoder-specific params.

    Supports: StarCoder 2, StarChat, StarCoder 15B, etc.

    Official Documentation:
        - BigCode: https://www.bigcode-project.org/
        - HuggingFace StarCoder: https://huggingface.co/bigcode

    Note: StarCoder models are optimized for code generation and typically use TGI.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    do_sample: bool | None = None
    """Activate logits sampling."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""

    return_full_text: bool | None = None
    """Return the full text including prompt, or only the generated part."""


class StableLMConfig(MicrosoftFoundryConfig):
    """Configuration for Stability AI StableLM models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus StableLM-specific params.

    Supports: StableLM 2, StableLM Zephyr, etc.

    Official Documentation:
        - Stability AI: https://stability.ai/
        - HuggingFace StableLM: https://huggingface.co/stabilityai

    Note: StableLM models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    do_sample: bool | None = None
    """Activate logits sampling."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""


class MptConfig(MicrosoftFoundryConfig):
    """Configuration for MosaicML MPT models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus MPT-specific params.

    Supports: MPT-30B, MPT-7B, MPT-7B-Instruct, etc.

    Official Documentation:
        - MosaicML (Databricks): https://www.mosaicml.com/
        - HuggingFace MPT: https://huggingface.co/mosaicml

    Note: MPT models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    do_sample: bool | None = None
    """Activate logits sampling."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""


class TimeSeriesConfig(MicrosoftFoundryConfig):
    """Configuration for time series forecasting models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus time series-specific params.

    Supports: TimesFM, Chronos, etc.

    These models are specialized for time series forecasting rather than
    traditional text generation.

    Official Documentation:
        - Google TimesFM: https://github.com/google-research/timesfm
        - Amazon Chronos: https://github.com/amazon-science/chronos-forecasting
    """

    prediction_length: int | None = None
    """Number of time steps to forecast."""

    context_length: int | None = None
    """Number of historical time steps to use as context."""

    num_samples: int | None = None
    """Number of sample paths to generate for probabilistic forecasting."""


class YiConfig(MicrosoftFoundryConfig):
    """Configuration for 01.AI Yi models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Yi-specific params.

    Supports: Yi-1.5, Yi-34B, Yi-6B, etc.

    Official Documentation:
        - 01.AI Platform: https://platform.01.ai/docs
        - HuggingFace Yi: https://huggingface.co/01-ai

    Note: Yi models use an OpenAI-compatible interface.
    """

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""


class GlmConfig(MicrosoftFoundryConfig):
    """Configuration for Zhipu AI GLM models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus GLM-specific params.

    Supports: GLM-4, ChatGLM, GLM-4V (vision), etc.

    Official Documentation:
        - Zhipu AI: https://open.bigmodel.cn/dev/api
        - HuggingFace GLM: https://huggingface.co/THUDM

    Note: GLM models use an OpenAI-compatible interface.
    """

    do_sample: bool | None = None
    """Whether to use sampling. Set to false for greedy decoding."""


class BaichuanConfig(MicrosoftFoundryConfig):
    """Configuration for Baichuan models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Baichuan-specific params.

    Supports: Baichuan 2, Baichuan-13B, Baichuan-7B, etc.

    Official Documentation:
        - Baichuan: https://www.baichuan-ai.com/
        - HuggingFace Baichuan: https://huggingface.co/baichuan-inc

    Note: Baichuan models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""

    do_sample: bool | None = None
    """Activate logits sampling."""


class InternLMConfig(MicrosoftFoundryConfig):
    """Configuration for Shanghai AI Lab InternLM models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus InternLM-specific params.

    Supports: InternLM2, InternLM-20B, InternLM-7B, etc.

    Official Documentation:
        - InternLM: https://github.com/InternLM/InternLM
        - HuggingFace InternLM: https://huggingface.co/internlm

    Note: InternLM models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""

    do_sample: bool | None = None
    """Activate logits sampling."""


class ArcticConfig(MicrosoftFoundryConfig):
    """Configuration for Snowflake Arctic models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Arctic is Snowflake's enterprise-grade LLM optimized for SQL and data tasks.

    Official Documentation:
        - Snowflake Arctic: https://www.snowflake.com/blog/arctic-open-efficient-enterprise-llms/
        - HuggingFace Arctic: https://huggingface.co/Snowflake

    Note: Arctic uses an OpenAI-compatible interface.
    """

    pass  # Arctic uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class WriterConfig(MicrosoftFoundryConfig):
    """Configuration for Writer Palmyra models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus Writer-specific params.

    Supports: Palmyra X, Palmyra Med, Palmyra Fin, etc.

    Official Documentation:
        - Writer API: https://dev.writer.com/api-guides/chat-completion
        - Writer: https://writer.com/

    Note: Writer models use an OpenAI-compatible interface.
    """

    best_of: int | None = None
    """Generate best_of completions and return the best."""


class RekaConfig(MicrosoftFoundryConfig):
    """Configuration for Reka AI models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Reka Core, Reka Flash, Reka Edge

    Official Documentation:
        - Reka API: https://docs.reka.ai/
        - Reka: https://www.reka.ai/

    Note: Reka models are multimodal and support text, images, and video.
    """

    pass  # Reka uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class MiniCPMConfig(MicrosoftFoundryConfig):
    """Configuration for OpenBMB MiniCPM models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus MiniCPM-specific params.

    MiniCPM is a series of efficient small language models.

    Supports: MiniCPM-2B, MiniCPM-V (vision), etc.

    Official Documentation:
        - MiniCPM: https://github.com/OpenBMB/MiniCPM
        - HuggingFace MiniCPM: https://huggingface.co/openbmb

    Note: MiniCPM models typically use the TGI or OpenAI-compatible interface.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""

    do_sample: bool | None = None
    """Activate logits sampling."""


class InflectionConfig(MicrosoftFoundryConfig):
    """Configuration for Inflection Pi models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig.

    Supports: Pi (Inflection's conversational AI)

    Official Documentation:
        - Inflection: https://inflection.ai/

    Note: Inflection models use an OpenAI-compatible interface.
    """

    pass  # Inflection uses standard OpenAI-compatible parameters from MicrosoftFoundryConfig


class XGenConfig(MicrosoftFoundryConfig):
    """Configuration for Salesforce XGen / CodeGen models on Azure AI Foundry.

    Inherits all Genkit common parameters from MicrosoftFoundryConfig plus XGen-specific params.

    Supports: XGen-7B, CodeGen 2.5, CodeGen 16B, etc.

    Official Documentation:
        - Salesforce XGen: https://blog.salesforceairesearch.com/xgen/
        - HuggingFace XGen: https://huggingface.co/Salesforce

    Note: XGen models are optimized for long context and code generation.
    """

    max_new_tokens: int | None = None
    """Maximum tokens to generate (TGI naming)."""

    repetition_penalty: float | None = None
    """Penalizes repeated tokens."""

    do_sample: bool | None = None
    """Activate logits sampling."""


class TextEmbeddingConfig(BaseModel):
    """Configuration for text embedding requests.

    See: https://learn.microsoft.com/en-us/rest/api/aifoundry/model-inference/get-embeddings/

    Attributes:
        dimensions: Output embedding dimensions (model-dependent).
        encoding_format: Output encoding format ('float' or 'base64').
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra='allow',
        populate_by_name=True,
        alias_generator=to_camel,
    )

    dimensions: int | None = None
    encoding_format: str | None = None


__all__ = [
    'AI21JambaConfig',
    'AnthropicConfig',
    'AnthropicServiceTier',
    'ArcticConfig',
    'BaichuanConfig',
    'CohereConfig',
    'CohereSafetyMode',
    'CohereToolChoice',
    'DbrxConfig',
    'DeepSeekConfig',
    'DeepSeekThinkingType',
    'FalconConfig',
    'GemmaConfig',
    # Mixins
    'GenkitCommonConfigMixin',
    'GlmConfig',
    'GraniteConfig',
    'GrokConfig',
    'InflectionConfig',
    'InternLMConfig',
    'JaisConfig',
    'LlamaConfig',
    # Base/OpenAI Configs (1-2)
    'MicrosoftFoundryConfig',
    'MiniCPMConfig',
    # Model-Specific Configs (3-30)
    'MistralConfig',
    'MptConfig',
    'NvidiaConfig',
    'OpenAIConfig',
    'PhiConfig',
    'QwenConfig',
    'ReasoningEffort',
    'RekaConfig',
    'StableLMConfig',
    'StarCoderConfig',
    # Embedding Config
    'TextEmbeddingConfig',
    'TimeSeriesConfig',
    'Verbosity',
    # Enums
    'VisualDetailLevel',
    'WriterConfig',
    'XGenConfig',
    'YiConfig',
]
