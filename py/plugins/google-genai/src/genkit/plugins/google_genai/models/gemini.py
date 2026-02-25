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

"""Gemini models for use with Genkit.

# Naming convention
Gemini models follow the following naming conventions:

                             +------- Tier/Variant (e.g., pro, flash)
                             |      +---------- Modifier (Optional, e.g., exp)
                             |      |         +--- Date/Snapshot ID (Optional)
                             v      v         v
        gemini - <VER> - <TIER> [-MOD] [-DATE]
          ^        ^           ^
          |        |           |
(Family)--+        |           +-- Size Specifier (Optional, e.g., -8b,
          |        |               often follows TIER like 'flash')
          |        |
          +--------+---------- Version (Major generation, e.g., 1.0, 1.5, 2.0)


## Examples

gemini - 1.5 - flash - 8b
  ^      ^      ^      ^
  |      |      |      +-- Size Specifier
  |      |      +--------- Tier/Variant
  |      +----------------- Version
  +------------------------ Family

gemini - 2.0 - pro - exp - 02-05
  ^      ^      ^     ^      ^
  |      |      |     |      +-- Date/Snapshot ID
  |      |      |     +--------- Modifier
  |      |      +---------------- Tier/Variant
  |      +------------------------ Version
  +------------------------------- Family

## Terminology

Family (`gemini`)
: The base name identifying the overarching group or brand of related AI models
  developed by Google (e.g., Gemini).

Version Number (e.g., `1.0`, `1.5`, `2.0`, `2.5`)
: Indicates the major generation or release cycle of the model within the
  family. Higher numbers typically denote newer iterations, often incorporating
  significant improvements, architectural changes, or new capabilities compared
  to previous versions.

Tier / Variant (e.g., `pro`, `flash`)
: Distinguishes models within the same generation based on specific
  characteristics like performance profile, size, speed, efficiency, or intended
  primary use case.

  * **`pro`**: Generally indicates a high-capability, powerful, and versatile
    model within its generation, suitable for a wide range of complex tasks.

  * **`flash`**: Often signifies a model optimized for speed, latency, and
    cost-efficiency, potentially offering a different balance of performance
    characteristics compared to the `pro` variant.

Size Specifier (e.g., `8b`)
: An optional component, frequently appended to a Tier/Variant (like `flash`),
  providing more specific detail about the model's scale. This often relates to
  the approximate number of parameters (e.g., `8b` likely suggests 8 billion
  parameters), influencing its performance and resource requirements.

Modifier (e.g., `exp`)
: An optional flag indicating the model's release status, stability, or intended
  audience.

  * **`exp`**: Stands for "Experimental". Models marked with `exp` are typically
    previews or early releases. They are subject to change, updates, or removal
    without the standard notice periods applied to stable models, and they lack
    long-term stability guarantees, making them generally unsuitable for
    production systems requiring stability.

Date / Snapshot ID (e.g., `02-05`, `03-25`)
: An optional identifier, commonly seen with experimental (`exp`) models. It
  likely represents a specific build date (often in MM-DD format) or a unique
  snapshot identifier, helping to distinguish between different iterations or
  releases within the experimental track.

# Model support

The following models are currently supported by GoogleAI API:

| Model                                | Description                          | Status     |
|--------------------------------------|--------------------------------------|------------|
| `gemini-1.5-pro`                     | Gemini 1.5 Pro                       | Deprecated |
| `gemini-1.5-flash`                   | Gemini 1.5 Flash                     | Deprecated |
| `gemini-1.5-flash-8b`                | Gemini 1.5 Flash 8B                  | Deprecated |
| `gemini-2.0-flash`                   | Gemini 2.0 Flash                     | Supported  |
| `gemini-2.0-flash-lite`              | Gemini 2.0 Flash Lite                | Supported  |
| `gemini-2.0-pro-exp-02-05`           | Gemini 2.0 Pro Exp 02-05             | Supported  |
| `gemini-2.5-pro-exp-03-25`           | Gemini 2.5 Pro Exp 03-25             | Supported  |
| `gemini-2.0-flash-exp`               | Gemini 2.0 Flash Experimental        | Supported  |
| `gemini-2.0-flash-thinking-exp-01-21`| Gemini 2.0 Flash Thinking Exp 01-21  | Supported  |
| `gemini-2.5-pro-preview-03-25`       | Gemini 2.5 Pro Preview 03-25         | Supported  |
| `gemini-2.5-pro-preview-05-06`       | Gemini 2.5 Pro Preview 05-06         | Supported  |


The following models are currently supported by VertexAI API:

| Model                                | Description                          | Status       |
|--------------------------------------|--------------------------------------|--------------|
| `gemini-1.5-pro`                     | Gemini 1.5 Pro                       | Deprecated   |
| `gemini-1.5-flash`                   | Gemini 1.5 Flash                     | Deprecated   |
| `gemini-1.5-flash-8b`                | Gemini 1.5 Flash 8B                  | Deprecated   |
| `gemini-2.0-flash`                   | Gemini 2.0 Flash                     | Supported    |
| `gemini-2.0-flash-lite`              | Gemini 2.0 Flash Lite                | Supported    |
| `gemini-2.0-pro-exp-02-05`           | Gemini 2.0 Pro Exp 02-05             | Supported    |
| `gemini-2.5-pro-exp-03-25`           | Gemini 2.5 Pro Exp 03-25             | Supported    |
| `gemini-2.0-flash-exp`               | Gemini 2.0 Flash Experimental        | Unavailable  |
| `gemini-2.0-flash-thinking-exp-01-21`| Gemini 2.0 Flash Thinking Exp 01-21  | Supported    |
| `gemini-2.5-pro-preview-03-25`       | Gemini 2.5 Pro Preview 03-25         | Supported    |
| `gemini-2.5-pro-preview-05-06`       | Gemini 2.5 Pro Preview 05-06         | Supported  |
"""

import sys
from datetime import datetime, timedelta, timezone

from genkit.plugins.google_genai.models.context_caching.constants import DEFAULT_TTL
from genkit.plugins.google_genai.models.context_caching.utils import generate_cache_key, validate_context_cache_request

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

from functools import cached_property
from typing import Annotated, Any, cast

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError
from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from genkit.ai import (
    ActionRunContext,
)
from genkit.blocks.model import get_basic_usage_stats
from genkit.codec import dump_dict, dump_json
from genkit.core.error import GenkitError, StatusName
from genkit.core.tracing import tracer
from genkit.core.typing import (
    Candidate,
    FinishReason,
)
from genkit.lang.deprecations import (
    deprecated_enum_metafactory,
)
from genkit.plugins.google_genai.models.utils import PartConverter
from genkit.types import (
    Constrained,
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationCommonConfig,
    GenerationUsage,
    Message,
    ModelInfo,
    Part,
    Role,
    Stage,
    Supports,
    TextPart,
    ToolDefinition,
)


class HarmCategory(StrEnum):
    """Harm categories."""

    HARM_CATEGORY_UNSPECIFIED = 'HARM_CATEGORY_UNSPECIFIED'
    HARM_CATEGORY_HATE_SPEECH = 'HARM_CATEGORY_HATE_SPEECH'
    HARM_CATEGORY_SEXUALLY_EXPLICIT = 'HARM_CATEGORY_SEXUALLY_EXPLICIT'
    HARM_CATEGORY_HARASSMENT = 'HARM_CATEGORY_HARASSMENT'
    HARM_CATEGORY_DANGEROUS_CONTENT = 'HARM_CATEGORY_DANGEROUS_CONTENT'


class HarmBlockThreshold(StrEnum):
    """Harm block thresholds."""

    BLOCK_LOW_AND_ABOVE = 'BLOCK_LOW_AND_ABOVE'
    BLOCK_MEDIUM_AND_ABOVE = 'BLOCK_MEDIUM_AND_ABOVE'
    BLOCK_ONLY_HIGH = 'BLOCK_ONLY_HIGH'
    BLOCK_NONE = 'BLOCK_NONE'


class SafetySettingsSchema(BaseModel):
    """Safety settings schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    category: HarmCategory
    threshold: HarmBlockThreshold


class PrebuiltVoiceConfig(BaseModel):
    """Prebuilt voice config."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    voice_name: str | None = Field(None, alias='voiceName')


class FunctionCallingMode(StrEnum):
    """Function calling mode."""

    MODE_UNSPECIFIED = 'MODE_UNSPECIFIED'
    AUTO = 'AUTO'
    ANY = 'ANY'
    NONE = 'NONE'


class FunctionCallingConfig(BaseModel):
    """Function calling config."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    mode: FunctionCallingMode | None = None
    allowed_function_names: list[str] | None = Field(None, alias='allowedFunctionNames')


class ThinkingLevel(StrEnum):
    """Thinking level."""

    MINIMAL = 'MINIMAL'
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class ThinkingConfigSchema(BaseModel):
    """Thinking config schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    include_thoughts: bool | None = Field(None, alias='includeThoughts')
    thinking_budget: int | None = Field(None, alias='thinkingBudget')
    thinking_level: ThinkingLevel | None = Field(None, alias='thinkingLevel')


class FileSearchConfigSchema(BaseModel):
    """File search config schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    file_search_store_names: list[str] | None = Field(None, alias='fileSearchStoreNames')
    metadata_filter: str | None = Field(None, alias='metadataFilter')
    top_k: int | None = Field(None, alias='topK')


class ImageAspectRatio(StrEnum):
    """Image aspect ratio."""

    RATIO_1_1 = '1:1'
    RATIO_2_3 = '2:3'
    RATIO_3_2 = '3:2'
    RATIO_3_4 = '3:4'
    RATIO_4_3 = '4:3'
    RATIO_4_5 = '4:5'
    RATIO_5_4 = '5:4'
    RATIO_9_16 = '9:16'
    RATIO_16_9 = '16:9'
    RATIO_21_9 = '21:9'


class ImageSize(StrEnum):
    """Image size."""

    SIZE_1K = '1K'
    SIZE_2K = '2K'
    SIZE_4K = '4K'


class ImageConfigSchema(BaseModel):
    """Image config schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    aspect_ratio: ImageAspectRatio | None = Field(None, alias='aspectRatio')
    image_size: ImageSize | None = Field(None, alias='imageSize')


class VoiceConfigSchema(BaseModel):
    """Voice config schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)
    prebuilt_voice_config: PrebuiltVoiceConfig | None = Field(None, alias='prebuiltVoiceConfig')


class GeminiConfigSchema(GenerationCommonConfig):
    """Gemini Config Schema."""

    model_config = ConfigDict(extra='allow', populate_by_name=True)

    api_key: str | None = Field(
        None, description='Overrides the plugin-configured API key, if specified.', alias='apiKey', exclude=True
    )
    base_url: str | None = Field(
        None, description='Overrides the plugin-configured or default baseUrl, if specified.', alias='baseUrl'
    )
    api_version: str | None = Field(
        None, description='Overrides the plugin-configured or default apiVersion, if specified.', alias='apiVersion'
    )

    safety_settings: Annotated[
        list[SafetySettingsSchema] | None,
        WithJsonSchema({
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'category': {'type': 'string', 'enum': [e.value for e in HarmCategory]},
                    'threshold': {'type': 'string', 'enum': [e.value for e in HarmBlockThreshold]},
                },
                'required': ['category', 'threshold'],
                'additionalProperties': True,
            },
            'description': (
                'Adjust how likely you are to see responses that could be harmful. '
                'Content is blocked based on the probability that it is harmful.'
            ),
        }),
    ] = Field(
        None,
        alias='safetySettings',
    )

    code_execution: bool | dict[str, Any] | None = Field(
        None, description='Enables the model to generate and run code.', alias='codeExecution'
    )

    context_cache: bool | None = Field(
        None,
        description=(
            'Context caching allows you to save and reuse precomputed input tokens that you wish to use repeatedly.'
        ),
        alias='contextCache',
    )

    function_calling_config: Annotated[
        FunctionCallingConfig | None,
        WithJsonSchema({
            'type': 'object',
            'properties': {
                'mode': {'type': 'string', 'enum': [e.value for e in FunctionCallingMode]},
                'allowedFunctionNames': {'type': 'array', 'items': {'type': 'string'}},
            },
            'description': (
                'Controls how the model uses the provided tools (function declarations). With AUTO (Default) '
                'mode, the model decides whether to generate a natural language response or suggest a function '
                'call based on the prompt and context. With ANY, the model is constrained to always predict a '
                'function call and guarantee function schema adherence. With NONE, the model is prohibited '
                'from making function calls.'
            ),
            'additionalProperties': True,
        }),
    ] = Field(
        None,
        alias='functionCallingConfig',
    )

    response_modalities: list[str] | None = Field(
        None,
        description=(
            "The modalities to be used in response. Only supported for 'gemini-2.0-flash-exp' model at present."
        ),
        alias='responseModalities',
    )

    google_search_retrieval: bool | dict[str, Any] | None = Field(
        None,
        description=(
            'Retrieve public web data for grounding, powered by Google Search. '
            'Note: This feature is not supported on all models. '
            'If you get an error, use the google_search tool instead.'
        ),
        alias='googleSearchRetrieval',
    )

    file_search: Annotated[
        FileSearchConfigSchema | None,
        WithJsonSchema({
            'type': 'object',
            'properties': {
                'fileSearchStoreNames': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'The names of the fileSearchStores to retrieve from. '
                        'Example: fileSearchStores/my-file-search-store-123'
                    ),
                },
                'metadataFilter': {
                    'type': 'string',
                    'description': 'Metadata filter to apply to the semantic retrieval documents and chunks.',
                },
                'topK': {
                    'type': 'integer',
                    'description': 'The number of semantic retrieval chunks to retrieve.',
                },
            },
            'additionalProperties': True,
        }),
    ] = Field(None, alias='fileSearch')

    url_context: bool | dict[str, Any] | None = Field(
        None, description='Return grounding metadata from links included in the query', alias='urlContext'
    )

    # inherited from GenerationCommonConfig:
    # version, temperature, max_output_tokens, top_k, top_p, stop_sequences

    temperature: Annotated[
        float | None,
        WithJsonSchema({
            'type': 'number',
            'minimum': 0.0,
            'maximum': 2.0,
            'description': (
                'Controls the randomness of the output. Values can range over [0.0, 2.0]. The default value is 1.0.'
            ),
        }),
    ] = Field(
        default=None,
        ge=0.0,
        le=2.0,
    )

    top_p: Annotated[
        float | None,
        WithJsonSchema({
            'type': 'number',
            'minimum': 0.0,
            'maximum': 1.0,
            'description': (
                'The maximum cumulative probability of tokens to consider when sampling. '
                'Values can range over [0.0, 1.0]. The default value is 0.95.'
            ),
        }),
    ] = Field(
        default=None,
        alias='topP',
        ge=0.0,
        le=1.0,
    )
    top_k: int | None = Field(  # pyrefly: ignore[bad-override]
        default=None,
        alias='topK',
        description=('The maximum number of tokens to consider when sampling.'),
    )

    thinking_config: Annotated[
        ThinkingConfigSchema | None,
        WithJsonSchema({
            'type': 'object',
            'properties': {
                'includeThoughts': {
                    'type': 'boolean',
                    'description': (
                        'Indicates whether to include thoughts in the response. If true, thoughts are returned only if '
                        'the model supports thought and thoughts are available.'
                    ),
                },
                'thinkingBudget': {
                    'type': 'integer',
                    'description': (
                        'For Gemini 2.5 - Indicates the thinking budget in tokens. 0 is DISABLED. -1 is AUTOMATIC. '
                        'The default values and allowed ranges are model dependent. The thinking budget parameter '
                        'gives the model guidance on the number of thinking tokens it can use when generating a '
                        'response. A greater number of tokens is typically associated with more detailed thinking, '
                        'which is needed for solving more complex tasks.'
                    ),
                },
                'thinkingLevel': {
                    'type': 'string',
                    'enum': [e.value for e in ThinkingLevel],
                    'description': (
                        'For Gemini 3.0 - Indicates the thinking level. A higher level is associated with more '
                        'detailed thinking, which is needed for solving more complex tasks.'
                    ),
                },
            },
            'additionalProperties': True,
        }),
    ] = Field(None, alias='thinkingConfig')

    max_output_tokens: int | None = Field(  # pyrefly: ignore[bad-override]
        default=None, alias='maxOutputTokens', description='Maximum number of tokens to generate.'
    )
    stop_sequences: list[str] | None = Field(default=None, alias='stopSequences', description='Stop sequences.')


class SpeechConfigSchema(BaseModel):
    """Speech config schema."""

    voice_config: VoiceConfigSchema | None = Field(None, alias='voiceConfig')

    http_options: Any | None = Field(None, exclude=True)
    tools: Any | None = Field(None, exclude=True)
    tool_config: Any | None = Field(None, exclude=True)
    response_schema: Any | None = Field(None, exclude=True)
    response_json_schema: Any | None = Field(None, exclude=True)


class GeminiTtsConfigSchema(GeminiConfigSchema):
    """Gemini TTS Config Schema."""

    speech_config: SpeechConfigSchema | None = Field(None, alias='speechConfig')


class GeminiImageConfigSchema(GeminiConfigSchema):
    """Gemini Image Config Schema."""

    image_config: Annotated[
        ImageConfigSchema | None,
        WithJsonSchema({
            'type': 'object',
            'properties': {
                'aspectRatio': {'type': 'string', 'enum': [e.value for e in ImageAspectRatio]},
                'imageSize': {'type': 'string', 'enum': [e.value for e in ImageSize]},
            },
            'additionalProperties': True,
        }),
    ] = Field(None, alias='imageConfig')


class GemmaConfigSchema(GeminiConfigSchema):
    """Gemma Config Schema."""

    # Inherits temperature from GeminiConfigSchema
    temperature: float | None = None


GEMINI_1_5_PRO = ModelInfo(
    label='Google AI - Gemini 1.5 Pro',
    stage=Stage.DEPRECATED,
    versions=[
        'gemini-1.5-pro-latest',
        'gemini-1.5-pro-001',
        'gemini-1.5-pro-002',
    ],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
    ),
)

GEMINI_1_5_FLASH = ModelInfo(
    label='Google AI - Gemini 1.5 Flash',
    stage=Stage.DEPRECATED,
    versions=[
        'gemini-1.5-flash-latest',
        'gemini-1.5-flash-001',
        'gemini-1.5-flash-002',
    ],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_1_5_FLASH_8B = ModelInfo(
    label='Google AI - Gemini 1.5 Flash',
    stage=Stage.DEPRECATED,
    versions=['gemini-1.5-flash-8b-latest', 'gemini-1.5-flash-8b-001'],
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_0_FLASH = ModelInfo(
    label='Google AI - Gemini 2.0 Flash',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_0_FLASH_LITE = ModelInfo(
    label='Google AI - Gemini 2.0 Flash Lite',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_0_PRO_EXP_02_05 = ModelInfo(
    label='Google AI - Gemini 2.0 Pro Exp 02-05',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_0_FLASH_EXP_IMAGEN = ModelInfo(
    label='Google AI - Gemini 2.0 Flash Experimental',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_0_FLASH_THINKING_EXP_01_21 = ModelInfo(
    label='Google AI - Gemini 2.0 Flash Thinking Exp 01-21',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_5_PRO_EXP_03_25 = ModelInfo(
    label='Google AI - Gemini 2.5 Pro Exp 03-25',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_5_PRO_PREVIEW_03_25 = ModelInfo(
    label='Google AI - Gemini 2.5 Pro Preview 03-25',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_5_PRO_PREVIEW_05_06 = ModelInfo(
    label='Google AI - Gemini 2.5 Pro Preview 05-06',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GEMINI_2_5_FLASH_PREVIEW_04_17 = ModelInfo(
    label='Google AI - Gemini 2.5 Flash Preview 04-17',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GENERIC_GEMINI_MODEL = ModelInfo(
    label='Google AI - Gemini',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)

GENERIC_TTS_MODEL = ModelInfo(
    label='Google AI - Gemini TTS',
    supports=Supports(
        multiturn=False,
        media=False,
        tools=False,
        tool_choice=False,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
    ),
)

GENERIC_IMAGE_MODEL = ModelInfo(
    label='Google AI - Gemini Image',
    supports=Supports(
        multiturn=False,
        media=True,
        tools=False,
        tool_choice=False,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['media'],
    ),
)

GENERIC_GEMMA_MODEL = ModelInfo(
    label='Google AI - Gemma',
    supports=Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.NO_TOOLS,
        output=['text', 'json'],
    ),
)


Deprecations = deprecated_enum_metafactory({})


class VertexAIGeminiVersion(StrEnum, metaclass=Deprecations):
    """VertexAIGemini models.

    Model Support:

    | Model                                | Description                          | Status       |
    |--------------------------------------|--------------------------------------|--------------|
    | `gemini-1.5-flash-8b`                | Gemini 1.5 Flash 8B                  | Deprecated   |
    | `gemini-1.5-flash`                   | Gemini 1.5 Flash                     | Deprecated   |
    | `gemini-1.5-pro`                     | Gemini 1.5 Pro                       | Deprecated   |
    | `gemini-2.0-flash-exp`               | Gemini 2.0 Flash Exp                 | Supported    |
    | `gemini-2.0-flash-lite`              | Gemini 2.0 Flash Lite                | Supported    |
    | `gemini-2.0-flash-thinking-exp-01-21`| Gemini 2.0 Flash Thinking Exp 01-21  | Supported    |
    | `gemini-2.0-flash`                   | Gemini 2.0 Flash                     | Supported    |
    | `gemini-2.0-pro-exp-02-05`           | Gemini 2.0 Pro Exp 02-05             | Supported    |
    | `gemini-2.5-pro-exp-03-25`           | Gemini 2.5 Pro Exp 03-25             | Supported    |
    | `gemini-2.5-pro-preview-03-25`       | Gemini 2.5 Pro Preview 03-25         | Supported    |
    | `gemini-2.5-pro-preview-05-06`       | Gemini 2.5 Pro Preview 05-06         | Supported    |
    | `gemini-3-flash-preview`             | Gemini 3 Flash Preview               | Supported    |
    | `gemini-3-pro-preview`               | Gemini 3 Pro Preview                 | Supported    |
    | `gemini-2.5-pro`                     | Gemini 2.5 Pro                       | Supported    |
    | `gemini-2.5-flash`                   | Gemini 2.5 Flash                     | Supported    |
    | `gemini-2.5-flash-lite`              | Gemini 2.5 Flash Lite                | Supported    |
    | `gemini-2.5-flash-preview-tts`       | Gemini 2.5 Flash Preview TTS         | Supported    |
    | `gemini-2.5-pro-preview-tts`         | Gemini 2.5 Pro Preview TTS           | Supported    |
    | `gemini-3-pro-image-preview`         | Gemini 3 Pro Image Preview           | Supported    |
    | `gemini-2.5-flash-image-preview`     | Gemini 2.5 Flash Image Preview       | Supported    |
    | `gemini-2.5-flash-image`             | Gemini 2.5 Flash Image               | Supported    |
    | `gemma-3-12b-it`                     | Gemma 3 12B IT                       | Supported    |
    | `gemma-3-1b-it`                      | Gemma 3 1B IT                        | Supported    |
    | `gemma-3-27b-it`                     | Gemma 3 27B IT                       | Supported    |
    | `gemma-3-4b-it`                      | Gemma 3 4B IT                        | Supported    |
    | `gemma-3n-e4b-it`                    | Gemma 3n E4B IT                      | Supported    |
    """

    GEMINI_2_0_FLASH = 'gemini-2.0-flash'
    GEMINI_2_0_FLASH_EXP = 'gemini-2.0-flash-exp'
    GEMINI_2_0_FLASH_LITE = 'gemini-2.0-flash-lite'
    GEMINI_2_0_FLASH_THINKING_EXP_01_21 = 'gemini-2.0-flash-thinking-exp-01-21'
    GEMINI_2_0_PRO_EXP_02_05 = 'gemini-2.0-pro-exp-02-05'
    GEMINI_2_5_PRO_EXP_03_25 = 'gemini-2.5-pro-exp-03-25'
    GEMINI_2_5_PRO_PREVIEW_03_25 = 'gemini-2.5-pro-preview-03-25'
    GEMINI_2_5_PRO_PREVIEW_05_06 = 'gemini-2.5-pro-preview-05-06'
    GEMINI_3_FLASH_PREVIEW = 'gemini-3-flash-preview'
    GEMINI_3_PRO_PREVIEW = 'gemini-3-pro-preview'
    GEMINI_2_5_PRO = 'gemini-2.5-pro'
    GEMINI_2_5_FLASH = 'gemini-2.5-flash'
    GEMINI_2_5_FLASH_LITE = 'gemini-2.5-flash-lite'
    GEMINI_2_5_FLASH_PREVIEW_TTS = 'gemini-2.5-flash-preview-tts'
    GEMINI_2_5_PRO_PREVIEW_TTS = 'gemini-2.5-pro-preview-tts'
    GEMINI_3_PRO_IMAGE_PREVIEW = 'gemini-3-pro-image-preview'
    GEMINI_2_5_FLASH_IMAGE_PREVIEW = 'gemini-2.5-flash-image-preview'
    GEMINI_2_5_FLASH_IMAGE = 'gemini-2.5-flash-image'
    GEMMA_3_12B_IT = 'gemma-3-12b-it'
    GEMMA_3_1B_IT = 'gemma-3-1b-it'
    GEMMA_3_27B_IT = 'gemma-3-27b-it'
    GEMMA_3_4B_IT = 'gemma-3-4b-it'
    GEMMA_3N_E4B_IT = 'gemma-3n-e4b-it'


class GoogleAIGeminiVersion(StrEnum, metaclass=Deprecations):
    """GoogleAI Gemini models.

    Model Support:

    | Model                                | Description                          | Status     |
    |--------------------------------------|--------------------------------------|------------|
    | `gemini-1.5-flash-8b`                | Gemini 1.5 Flash 8B                  | Deprecated |
    | `gemini-1.5-flash`                   | Gemini 1.5 Flash                     | Deprecated |
    | `gemini-1.5-pro`                     | Gemini 1.5 Pro                       | Deprecated |
    | `gemini-2.0-flash-exp`               | Gemini 2.0 Flash Exp                 | Supported  |
    | `gemini-2.0-flash-lite`              | Gemini 2.0 Flash Lite                | Supported  |
    | `gemini-2.0-flash-thinking-exp-01-21`| Gemini 2.0 Flash Thinking Exp 01-21  | Supported  |
    | `gemini-2.0-flash`                   | Gemini 2.0 Flash                     | Supported  |
    | `gemini-2.0-pro-exp-02-05`           | Gemini 2.0 Pro Exp 02-05             | Supported  |
    | `gemini-2.5-pro-exp-03-25`           | Gemini 2.5 Pro Exp 03-25             | Supported  |
    | `gemini-2.5-pro-preview-03-25`       | Gemini 2.5 Pro Preview 03-25         | Supported  |
    | `gemini-2.5-pro-preview-05-06`       | Gemini 2.5 Pro Preview 05-06         | Supported  |
    | `gemini-3-flash-preview`             | Gemini 3 Flash Preview               | Supported  |
    | `gemini-3-pro-preview`               | Gemini 3 Pro Preview                 | Supported  |
    | `gemini-2.5-pro`                     | Gemini 2.5 Pro                       | Supported  |
    | `gemini-2.5-flash`                   | Gemini 2.5 Flash                     | Supported  |
    | `gemini-2.5-flash-lite`              | Gemini 2.5 Flash Lite                | Supported  |
    | `gemini-2.5-flash-preview-tts`       | Gemini 2.5 Flash Preview TTS         | Supported  |
    | `gemini-2.5-pro-preview-tts`         | Gemini 2.5 Pro Preview TTS           | Supported  |
    | `gemini-3-pro-image-preview`         | Gemini 3 Pro Image Preview           | Supported  |
    | `gemini-2.5-flash-image-preview`     | Gemini 2.5 Flash Image Preview       | Supported  |
    | `gemini-2.5-flash-image`             | Gemini 2.5 Flash Image               | Supported  |
    | `gemma-3-12b-it`                     | Gemma 3 12B IT                       | Supported  |
    | `gemma-3-1b-it`                      | Gemma 3 1B IT                        | Supported  |
    | `gemma-3-27b-it`                     | Gemma 3 27B IT                       | Supported  |
    | `gemma-3-4b-it`                      | Gemma 3 4B IT                        | Supported  |
    | `gemma-3n-e4b-it`                    | Gemma 3n E4B IT                      | Supported  |
    """

    GEMINI_2_0_FLASH = 'gemini-2.0-flash'
    GEMINI_2_0_FLASH_EXP = 'gemini-2.0-flash-exp'
    GEMINI_2_0_FLASH_LITE = 'gemini-2.0-flash-lite'
    GEMINI_2_0_FLASH_THINKING_EXP_01_21 = 'gemini-2.0-flash-thinking-exp-01-21'
    GEMINI_2_0_PRO_EXP_02_05 = 'gemini-2.0-pro-exp-02-05'
    GEMINI_2_5_PRO_EXP_03_25 = 'gemini-2.5-pro-exp-03-25'
    GEMINI_2_5_PRO_PREVIEW_03_25 = 'gemini-2.5-pro-preview-03-25'
    GEMINI_2_5_PRO_PREVIEW_05_06 = 'gemini-2.5-pro-preview-05-06'
    GEMINI_3_FLASH_PREVIEW = 'gemini-3-flash-preview'
    GEMINI_3_PRO_PREVIEW = 'gemini-3-pro-preview'
    GEMINI_2_5_PRO = 'gemini-2.5-pro'
    GEMINI_2_5_FLASH = 'gemini-2.5-flash'
    GEMINI_2_5_FLASH_LITE = 'gemini-2.5-flash-lite'
    GEMINI_2_5_FLASH_PREVIEW_TTS = 'gemini-2.5-flash-preview-tts'
    GEMINI_2_5_PRO_PREVIEW_TTS = 'gemini-2.5-pro-preview-tts'
    GEMINI_3_PRO_IMAGE_PREVIEW = 'gemini-3-pro-image-preview'
    GEMINI_2_5_FLASH_IMAGE_PREVIEW = 'gemini-2.5-flash-image-preview'
    GEMINI_2_5_FLASH_IMAGE = 'gemini-2.5-flash-image'
    GEMMA_3_12B_IT = 'gemma-3-12b-it'
    GEMMA_3_1B_IT = 'gemma-3-1b-it'
    GEMMA_3_27B_IT = 'gemma-3-27b-it'
    GEMMA_3_4B_IT = 'gemma-3-4b-it'
    GEMMA_3N_E4B_IT = 'gemma-3n-e4b-it'


SUPPORTED_MODELS = {}


DEFAULT_SUPPORTS_MODEL = Supports(
    multiturn=True,
    media=True,
    tools=True,
    tool_choice=True,
    system_role=True,
    constrained=Constrained.NO_TOOLS,
)


def is_gemini_model(name: str) -> bool:
    """Check if the model is a standard Gemini text generation model.

    Excludes TTS and image variants which have different capabilities.

    Args:
        name: The model name to check.

    Returns:
        True if this is a standard Gemini model (not TTS or image).

    Example:
        >>> is_gemini_model('gemini-2.0-flash-001')
        True
        >>> is_gemini_model('gemini-2.5-flash-preview-tts')
        False
    """
    return name.startswith('gemini-') and not is_tts_model(name) and not is_image_model(name)


def is_tts_model(name: str) -> bool:
    """Check if the model is a text-to-speech (TTS) model.

    TTS models output audio instead of text and use GeminiTtsConfigSchema.

    Args:
        name: The model name to check.

    Returns:
        True if this is a TTS model.

    Example:
        >>> is_tts_model('gemini-2.5-flash-preview-tts')
        True
    """
    return (name.startswith('gemini-') and name.endswith('-tts')) or 'tts' in name


def is_image_model(name: str) -> bool:
    """Check if the model is a Gemini image generation model.

    Image models output images instead of text and use GeminiImageConfigSchema.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Gemini image model.

    Example:
        >>> is_image_model('gemini-2.0-flash-preview-image-generation')
        True
    """
    return (name.startswith('gemini-') and '-image' in name) or 'image' in name


def is_gemma_model(name: str) -> bool:
    """Check if the model is a Gemma open model.

    Gemma models are Google's open-weight models with different configuration.

    Args:
        name: The model name to check.

    Returns:
        True if this is a Gemma model.

    Example:
        >>> is_gemma_model('gemma-2-27b-it')
        True
    """
    return name.startswith('gemma-')


def get_model_config_schema(name: str) -> type[GeminiConfigSchema]:
    """Get the appropriate config schema for a dynamically discovered model.

    Different model types (TTS, image, Gemma, standard) have different
    configuration options. This function returns the correct schema based
    on the model name.

    Args:
        name: The model name to determine schema for.

    Returns:
        The appropriate config schema class:
        - GeminiTtsConfigSchema for TTS models
        - GeminiImageConfigSchema for image models
        - GemmaConfigSchema for Gemma models
        - GeminiConfigSchema for standard Gemini models
    """
    if is_tts_model(name):
        return GeminiTtsConfigSchema
    if is_image_model(name):
        return GeminiImageConfigSchema
    if is_gemma_model(name):
        return GemmaConfigSchema
    return GeminiConfigSchema


def google_model_info(
    version: str,
) -> ModelInfo:
    """Generates a ModelInfo object.

    This function returns the best ModelInfo Supports based on model type.
    Detects TTS, Image, Gemma, and standard Gemini models.

    Args:
        version: Version of the model.

    Returns:
        ModelInfo object with appropriate capabilities.
    """
    if version in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[version]

    if is_tts_model(version):
        return GENERIC_TTS_MODEL
    if is_image_model(version):
        return GENERIC_IMAGE_MODEL
    if is_gemma_model(version):
        return GENERIC_GEMMA_MODEL

    return ModelInfo(
        label=f'Google AI - {version}',
        supports=DEFAULT_SUPPORTS_MODEL,
    )


class GeminiModel:
    """Gemini model."""

    def __init__(
        self,
        version: str | GoogleAIGeminiVersion | VertexAIGeminiVersion,
        client: genai.Client,
    ) -> None:
        """Initialize Gemini model.

        Args:
            version: Gemini version
            client: Google AI client
        """
        self._version = version
        self._client = client

    def _get_tools(self, request: GenerateRequest) -> list[genai_types.Tool]:
        """Generates VertexAI Gemini compatible tool definitions.

        Args:
            request: The generation request.

        Returns:
             list of Gemini tools
        """
        tools = []
        for tool in request.tools or []:
            genai_tool = self._create_tool(tool)
            tools.append(genai_tool)

        return tools

    def _create_tool(self, tool: ToolDefinition) -> genai_types.Tool:
        """Create a tool that is compatible with Google Genai API.

        Args:
            tool: Genkit Tool Definition

        Returns:
            Genai tool compatible with Gemini API.
        """
        params = self._convert_schema_property(tool.input_schema)
        # Fix for no-arg tools: parameters cannot be None if we want the tool to be callable?
        # Actually Google GenAI expects type=OBJECT for params usually.
        if not params:
            params = genai_types.Schema(type=genai_types.Type.OBJECT, properties={})

        function = genai_types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=params,
            response=self._convert_schema_property(tool.output_schema) if tool.output_schema else None,
        )
        return genai_types.Tool(function_declarations=[function])

    def _convert_schema_property(
        self, input_schema: dict[str, object] | None, defs: dict[str, object] | None = None
    ) -> genai_types.Schema | None:
        """Sanitizes a schema to be compatible with Gemini API.

        Args:
            input_schema: A dictionary with input parameters
            defs: Dictionary with definitions. Optional.

        Returns:
            Schema or None
        """
        if input_schema is None:
            return None

        if defs is None:
            defs_value = input_schema.get('$defs')
            defs = cast(dict[str, object], defs_value) if isinstance(defs_value, dict) else {}

        if '$ref' in input_schema:
            ref_path = input_schema['$ref']
            if isinstance(ref_path, str):
                ref_tokens = ref_path.split('/')
                ref_name = ref_tokens[-1]

                if defs is None or ref_name not in defs:
                    raise ValueError(f'Failed to resolve schema for {ref_name}')

                ref_schema = defs[ref_name]
                if isinstance(ref_schema, dict):
                    schema = self._convert_schema_property(cast(dict[str, object], ref_schema), defs)
                else:
                    schema = None

                if schema and input_schema.get('description'):
                    schema.description = cast(str, input_schema['description'])

                return schema

        if 'type' not in input_schema:
            return None

        schema = genai_types.Schema()
        if input_schema.get('description'):
            schema.description = cast(str, input_schema['description'])

        if 'required' in input_schema:
            schema.required = cast(list[str], input_schema['required'])

        if 'type' in input_schema:
            raw_type = input_schema['type']
            if isinstance(raw_type, list):
                non_null = [t for t in raw_type if t != 'null']
                schema.nullable = True
                raw_type = non_null[0] if non_null else 'string'
            schema_type = genai_types.Type(cast(str, raw_type))
            schema.type = schema_type

            if 'enum' in input_schema:
                schema.enum = cast(list[str], input_schema['enum'])

            if schema_type == genai_types.Type.ARRAY:
                items_value = input_schema.get('items')
                if isinstance(items_value, dict):
                    schema.items = self._convert_schema_property(cast(dict[str, object], items_value), defs)

            if schema_type == genai_types.Type.OBJECT:
                schema.properties = {}
                properties_value = input_schema.get('properties', {})
                if isinstance(properties_value, dict):
                    properties = cast(dict[str, dict[str, object]], properties_value)
                    for key in properties:
                        nested_schema = self._convert_schema_property(properties[key], defs)
                        if nested_schema:
                            schema.properties[key] = nested_schema

        return schema

    async def _retrieve_cached_content(
        self, request: GenerateRequest, model_name: str, cache_config: dict, contents: list[genai_types.Content]
    ) -> genai_types.CachedContent:
        """Retrieves cached content from the Google API if exists.

        If content is present - increases storage ttl based on the configured `ttl_seconds`
        If content is not present - creates it and returns creates instance.

        Args:
            request: incoming generation instance
            model_name: name of the generation model to use
            cache_config: user-defined cache configuration (e.g. ttl_seconds)
            contents: content to submit for cached context creation

        Returns:
            Cached Content instance based on provided params
        """
        validate_context_cache_request(request=request, model_name=model_name)

        ttl_value = cache_config.get('ttl_seconds', DEFAULT_TTL)
        ttl: float = float(ttl_value) if ttl_value is not None else DEFAULT_TTL
        cache_key = generate_cache_key(request=request)

        iterator_config = genai_types.ListCachedContentsConfig()
        cache = None
        pages = await self._client.aio.caches.list(config=iterator_config)

        async for item in pages:
            if item.display_name == cache_key:
                cache = item
                break
        if cache and cache.name:
            updated_expiration_time = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            cache = await self._client.aio.caches.update(
                name=cache.name, config=genai_types.UpdateCachedContentConfig(expire_time=updated_expiration_time)
            )
        else:
            cache = await self._client.aio.caches.create(
                model=model_name,
                config=genai_types.CreateCachedContentConfig(
                    contents=cast(genai_types.ContentListUnion, contents),
                    display_name=cache_key,
                    ttl=f'{ttl}s',
                ),
            )
        return cache

    async def generate(self, request: GenerateRequest, ctx: ActionRunContext) -> GenerateResponse:
        """Handle a generation request.

        Args:
            request: The generation request containing messages and parameters.
            ctx: action context

        Returns:
            The model's response to the generation request.
        """
        model_name = self._version
        if request.config:
            version = getattr(request.config, 'version', None)
            if version:
                model_name = version

        # TODO(#4361): Do not move - this method mutates `request` by extracting system
        # prompts into configuration object
        request_cfg = await self._genkit_to_googleai_cfg(request=request)

        # TTS models require response_modalities: ["AUDIO"]
        if is_tts_model(model_name):
            if not request_cfg:
                request_cfg = genai_types.GenerateContentConfig()
            request_cfg.response_modalities = ['AUDIO']

        # Image models require response_modalities: ["TEXT", "IMAGE"]
        if is_image_model(model_name):
            if not request_cfg:
                request_cfg = genai_types.GenerateContentConfig()
            request_cfg.response_modalities = ['TEXT', 'IMAGE']

        request_contents, cached_content = await self._build_messages(request=request, model_name=model_name)

        if cached_content and cached_content.name:
            if not request_cfg:
                request_cfg = genai_types.GenerateContentConfig()
            request_cfg.cached_content = cached_content.name

        client = self._client
        # If config specifies api_key, base_url, or api_version different from default,
        # create a temporary client with those settings.
        api_version = None
        api_key_override = None
        base_url_override = None

        if request.config:
            if isinstance(request.config, dict):
                api_version = request.config.get('api_version') or request.config.get('apiVersion')
                api_key_override = request.config.get('api_key') or request.config.get('apiKey')
                base_url_override = request.config.get('base_url') or request.config.get('baseUrl')
            else:
                api_version = getattr(request.config, 'api_version', None)
                api_key_override = getattr(request.config, 'api_key', None)
                base_url_override = getattr(request.config, 'base_url', None)

        if api_version or api_key_override or base_url_override:
            # TODO(#4362): Request public API from google-genai maintainers.
            # Currently, there is no public way to access the configured api_key, project, or location
            # from an existing Client instance. We need to access the private _api_client to
            # clone the configuration when overriding these settings.
            # This is brittle and relies on internal implementation details of the google-genai library.
            # If the library changes its internal structure (e.g. renames _api_client or _credentials),
            # this code WILL BREAK.
            api_client = self._client._api_client
            kwargs: dict[str, Any] = {
                'vertexai': api_client.vertexai,
            }

            # Set http_options if we have api_version or base_url
            http_options = {}
            if api_version:
                http_options['api_version'] = api_version
            if base_url_override:
                http_options['base_url'] = base_url_override
            if http_options:
                kwargs['http_options'] = http_options

            if api_client.vertexai:
                # Vertex AI mode: requires project/location (api_key is optional/unlikely)
                if api_client.project:
                    kwargs['project'] = api_client.project
                if api_client.location:
                    kwargs['location'] = api_client.location
                if api_client._credentials:
                    kwargs['credentials'] = api_client._credentials
                # Don't pass api_key if we are in Vertex AI mode with credentials/project
            else:
                # Google AI mode: primarily uses api_key
                # Use override if provided, otherwise fall back to existing client's api_key
                if api_key_override:
                    kwargs['api_key'] = api_key_override
                elif api_client.api_key:
                    kwargs['api_key'] = api_client.api_key
                # Do NOT pass project/location/credentials if in Google AI mode to be safe
                if api_client._credentials and not kwargs.get('api_key'):
                    # Fallback if no api_key but credentials present (unlikely for pure Google AI but possible)
                    kwargs['credentials'] = api_client._credentials

            try:
                client = genai.Client(**kwargs)
            except Exception as e:
                # If client creation fails (e.g., invalid API key format), raise a clear error
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'Failed to create Google AI client: {str(e)}',
                ) from e

        if ctx.is_streaming:
            response = await self._streaming_generate(
                request_contents=request_contents,
                request_cfg=request_cfg,
                ctx=ctx,
                model_name=model_name,
                client=client,
            )
        else:
            response = await self._generate(
                request_contents=request_contents, request_cfg=request_cfg, model_name=model_name, client=client
            )

        response.usage = self._create_usage_stats(request=request, response=response)

        return response

    async def _generate(
        self,
        request_contents: list[genai_types.Content],
        request_cfg: genai_types.GenerateContentConfig | None,
        model_name: str,
        client: genai.Client | None = None,
    ) -> GenerateResponse:
        """Call google-genai generate.

        Args:
            request_contents: request contents
            request_cfg: request configuration
            model_name: name of generation model to use
            client: optional client to use for the request

        Returns:
            genai response.
        """
        with tracer.start_as_current_span('generate_content') as span:
            span.set_attribute(
                'genkit:input',
                dump_json(
                    {
                        'config': dump_dict(request_cfg),
                        'contents': [dump_dict(c) for c in request_contents],
                        'model': model_name,
                    },
                    fallback=lambda _: '[!! failed to serialize !!]',
                ),
            )
            client = client or self._client
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=cast(genai_types.ContentListUnion, request_contents),
                    config=request_cfg,
                )
            except ClientError as e:
                status: StatusName = 'INTERNAL'
                if e.code == 400:
                    status = 'INVALID_ARGUMENT'
                elif e.code == 401:
                    status = 'UNAUTHENTICATED'
                elif e.code == 403:
                    status = 'PERMISSION_DENIED'
                elif e.code == 404:
                    status = 'NOT_FOUND'
                elif e.code == 429:
                    status = 'RESOURCE_EXHAUSTED'

                raise GenkitError(
                    status=status,
                    message=e.message or 'Unknown error',
                    cause=e,
                ) from e
            except Exception as e:
                # Catch any other exceptions and provide a clear error message
                # This helps debug issues like authentication errors that might not be ClientError
                import logging

                logger = logging.getLogger(__name__)
                logger.error(f'Unexpected error during generate_content: {type(e).__name__}: {str(e)}')
                raise GenkitError(
                    status='INTERNAL',
                    message=f'Unexpected error during generation: {type(e).__name__}: {str(e)}',
                ) from e
            span.set_attribute('genkit:output', dump_json(response))

        content = await self._contents_from_response(response)

        # Ensure we always have at least one content item to avoid UI errors
        if not content:
            content = [Part(root=TextPart(text=''))]

        finish_reason = FinishReason.OTHER
        candidates = []
        if response.candidates:
            for i, c in enumerate(response.candidates):
                c_content = []
                if c.content and c.content.parts:
                    for j, part in enumerate(c.content.parts):
                        converted = PartConverter.from_gemini(part=part, ref=str(j))
                        if converted:
                            c_content.append(converted)

                if not c_content:
                    c_content = [Part(root=TextPart(text=''))]

                c_finish_reason = FinishReason.OTHER
                if c.finish_reason:
                    fr_name = c.finish_reason.name
                    if fr_name == 'STOP':
                        c_finish_reason = FinishReason.STOP
                    elif fr_name == 'MAX_TOKENS':
                        c_finish_reason = FinishReason.LENGTH
                    elif fr_name in ['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII']:
                        c_finish_reason = FinishReason.BLOCKED
                    elif fr_name == 'OTHER':
                        c_finish_reason = FinishReason.OTHER

                if i == 0:
                    finish_reason = c_finish_reason

                candidates.append(
                    Candidate(
                        index=float(i),
                        message=Message(role=Role.MODEL, content=c_content),
                        finish_reason=c_finish_reason,
                    )
                )

        return GenerateResponse(
            message=Message(
                content=content,
                role=Role.MODEL,
            ),
            finish_reason=finish_reason,
            candidates=candidates,
            usage=GenerationUsage(
                input_tokens=float(response.usage_metadata.prompt_token_count or 0)
                if response.usage_metadata
                else None,
                output_tokens=float(response.usage_metadata.candidates_token_count or 0)
                if response.usage_metadata
                else None,
                total_tokens=float(response.usage_metadata.total_token_count or 0) if response.usage_metadata else None,
            ),
        )

    async def _streaming_generate(
        self,
        request_contents: list[genai_types.Content],
        request_cfg: genai_types.GenerateContentConfig | None,
        ctx: ActionRunContext,
        model_name: str,
        client: genai.Client | None = None,
    ) -> GenerateResponse:
        """Call google-genai generate for streaming.

        Args:
            request_contents: request contents
            request_cfg: request configuration
            ctx: action context
            model_name: name of generation model to use
            client: optional client to use for the request

        Returns:
            empty genai response
        """
        with tracer.start_as_current_span('generate_content_stream') as span:
            span.set_attribute(
                'genkit:input',
                dump_json({
                    'config': dump_dict(request_cfg),
                    'contents': [dump_dict(c) for c in request_contents],
                    'model': model_name,
                }),
            )
            client = client or self._client
            try:
                generator = await client.aio.models.generate_content_stream(
                    model=model_name,
                    contents=cast(genai_types.ContentListUnion, request_contents),
                    config=request_cfg,
                )
            except ClientError as e:
                status: StatusName = 'INTERNAL'
                if e.code == 400:
                    status = 'INVALID_ARGUMENT'
                elif e.code == 401:
                    status = 'UNAUTHENTICATED'
                elif e.code == 403:
                    status = 'PERMISSION_DENIED'
                elif e.code == 404:
                    status = 'NOT_FOUND'
                elif e.code == 429:
                    status = 'RESOURCE_EXHAUSTED'

                raise GenkitError(
                    status=status,
                    message=e.message or 'Unknown error',
                    cause=e,
                ) from e
        accumulated_content = []
        async for response_chunk in generator:
            content = await self._contents_from_response(response_chunk)
            if content:  # Only process if we have content
                accumulated_content.extend(content)
                ctx.send_chunk(
                    chunk=GenerateResponseChunk(
                        content=content,
                        role=Role.MODEL,
                    )
                )

        return GenerateResponse(
            message=Message(
                role=Role.MODEL,
                content=accumulated_content,
            )
        )

    @cached_property
    def metadata(self) -> dict:
        """Get model metadata.

        Returns:
            model metadata.
        """
        if self._version in SUPPORTED_MODELS:
            supports = SUPPORTED_MODELS[self._version].supports.model_dump(by_alias=True, exclude_none=True)
        else:
            # Fallback to default supports for models not explicitly listed
            supports = DEFAULT_SUPPORTS_MODEL.model_dump(by_alias=True, exclude_none=True)
        return {
            'model': {
                'label': f'Google AI - {self._version}',
                'supports': supports,
            }
        }

    async def _build_messages(
        self, request: GenerateRequest, model_name: str
    ) -> tuple[list[genai_types.Content], genai_types.CachedContent | None]:
        """Build google-genai request contents from Genkit request.

        Args:
            request: Genkit request.
            model_name: name of generation model to use

        Returns:
            list of google-genai contents.
        """
        request_contents: list[genai_types.Content] = []
        cache = None

        for msg in request.messages:
            if msg.role == Role.SYSTEM:
                continue
            content_parts: list[genai_types.Part] = []
            for p in msg.content:
                converted = await PartConverter.to_gemini(p)
                if isinstance(converted, list):
                    content_parts.extend(converted)
                else:
                    content_parts.append(converted)
            request_contents.append(genai_types.Content(parts=content_parts, role=msg.role))

            if msg.metadata and msg.metadata.get('cache'):
                cache = await self._retrieve_cached_content(
                    request=request,
                    model_name=model_name,
                    cache_config=msg.metadata['cache'],
                    contents=request_contents,
                )

        if not request_contents:
            request_contents.append(genai_types.Content(parts=[genai_types.Part(text=' ')], role='user'))

        return request_contents, cache

    async def _contents_from_response(self, response: genai_types.GenerateContentResponse) -> list:
        """Retrieve contents from google-genai response.

        Args:
            response: google-genai response.

        Returns:
            list of generated contents.
        """
        content = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        converted = PartConverter.from_gemini(part=part, ref=str(i))
                        if converted:  # Only append if conversion succeeded
                            content.append(converted)

        # Ensure we always return a list, even if empty
        return content if content else []

    async def _genkit_to_googleai_cfg(self, request: GenerateRequest) -> genai_types.GenerateContentConfig | None:
        """Converts a Genkit GenerateRequest to a Gemini GenerateContentConfig.

        The conversion follows a linear pipeline:
        1. Extract system instructions from messages
        2. Normalize request.config into a dict (regardless of input type)
        3. Extract tool-related fields from the dict
        4. Clean Genkit-specific / unsupported keys from the dict
        5. Build the final GenerateContentConfig
        """
        system_instruction: list[genai.types.Part] = []

        # 1. System messages
        system_messages = list(filter(lambda m: m.role == Role.SYSTEM, request.messages))
        for m in system_messages:
            if m.content:
                for p in m.content:
                    converted = await PartConverter.to_gemini(p)
                    if isinstance(converted, list):
                        system_instruction.extend(converted)
                    else:
                        system_instruction.append(converted)

        cfg = None
        tools: list[genai_types.Tool] = []

        if request.config:
            # 2. Normalize config into a dict
            dumped_config = self._normalize_config_to_dict(request.config)

            if dumped_config is not None:
                # 3. Extract tool-related fields
                self._extract_tools_from_config(dumped_config, tools)

                # 4. Clean Genkit-specific and unsupported keys
                self._clean_unsupported_keys(dumped_config)

                # 5. Build GenerateContentConfig
                if dumped_config:
                    cfg = genai_types.GenerateContentConfig(**dumped_config)
                else:
                    cfg = None

        # Tools from top-level field and config-level fields
        tools.extend(self._get_tools(request))

        if cfg is not None or tools or system_instruction or request.output:
            if cfg is None:
                cfg = genai_types.GenerateContentConfig()

            if request.output:
                response_mime_type = (
                    'application/json' if request.output.format == 'json' and not request.tools else None
                )
                cfg.response_mime_type = response_mime_type

                if request.output.schema and request.output.constrained:
                    cfg.response_schema = self._convert_schema_property(request.output.schema)

            if tools:
                cfg.tools = cast(genai_types.ToolListUnion, tools)

            cfg.system_instruction = genai_types.Content(parts=system_instruction) if system_instruction else None
            return cfg

        return None

    # -- Config conversion helpers (called by _genkit_to_googleai_cfg) --

    # Keys that are Genkit-specific and must not be forwarded to the API.
    # 'version' overrides the model name, others are client-level settings.
    _GENKIT_ONLY_KEYS = frozenset(['version', 'api_version', 'api_key', 'base_url', 'context_cache'])

    # Keys that may not be supported by older google-genai SDK versions.
    _SDK_GATED_KEYS = frozenset(['image_config', 'thinking_config', 'response_modalities'])

    def _normalize_config_to_dict(
        self,
        config: GeminiConfigSchema | GenerationCommonConfig | dict,
    ) -> dict[str, Any] | None:
        """Normalize any config type into a plain dict for uniform processing.

        Handles three input shapes:
        - GeminiConfigSchema (and subclasses like TTS/Image): model_dump
        - GenerationCommonConfig: model_dump
        - dict: route to the appropriate schema first, then model_dump

        Returns:
            A mutable dict ready for tool extraction and key cleanup,
            or None if the config is empty after dumping.
        """
        if isinstance(config, GeminiConfigSchema):
            schema = config
        elif isinstance(config, GenerationCommonConfig):
            schema = config
        elif isinstance(config, dict):
            if 'image_config' in config:
                schema = GeminiImageConfigSchema(**config)
            elif 'speech_config' in config:
                schema = GeminiTtsConfigSchema(**config)
            else:
                schema = GeminiConfigSchema(**config)
        else:
            return None

        dumped = schema.model_dump(exclude_none=True)
        return dumped or None

    def _extract_tools_from_config(
        self,
        config: dict[str, Any],
        tools: list[genai_types.Tool],
    ) -> None:
        """Extract tool-related fields from config dict into the tools list.

        Mutates *config* by popping consumed keys and appends to *tools*.
        """
        # Code execution
        if config.pop('code_execution', None):
            tools.append(genai_types.Tool(code_execution=genai_types.ToolCodeExecution()))

        # Safety settings — filter out unspecified categories
        if 'safety_settings' in config:
            config['safety_settings'] = [
                s for s in config['safety_settings'] if s['category'] != HarmCategory.HARM_CATEGORY_UNSPECIFIED
            ]

        # Google Search
        val = config.pop('google_search_retrieval', None)
        if val is not None:
            val = {} if val is True else val
            tools.append(genai_types.Tool(google_search=genai_types.GoogleSearch(**val)))

        # File Search
        val = config.pop('file_search', None)
        if val and val.get('file_search_store_names'):
            valid_stores = [s for s in val['file_search_store_names'] if s]
            if valid_stores:
                val['file_search_store_names'] = valid_stores
                tools.append(genai_types.Tool(file_search=genai_types.FileSearch(**val)))

        # URL Context
        val = config.pop('url_context', None)
        if val is not None:
            val = {} if val is True else val
            tools.append(genai_types.Tool(url_context=genai_types.UrlContext(**val)))

        # Function Calling Config → ToolConfig
        fcc = config.pop('function_calling_config', None)
        if fcc:
            config['tool_config'] = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(**fcc)
            )

    def _clean_unsupported_keys(self, config: dict[str, Any]) -> None:
        """Remove Genkit-specific and SDK-gated keys from the config dict.

        Mutates *config* in place.
        """
        for key in self._GENKIT_ONLY_KEYS:
            config.pop(key, None)

        for key in self._SDK_GATED_KEYS:
            if key in config and key not in genai_types.GenerateContentConfig.model_fields:
                del config[key]

    def _create_usage_stats(self, request: GenerateRequest, response: GenerateResponse) -> GenerationUsage:
        """Create usage statistics.

        Args:
            request: Genkit request
            response: Genkit response

        Returns:
            usage statistics
        """
        if not response.message:
            usage = GenerationUsage()
            usage.input_tokens = 0
            usage.output_tokens = 0
            usage.total_tokens = 0
            return usage

        usage = get_basic_usage_stats(input_=request.messages, response=response.message)
        if response.usage:
            usage.input_tokens = response.usage.input_tokens
            usage.output_tokens = response.usage.output_tokens
            usage.total_tokens = response.usage.total_tokens

        return usage
