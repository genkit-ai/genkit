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

"""OpenAI configuration for Genkit.

This module defines configuration schemas that align with the OpenAI Chat
Completions API.

See Also:
    - OpenAI API Reference: https://platform.openai.com/docs/api-reference/chat/create
    - OpenAI Python SDK: https://github.com/openai/openai-python
    - Text Generation Guide: https://platform.openai.com/docs/guides/text-generation
    - Reasoning Models Guide: https://platform.openai.com/docs/guides/reasoning
    - Structured Outputs Guide: https://platform.openai.com/docs/guides/structured-outputs
    - Function Calling Guide: https://platform.openai.com/docs/guides/function-calling
    - Audio Guide: https://platform.openai.com/docs/guides/audio
    - Prompt Caching Guide: https://platform.openai.com/docs/guides/prompt-caching
"""

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from genkit import GenkitError
from genkit.model import ModelConfig
from genkit.plugin_api import StrEnum


class ReasoningEffort(StrEnum):
    """Reasoning effort level for reasoning models (o1, o3, o4 series).

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


class ServiceTier(StrEnum):
    """Service tier for request processing.

    Controls the processing type used for serving the request.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier
    """

    AUTO = 'auto'
    DEFAULT = 'default'
    FLEX = 'flex'
    SCALE = 'scale'
    PRIORITY = 'priority'


class PromptCacheRetention(StrEnum):
    """Prompt cache retention policy.

    Controls how long cached prefixes are kept active.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-prompt_cache_retention
    """

    IN_MEMORY = 'in-memory'
    HOURS_24 = '24h'


class WebSearchContextSize(StrEnum):
    """Web search context size.

    Controls the amount of context window space to use for search results.

    See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-web_search_options
    """

    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


# OpenAI's names for settings ModelConfig already has, keyed to the common
# field. The plugin sends max_output_tokens as max_tokens or
# max_completion_tokens (per model) and stop_sequences as stop, so these are
# not declared and ``extra`` can't set them either.
NATIVE_NAMES: dict[str, str] = {
    'max_tokens': 'max_output_tokens',
    'max_completion_tokens': 'max_output_tokens',
    'stop': 'stop_sequences',
}
REMOVED_KEYS: dict[str, str] = {
    **NATIVE_NAMES,
    'maxTokens': 'max_output_tokens',
    'maxCompletionTokens': 'max_output_tokens',
}


class OpenAIConfig(ModelConfig):
    """OpenAI configuration for Genkit.

    This schema provides full control over OpenAI Chat Completions API parameters.

    Official Documentation:
        - API Reference: https://platform.openai.com/docs/api-reference/chat/create
        - Python SDK Types: https://github.com/openai/openai-python/blob/main/src/openai/types/chat/completion_create_params.py

    Attributes:
        model: Model ID override (e.g., 'gpt-4o', 'o3').
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-model

        temperature: Sampling temperature (0.0 to 2.0). Higher = more random.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature

        top_p: Nucleus sampling probability (0.0 to 1.0).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p

        stream: Whether to stream the response.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream

        n: Number of completions to generate.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-n

        frequency_penalty: Penalize tokens by frequency (-2.0 to 2.0).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty

        presence_penalty: Penalize tokens by presence (-2.0 to 2.0).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-presence_penalty

        logit_bias: Modify likelihood of specific tokens (-100 to 100).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias

        logprobs: Whether to return log probabilities.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs

        top_logprobs: Number of top log probabilities to return (0-20).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs

        seed: Random seed for deterministic sampling (beta).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-seed

        safety_identifier: Stable identifier for detecting policy violations.
            OpenAI's ``user`` field is not declared; send it with
            ``extra={'user': ...}`` for a server that reads it.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-safety_identifier

        prompt_cache_key: Identifier for caching optimization.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-prompt_cache_key

        prompt_cache_retention: Cache retention policy ('in-memory' or '24h').
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-prompt_cache_retention

        reasoning_effort: Reasoning effort for o-series models.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort

        verbosity: Response verbosity level (low, medium, high).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-verbosity

        parallel_tool_calls: Enable parallel function calling.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-parallel_tool_calls

        response_format: Output format (text, json_object, json_schema).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format

        modalities: Output modalities (['text'] or ['text', 'audio']).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-modalities

        audio: Audio output parameters.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-audio

        service_tier: Processing tier (auto, default, flex, priority).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier

        store: Store completion for distillation/evals.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-store

        metadata: Key-value pairs for the object (up to 16).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-metadata

        prediction: Predicted output content for regeneration.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-prediction

        stream_options: Streaming options (e.g., include_usage).
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options

        web_search_options: Web search tool configuration.
            See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-web_search_options

        max_output_tokens: Genkit's cross-model reply cap (``maxOutputTokens``
            in the Dev UI). Sent as ``max_tokens``, or ``max_completion_tokens``
            on reasoning models and when ``reasoning_effort`` is set.

        stop_sequences: Genkit's cross-model stop strings. Sent as ``stop``.

        extra: Request body fields this class doesn't declare, such as a
            field a newer API version or an OpenAI-compatible server added.
            Sent as the OpenAI SDK's ``extra_body``, which replaces a
            top-level field of the same name, so ``extra={'temperature': 0.9}``
            wins over ``temperature``. Nothing inside it is checked.

    Any other key raises, so a typo like ``temprature`` fails by name instead
    of reaching OpenAI. OpenAI's own names for the reply cap and stop strings
    (``max_tokens``, ``max_completion_tokens``, ``stop``) raise naming the
    common field, at the top level or in ``extra``.
    """

    # Dev UI and reflection send camelCase. frequencyPenalty binds and goes
    # out as frequency_penalty; maxOutputTokens goes out as the token cap.
    # Both spellings and extra='forbid' come from GenkitConfig; a create()
    # param this class doesn't declare goes in ``extra`` and is sent as
    # ``extra_body``.

    @model_validator(mode='before')
    @classmethod
    def _name_common_field_for_native_key(cls, data: object) -> object:
        """Name the common field instead of the generic unknown-key error."""
        if not isinstance(data, Mapping):
            return data
        for key in cast(Mapping[str, object], data):
            if key in REMOVED_KEYS:
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'openai: {key} is not a config field; use {REMOVED_KEYS[key]}',
                )
        return data

    # Core generation parameters
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-model
    model: str | None = None

    # temperature and top_p come from ModelConfig; OpenAI's ranges are
    # applied without re-declaring them.
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_p
    common_field_bounds = {'temperature': (0.0, 2.0), 'top_p': (0.0, 1.0)}

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream
    stream: bool | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-n
    n: int | None = Field(default=None, ge=1)

    # Penalty parameters
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-presence_penalty
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    # Token probability parameters
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-logit_bias
    logit_bias: dict[str, int] | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
    logprobs: bool | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-top_logprobs
    top_logprobs: int | None = Field(default=None, ge=0, le=20)

    # Determinism (beta feature)
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-seed
    seed: int | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-safety_identifier
    safety_identifier: str | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-prompt_cache_key
    prompt_cache_key: str | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-prompt_cache_retention
    prompt_cache_retention: PromptCacheRetention | None = None

    # Reasoning models (o1, o3, o4 series)
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-reasoning_effort
    # https://platform.openai.com/docs/guides/reasoning
    reasoning_effort: ReasoningEffort | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-verbosity
    verbosity: Verbosity | None = None

    # Tool calling
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-parallel_tool_calls
    # https://platform.openai.com/docs/guides/function-calling#configuring-parallel-function-calling
    parallel_tool_calls: bool | None = None

    # Output format control
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-response_format
    # https://platform.openai.com/docs/guides/structured-outputs
    response_format: dict[str, Any] | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-modalities
    modalities: list[Literal['text', 'audio']] | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-audio
    # https://platform.openai.com/docs/guides/audio
    audio: dict[str, Any] | None = None

    # Service configuration
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-service_tier
    # https://platform.openai.com/docs/guides/flex-processing
    service_tier: ServiceTier | None = None

    # Storage and metadata
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-store
    # https://platform.openai.com/docs/guides/distillation
    store: bool | None = None

    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-metadata
    metadata: dict[str, str] | None = None

    # Optimization
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-prediction
    # https://platform.openai.com/docs/guides/predicted-outputs
    prediction: dict[str, Any] | None = None

    # Streaming options
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options
    stream_options: dict[str, Any] | None = None

    # Web search
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-web_search_options
    # https://platform.openai.com/docs/guides/tools-web-search
    web_search_options: dict[str, Any] | None = None
