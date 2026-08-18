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

"""Configuration types for the Amazon Bedrock plugin."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

from genkit import ModelConfig

DEFAULT_MAX_RETRIES = 3
# Socket read timeout: the gap botocore tolerates between two reads, which
# resets on every byte received. Generous because Bedrock generations can
# legitimately run for many minutes (Nova allows 60-minute inference).
DEFAULT_READ_TIMEOUT = 3600.0
DEFAULT_CONNECT_TIMEOUT = 60.0
# Whole-call deadline. Unlike the read timeout this does not reset, so a
# connection that dribbles a byte at a time still ends.
DEFAULT_TOTAL_TIMEOUT = 3600.0
# The botocore default of 10 pooled connections throttles LLM concurrency.
DEFAULT_MAX_POOL_CONNECTIONS = 50


class BedrockConfig(ModelConfig):
    """Per-call configuration for Bedrock models.

    Unknown keys are rejected: only the declared fields reach the Converse
    API, so a tolerated typo (``maxTokens`` for ``maxOutputTokens``) would run
    the call with the knob silently unset. Model-specific options travel
    through ``additional_model_request_fields``.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra='forbid',
    )

    tool_choice: str | None = None
    """Tool choice mode: ``auto``, ``required``/``any``, ``none``, or a tool name."""

    additional_model_request_fields: dict[str, Any] | None = None
    """Forwarded verbatim to the Converse API (e.g. Claude extended thinking)."""


class BedrockImageConfig(BaseModel):
    """Per-call configuration for Bedrock image models.

    Deliberately declares no properties and rejects nothing. Image config
    shapes vary per model family (Titan Image and Nova Canvas nest their
    options under ``imageGenerationConfig``; the Stability models take flat
    fields like ``aspect_ratio``, ``output_format``, and ``seed``), so any
    strict schema would silently reject every family-specific override at
    Genkit's request validation.
    """

    model_config = ConfigDict(extra='allow')


class ModelDefinition(BaseModel):
    """A Bedrock model to register with Genkit.

    Capabilities are inferred from the built-in registry when not provided;
    unknown chat models default to multimodal + tools at the unstable stage.
    """

    name: str
    """Bedrock model ID, e.g. ``anthropic.claude-sonnet-4-5-20250929-v1:0``."""

    type: Literal['chat', 'text', 'image'] = 'chat'
    """Routes generate calls: chat/text via Converse, image via InvokeModel.
    Embedders are configured separately, through ``Bedrock(embedders=...)``."""
