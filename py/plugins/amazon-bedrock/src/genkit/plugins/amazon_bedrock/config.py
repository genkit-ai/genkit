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

DEFAULT_REGION = 'us-east-1'
DEFAULT_MAX_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 30.0


class BedrockConfig(ModelConfig):
    """Per-call configuration for Bedrock models.

    Mirrors the Go plugin's ``Config`` surface. Unknown keys are allowed and
    forwarded so callers can use provider-specific options without a plugin
    release.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra='allow',
    )

    tool_choice: str | None = None
    """Tool choice mode: ``auto``, ``required``/``any``, ``none``, or a tool name."""

    additional_model_request_fields: dict[str, Any] | None = None
    """Forwarded verbatim to the Converse API (e.g. Claude extended thinking)."""


class ModelDefinition(BaseModel):
    """A Bedrock model to register with Genkit.

    Capabilities are inferred from the built-in registry when not provided;
    unknown chat models default to multimodal + tools at the unstable stage.
    """

    name: str
    """Bedrock model ID, e.g. ``anthropic.claude-sonnet-4-5-20250929-v1:0``."""

    type: Literal['chat', 'image', 'embedding'] = 'chat'
    """Routes generate calls: chat/text via Converse, image via InvokeModel."""
