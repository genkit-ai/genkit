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

"""Model protocol types for plugin authors."""

from genkit._ai._model import (
    ModelConfig,
    model_action_metadata,
    model_ref,
)
from genkit._core._background import BackgroundAction
from genkit._core._model import (
    GenerateActionOptions,
    Message,
    ModelRef,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    get_basic_usage_stats,
)
from genkit._core._typing import (
    Candidate,
    Constrained,
    Error,
    FinishReason,
    ModelInfo,
    Operation,
    Stage,
    Supports,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)

__all__ = [
    # Request/Response types
    'BackgroundAction',
    'ModelRequest',
    'ModelResponse',
    'ModelResponseChunk',
    # Usage and metadata
    'ModelUsage',
    'Candidate',
    'FinishReason',
    'GenerateActionOptions',
    # Error and operation
    'Error',
    'Operation',
    # Tool types
    'ToolRequest',
    'ToolDefinition',
    'ToolResponse',
    # Model info
    'ModelInfo',
    'Supports',
    'Constrained',
    'Stage',
    # Factory functions and metadata
    'model_action_metadata',
    'model_ref',
    # Reference types
    'ModelRef',
    # Config
    'ModelConfig',
    # Message
    'Message',
    # Usage
    'get_basic_usage_stats',
]
