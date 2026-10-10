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

"""Model protocol types for plugin authors; application code should call :class:`genkit.Genkit` ``generate``."""

from genkit._ai._generate import StreamingCallbackError
from genkit._ai._model import (
    model,
    model_action_metadata,
    model_ref,
)
from genkit._core._background import BackgroundAction, background_model
from genkit._core._model import (
    ABNORMAL_FINISH_REASONS,
    Candidate,
    GenerateActionOptions,
    ModelConfig,
    ModelRef,
    ModelRequest,
    ModelUsage,
    OutputConfig,
    get_basic_usage_stats,
)
from genkit._core._typing import (
    Constrained,
    ModelInfo,
    OperationError,
    Stage,
    Supports,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)

__all__ = [
    # Finish reasons
    'ABNORMAL_FINISH_REASONS',
    # Errors
    'StreamingCallbackError',
    # Request types
    'BackgroundAction',
    'GenerateActionOptions',
    'ModelRequest',
    'OutputConfig',
    # Usage and metadata
    'ModelUsage',
    'Candidate',
    # Long-running operations
    'OperationError',
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
    'model',
    'background_model',
    'model_action_metadata',
    'model_ref',
    'get_basic_usage_stats',
    # Reference types
    'ModelRef',
    # Config
    'ModelConfig',
]
