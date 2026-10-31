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

from genkit._ai._background_model import BackgroundModel
from genkit._ai._generate import StreamingCallbackError
from genkit._ai._model import (
    Model,
)
from genkit._core._action import ModelContext
from genkit._core._model import (
    GenerateActionOptions,
    GenerationUsage,
    ModelConfig,
    ModelRef,
    ModelRequest,
    OutputConfig,
)
from genkit._core._typing import (
    ConstrainedSupport,
    ModelInfo,
    ModelStage,
    ModelSupports,
    OperationError,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)

__all__ = [
    # Errors
    'StreamingCallbackError',
    # Models, request and context types
    'Model',
    'BackgroundModel',
    'GenerateActionOptions',
    'ModelContext',
    'ModelRequest',
    'OutputConfig',
    # Usage and metadata
    'GenerationUsage',
    # Long-running operations
    'OperationError',
    # Tool types
    'ToolRequest',
    'ToolDefinition',
    'ToolResponse',
    # Model info
    'ModelInfo',
    'ModelSupports',
    'ConstrainedSupport',
    'ModelStage',
    # Reference types
    'ModelRef',
    # Config
    'ModelConfig',
]
