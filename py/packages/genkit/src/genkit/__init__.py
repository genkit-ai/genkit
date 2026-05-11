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

"""Genkit — Build AI-powered applications."""

from genkit._ai._aio import ActionKind, ActionRunContext, Genkit
from genkit._ai._prompt import (
    ExecutablePrompt,
    ModelStreamResponse,
    PromptGenerateOptions,
)
from genkit._ai._tools import (
    Interrupt,
    Tool,
    ToolRunContext,
    respond_to_interrupt,
    restart_tool,
)
from genkit._core._action import Action, StreamResponse
from genkit._core._error import GenkitError, PublicError
from genkit._core._model import Document
from genkit._core._plugin import Plugin
from genkit._core._plugins import extend_plugin_namespace
from genkit._core._typing import (
    CustomPart,
    DocumentPart,
    Media,
    MediaPart,
    Metadata,
    Part,
    ReasoningPart,
    Role,
    TextPart,
    ToolChoice,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)

# Import embedder-related types from the embedder namespace
from genkit.embedder import (
    EmbedderOptions,
    EmbedderRef,
    Embedding,
    EmbedRequest,
    EmbedResponse,
)

# Import model-related types from the model namespace.
from genkit.model import (
    Constrained,
    FinishReason,
    Message,
    ModelConfig,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    Stage,
    Supports,
    ToolDefinition,
)

# Flow is an alias for Action (used in samples for flow type hints)
Flow = Action

extend_plugin_namespace()

__all__ = [
    # Main class
    'Genkit',
    'Flow',
    # Response types
    'Action',
    'StreamResponse',
    'EmbedRequest',
    'EmbedResponse',
    'EmbedderOptions',
    'EmbedderRef',
    'ModelConfig',
    'ModelInfo',
    'ModelStreamResponse',
    # Errors
    'GenkitError',
    'PublicError',
    'Interrupt',
    'Tool',
    'respond_to_interrupt',
    'restart_tool',
    # Content types
    'Constrained',
    'CustomPart',
    'Embedding',
    'Metadata',
    'ReasoningPart',
    'FinishReason',
    'ModelUsage',
    'Media',
    'MediaPart',
    'Message',
    'Part',
    'Role',
    'Stage',
    'Supports',
    'TextPart',
    'ToolChoice',
    'ToolDefinition',
    'ToolRequest',
    'ToolRequestPart',
    'ToolResponse',
    'ToolResponsePart',
    # Domain types
    'Document',
    'DocumentPart',
    # Plugin interface
    'Plugin',
    # AI runtime
    'ActionKind',
    'ActionRunContext',
    'ExecutablePrompt',
    'PromptGenerateOptions',
    'ToolRunContext',
    'ModelRequest',
    'ModelResponse',
    'ModelResponseChunk',
]
