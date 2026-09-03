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

"""Genkit — production-ready SDK for AI-powered applications.

Build AI agents with structured generation, tool calling, streaming, and
observability. Register plugins, define flows and tools, and run generation.

Example:
    from genkit import Genkit
    from genkit_google_genai import GoogleAI

    ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

    @ai.tool()
    async def current_weather(city: str) -> str:
        return f'Sunny in {city}'

    @ai.flow()
    async def my_flow(prompt: str) -> str:
        res = await ai.generate(prompt=prompt, tools=['current_weather'])
        return res.text

    if __name__ == '__main__':
        ai.run_main(my_flow('Weather in Paris?'))
"""

from genkit._ai._aio import ActionKind, Genkit
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
    tool,
)
from genkit._core._action import Action, ActionRunContext, StreamResponse
from genkit._core._error import ErrorResponseMetadata, GenkitError, PublicError
from genkit._core._model import Document
from genkit._core._plugin import Plugin
from genkit._core._typing import (
    CustomPart,
    DocumentPart,
    Media,
    MediaPart,
    Metadata,
    MiddlewareRef,
    MultipartToolResponse,
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
    EmbedderInfo,
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
    ModelConfigDict,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
    Stage,
    Supports,
    ToolDefinition,
    background_model,
)

# Flow is an alias for Action (used in samples for flow type hints)
Flow = Action

__all__ = [
    # Main class
    'Genkit',
    'Flow',
    # Response types
    'Action',
    'StreamResponse',
    'EmbedRequest',
    'EmbedResponse',
    'EmbedderInfo',
    'EmbedderRef',
    'ModelConfigDict',
    'ModelInfo',
    'ModelStreamResponse',
    # Errors
    'ErrorResponseMetadata',
    'GenkitError',
    'PublicError',
    # Tools
    'Interrupt',
    'Tool',
    'respond_to_interrupt',
    'restart_tool',
    'tool',
    'background_model',
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
    'MultipartToolResponse',
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
    # Middleware references (wire form for use= parameter)
    'MiddlewareRef',
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
