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

"""User-facing types for the Genkit framework.

This module re-exports all public types that users may need when building
Genkit applications. It provides a single import point for common types
like Message, Part, Document, and response types.

Overview:
    Types in Genkit are organized into categories based on their use case.
    This module exports types that are commonly needed in user code.

Type Categories:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Category                │ Key Types                                     │
    ├─────────────────────────┼───────────────────────────────────────────────┤
    │ Message/Part            │ Message, Part, TextPart, MediaPart, Role      │
    │ Document                │ Document, DocumentData                        │
    │ Generation              │ GenerateRequest, GenerateResponse, OutputConf │
    │ Tools                   │ ToolRequest, ToolResponse, ToolDefinition     │
    │ Embedding               │ Embedding, EmbedRequest, EmbedResponse        │
    │ Retrieval               │ RetrieverRequest, RetrieverResponse           │
    │ Evaluation              │ EvalRequest, EvalResponse, Score              │
    │ Model Info              │ ModelInfo, Supports, Constrained              │
    │ Errors                  │ GenkitError, ToolInterruptError               │
    └─────────────────────────┴───────────────────────────────────────────────┘

Example:
    Importing types for a Genkit application:

    ```python
    from genkit import Genkit
    from genkit.types import Message, Part, TextPart, Document

    ai = Genkit(...)

    # Create a message manually
    message = Message(
        role='user',
        content=[Part(root=TextPart(text='Hello!'))],
    )

    # Create a document
    doc = Document.from_text('Some content', metadata={'source': 'file.txt'})
    ```

    Plugin authors may need model info types:

    ```python
    from genkit.types import ModelInfo, Supports

    model_info = ModelInfo(
        label='My Model',
        supports=Supports(
            multiturn=True,
            tools=True,
            systemRole=True,
        ),
    )
    ```

See Also:
    - genkit.core.typing: Source definitions for these types
    - genkit.blocks.document: Document class implementation
"""

from genkit.blocks.document import Document
from genkit.blocks.tools import ToolInterruptError
from genkit.core.action._action import ActionRunContext
from genkit.core.error import GenkitError
from genkit.core.typing import (
    # Eval types
    BaseEvalDataPoint,
    Constrained,
    CustomPart,
    DataPart,
    # Document types
    DocumentData,
    # Embedding types
    Embedding,
    EmbedRequest,
    EmbedResponse,
    EvalFnResponse,
    EvalRequest,
    EvalResponse,
    EvalStatusEnum,
    FinishReason,
    GenerateActionOptions,
    # Generation types
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChunk,
    GenerationUsage,
    Media,
    MediaPart,
    # Message and Part types
    Message,
    Metadata,
    # Model info (for plugin authors)
    ModelInfo,
    OutputConfig,
    Part,
    ReasoningPart,
    # Retriever types
    RetrieverRequest,
    RetrieverResponse,
    Role,
    Score,
    Stage,
    Supports,
    TextPart,
    ToolChoice,
    ToolDefinition,
    # Tool types
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.model_types import GenerationCommonConfig, get_effective_api_key, get_request_api_key

__all__ = [
    # Errors
    'GenkitError',
    'ToolInterruptError',
    # Action context
    'ActionRunContext',
    # Message and Part types
    'CustomPart',
    'DataPart',
    'Media',
    'MediaPart',
    'Message',
    'Metadata',
    'Part',
    'ReasoningPart',
    'Role',
    'TextPart',
    'ToolRequestPart',
    'ToolResponsePart',
    # Document types
    'Document',
    'DocumentData',
    # Generation types
    'FinishReason',
    'GenerateActionOptions',
    'GenerateRequest',
    'GenerateResponse',
    'GenerateResponseChunk',
    'GenerationCommonConfig',
    'get_request_api_key',
    'get_effective_api_key',
    'GenerationUsage',
    'OutputConfig',
    'ToolChoice',
    # Tool types
    'ToolDefinition',
    'ToolRequest',
    'ToolResponse',
    # Embedding types
    'EmbedRequest',
    'EmbedResponse',
    'Embedding',
    # Retriever types
    'RetrieverRequest',
    'RetrieverResponse',
    # Eval types
    'BaseEvalDataPoint',
    'EvalFnResponse',
    'EvalRequest',
    'EvalResponse',
    'EvalStatusEnum',
    'Score',
    # Model info (for plugin authors)
    'Constrained',
    'ModelInfo',
    'Stage',
    'Supports',
]
