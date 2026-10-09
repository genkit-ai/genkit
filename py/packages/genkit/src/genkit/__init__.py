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

    @ai.flow()
    async def my_flow(prompt: str) -> str:
        res = await ai.generate(prompt=prompt)
        return res.text

    if __name__ == '__main__':
        ai.run_main(my_flow('Weather in Paris?'))
"""

from genkit._ai._aio import Genkit
from genkit._ai._formats._types import FormatDef, FormatterConfig
from genkit._ai._prompt import (
    ModelStreamResponse,
    Prompt,
)
from genkit._ai._tools import (
    MultipartToolResponse,
    ToolRunContext,
    tool,
    tool_response,
)
from genkit._core._action import ActionRunContext, StreamResponse
from genkit._core._context import ContextProvider, RequestData
from genkit._core._dap import DynamicActionProvider
from genkit._core._error import GenkitError, GenkitRuntimeError, Interrupt, PublicError, RuntimeErrorReason
from genkit._core._logger import get_logger
from genkit._core._model import (
    Document,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    ToolChoice,
)
from genkit._core._tool import Tool
from genkit._core._typing import (
    Embedding,
    FinishReason,
    Media,
    Operation,
    Role,
)

__all__ = [
    'Genkit',
    # Construct a turn
    'Message',
    'Role',
    'Part',
    'Media',
    'Document',
    'ToolChoice',
    # What came back
    'ModelResponse',
    'ModelResponseChunk',
    'GenkitRuntimeError',
    'ModelStreamResponse',
    'StreamResponse',
    'FinishReason',
    # Embed, evaluate, and background jobs
    'Embedding',
    'Operation',
    # Tools and HITL
    'tool',
    'Tool',
    'ToolRunContext',
    'Interrupt',
    'tool_response',
    'MultipartToolResponse',
    # Flows, prompts, errors
    'ActionRunContext',
    'Prompt',
    'GenkitError',
    'PublicError',
    'RuntimeErrorReason',
    'get_logger',
    # HTTP request context for flow handlers
    'ContextProvider',
    'RequestData',
    # Custom output formats and dynamic providers
    'FormatDef',
    'FormatterConfig',
    'DynamicActionProvider',
]
