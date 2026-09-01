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

from genkit._ai._aio import Genkit
from genkit._ai._prompt import (
    ExecutablePrompt,
    ModelStreamResponse,
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
from genkit._core._error import GenkitError, PublicError
from genkit._core._model import Document
from genkit._core._typing import (
    Media,
    Part,
    Role,
    ToolChoice,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.embedder import Embedding
from genkit.model import (
    FinishReason,
    Message,
    ModelConfigDict,
    ModelResponse,
    ModelResponseChunk,
    ModelUsage,
)

# Flow is an alias for Action (used in samples for flow type hints)
Flow = Action

__all__ = [
    # Main class & flows
    'Genkit',
    'Action',
    'ActionRunContext',
    'Flow',
    'StreamResponse',
    # Model generation & streaming
    'ModelResponse',
    'ModelResponseChunk',
    'ModelStreamResponse',
    'ModelConfigDict',
    'ModelUsage',
    'FinishReason',
    'ToolChoice',
    # Content & Messaging
    'Message',
    'Part',
    'Role',
    'Media',
    'Document',
    # Tools & HITL
    'Tool',
    'ToolRunContext',
    'Interrupt',
    'ToolRequest',
    'ToolResponse',
    'ToolRequestPart',
    'ToolResponsePart',
    'respond_to_interrupt',
    'restart_tool',
    'tool',
    # Embeddings & Prompts
    'Embedding',
    'ExecutablePrompt',
    # Errors
    'GenkitError',
    'PublicError',
]
