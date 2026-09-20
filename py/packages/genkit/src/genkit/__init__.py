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
from genkit._ai._prompt import (
    ExecutablePrompt,
    ModelStreamResponse,
)
from genkit._ai._tools import (
    Interrupt,
    MultipartToolResponse,
    Tool,
    ToolRunContext,
    respond_to_interrupt,
    response,
    restart_tool,
    tool,
)
from genkit._core._action import Action as Flow, ActionRunContext
from genkit._core._context import ContextProvider, RequestData
from genkit._core._error import GenkitError, PublicError
from genkit._core._model import Document, Message, Part
from genkit._core._typing import (
    Media,
    Role,
)
from genkit.model import (
    FinishReason,
    ModelResponse,
    ModelResponseChunk,
)

__all__ = [
    'Genkit',
    # Construct a turn
    'Message',
    'Role',
    'Part',
    'Media',
    'Document',
    # What came back
    'ModelResponse',
    'ModelResponseChunk',
    'ModelStreamResponse',
    'FinishReason',
    # Tools and HITL
    'tool',
    'Tool',
    'ToolRunContext',
    'Interrupt',
    'respond_to_interrupt',
    'restart_tool',
    'response',
    'MultipartToolResponse',
    # Flows, prompts, errors
    'Flow',
    'ActionRunContext',
    'ExecutablePrompt',
    'GenkitError',
    'PublicError',
    # HTTP request context for flow handlers
    'ContextProvider',
    'RequestData',
]
