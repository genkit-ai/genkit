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

"""Genkit AI module - core Genkit class and related utilities.

This module provides the main Genkit class for building AI applications,
along with essential types for actions, sessions, prompts, and tools.

Overview:
    The ``genkit.ai`` module exposes the primary entry points for building
    AI-powered applications. The ``Genkit`` class is the main orchestrator
    that manages plugins, models, prompts, flows, and sessions.

Terminology:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Term                │ Description                                       │
    ├─────────────────────┼───────────────────────────────────────────────────┤
    │ Genkit              │ Main orchestrator class for AI applications       │
    │ Plugin              │ Extension that adds models, embedders, or other   │
    │                     │ capabilities (e.g., GoogleAI, Anthropic)          │
    │ Flow                │ A durable, traceable function for AI workflows    │
    │ Action              │ A strongly-typed, remotely callable function      │
    │ Tool                │ A function callable by models during generation   │
    │ Session/Chat        │ State management for multi-turn conversations     │
    │ Prompt              │ Reusable template for model interactions          │
    └─────────────────────┴───────────────────────────────────────────────────┘

Key Components:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Component               │ Purpose                                       │
    ├─────────────────────────┼───────────────────────────────────────────────┤
    │ Genkit                  │ Main class for building AI applications       │
    │ GenkitRegistry          │ Internal registry for actions and resources   │
    │ ExecutablePrompt        │ Callable prompt with template rendering       │
    │ Chat                    │ Stateful multi-turn conversation interface    │
    │ ActionRunContext        │ Execution context for actions (streaming)     │
    │ ToolRunContext          │ Execution context for tools (with interrupt)  │
    │ GenerateResponseWrapper │ Enhanced response with helper methods         │
    │ GenerateStreamResponse  │ Streaming response with chunks and final resp │
    └─────────────────────────┴───────────────────────────────────────────────┘

Example:
    Basic usage:

    ```python
    from genkit import Genkit
    from genkit.plugins.google_genai import GoogleAI

    # Initialize with plugins
    ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.0-flash')


    # Define a flow
    @ai.flow()
    async def hello(name: str) -> str:
        response = await ai.generate(prompt=f'Say hello to {name}')
        return response.text


    # Define a tool
    @ai.tool()
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 72°F'


    # Create a chat session
    chat = ai.chat(system='You are a helpful assistant.')
    response = await chat.send('What is the weather in Paris?')
    ```

See Also:
    - Genkit documentation: https://genkit.dev/
    - JavaScript SDK: https://github.com/firebase/genkit
"""

from genkit.blocks.document import Document
from genkit.blocks.interfaces import Input
from genkit.blocks.model import GenerateResponseWrapper
from genkit.blocks.prompt import (
    ExecutablePrompt,
    GenerateStreamResponse,
    OutputOptions,
    PromptGenerateOptions,
    ResumeOptions,
)
from genkit.blocks.tools import ToolRunContext, tool_response
from genkit.core import GENKIT_CLIENT_HEADER, GENKIT_VERSION
from genkit.core.action import ActionRunContext
from genkit.core.action.types import ActionKind
from genkit.core.plugin import Plugin

from ._aio import Genkit, Output
from ._registry import FlowWrapper, GenkitRegistry, SimpleRetrieverOptions

__all__ = [
    # Version info
    'GENKIT_CLIENT_HEADER',
    'GENKIT_VERSION',
    # Main class
    'Genkit',
    'Input',
    'Output',
    # Actions
    'ActionKind',
    'ActionRunContext',
    # Document
    'Document',
    # Prompts
    'ExecutablePrompt',
    'OutputOptions',
    'PromptGenerateOptions',
    'ResumeOptions',
    # Registry and flow
    'FlowWrapper',
    'GenkitRegistry',
    'SimpleRetrieverOptions',
    # Response types
    'GenerateResponseWrapper',
    'GenerateStreamResponse',
    # Tools
    'ToolRunContext',
    'tool_response',
    # Plugin
    'Plugin',
]
