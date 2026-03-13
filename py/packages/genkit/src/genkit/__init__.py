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

"""Genkit - Build AI-powered applications with ease.

Genkit is an open-source Python toolkit designed to help you build
AI-powered features in web and mobile apps.

Basic usage:
    from genkit import Genkit
    from genkit.plugins.google_genai import GoogleAI

    ai = Genkit(plugins=[GoogleAI()])

    @ai.flow()
    async def hello(name: str) -> str:
        response = await ai.generate(model="gemini-2.0-flash", prompt=f"Hello {name}")
        return response.text
"""

# Main class
# Re-export everything from genkit.ai for backwards compatibility
from genkit.ai import (
    GENKIT_CLIENT_HEADER,
    GENKIT_VERSION,
    ActionKind,
    ActionRunContext,
    ExecutablePrompt,
    FlowWrapper,
    GenerateStreamResponse,
    GenkitRegistry,
    OutputOptions,
    PromptGenerateOptions,
    ResumeOptions,
    SimpleRetrieverOptions,
    ToolRunContext,
    tool_response,
)
from genkit.ai._aio import Genkit, Output

# Core types for convenience (also available from genkit.types)
from genkit.blocks.document import Document
from genkit.blocks.interfaces import Input

# Response types
from genkit.blocks.model import GenerateResponseWrapper

# Setup plugin discovery (must be done before any plugin imports)
from genkit.core._plugins import extend_plugin_namespace

# Errors (user-facing)
from genkit.core.error import GenkitError, UserFacingError

# Plugin interface
from genkit.core.plugin import Plugin
from genkit.core.typing import (
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
)

extend_plugin_namespace()

__all__ = [
    # Main class
    'Genkit',
    'Input',
    'Output',
    # Response types
    'GenerateResponseWrapper',
    'GenerateStreamResponse',
    # Errors
    'GenkitError',
    'UserFacingError',
    # Core types (convenience)
    'Document',
    'Media',
    'MediaPart',
    'Message',
    'Part',
    'Role',
    'TextPart',
    # Plugin interface
    'Plugin',
    # From genkit.ai
    'ActionKind',
    'ActionRunContext',
    'ExecutablePrompt',
    'FlowWrapper',
    'GenkitRegistry',
    'OutputOptions',
    'PromptGenerateOptions',
    'ResumeOptions',
    'SimpleRetrieverOptions',
    'ToolRunContext',
    'tool_response',
    # Version info
    'GENKIT_CLIENT_HEADER',
    'GENKIT_VERSION',
]
