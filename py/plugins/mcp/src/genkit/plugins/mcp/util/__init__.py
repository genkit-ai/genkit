# Copyright 2026 Google LLC
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

"""Utility functions for MCP plugin.

This module contains helper functions for:
- Tool conversion and registration
- Prompt conversion and rendering
- Resource handling
- Message mapping between Genkit and MCP formats
- Transport utilities
"""

from .message import from_mcp_part, from_mcp_prompt_message, to_mcp_prompt_message
from .prompts import convert_mcp_prompt_messages, convert_prompt_arguments_to_schema, to_mcp_prompt_arguments, to_schema
from .resource import (
    convert_resource_to_genkit_part,
    from_mcp_resource_part,
    process_resource_content,
    to_mcp_resource_contents,
)
from .tools import convert_tool_schema, process_result, process_tool_result, to_mcp_tool_result, to_text
from .transport import create_stdio_params, transport_from

__all__ = [
    'convert_mcp_prompt_messages',
    'convert_prompt_arguments_to_schema',
    'convert_resource_to_genkit_part',
    'convert_tool_schema',
    'create_stdio_params',
    'from_mcp_part',
    'from_mcp_prompt_message',
    'from_mcp_resource_part',
    'process_resource_content',
    'process_result',
    'process_tool_result',
    'to_mcp_prompt_arguments',
    'to_mcp_prompt_message',
    'to_mcp_resource_contents',
    'to_mcp_tool_result',
    'to_schema',
    'to_text',
    'transport_from',
]
