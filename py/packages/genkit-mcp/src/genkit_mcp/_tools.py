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

"""Translation between MCP tools and Genkit ``tool.v2`` actions."""

from mcp.types import CallToolResult, Tool as McpTool

from genkit import MultipartToolResponse
from genkit._core._action import Action


def mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Return the Genkit tool name for an MCP tool.

    The separator is an underscore because Gemini and OpenAI reject a ``/`` in a
    function declaration name.

    Args:
        server_name: Name the server advertises, or the caller's override.
        tool_name: Tool name as listed by the server.

    Returns:
        The namespaced name sent to the model.
    """
    raise NotImplementedError


def call_result_to_multipart(result: CallToolResult) -> MultipartToolResponse:
    """Convert an MCP ``CallToolResult`` into a Genkit multipart tool response.

    Text blocks are concatenated into ``output``, parsed as JSON only when the
    text begins with ``{`` or ``[``. ``structuredContent`` wins over text when
    present. Image, audio and blob resource blocks become media parts, and a
    resource link becomes a resource part. An error result yields an ``error``
    output and drops any media, so the model can read the failure and retry.

    Args:
        result: What ``ClientSession.call_tool`` returned.

    Returns:
        The envelope ``generate`` expects from a ``tool.v2`` action.
    """
    raise NotImplementedError


def mcp_tool_to_action(
    tool: McpTool,
    server_name: str,
    call: object,
) -> Action:
    """Build a Genkit ``tool.v2`` action that invokes one MCP tool.

    The returned action carries its own name under the ``name`` metadata key,
    which is what a dynamic action provider matches a wildcard selector against.

    Args:
        tool: Tool as listed by the server.
        server_name: Prefix for the Genkit tool name.
        call: Coroutine function invoking the tool on the live session.

    Returns:
        An unregistered action, for a dynamic action provider to hand out.
    """
    raise NotImplementedError
