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

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import structlog
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
    Tool as McpTool,
)

import genkit
from genkit import Action, ActionRunContext, MultipartToolResponse, Part
from genkit._core._typing import Resource
from genkit_mcp._errors import McpToolResultError

McpToolCall = Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[CallToolResult]]

logger = structlog.get_logger(__name__)

MAX_TOOL_NAME_LENGTH = 64
"""Longest function declaration name both Gemini and OpenAI accept."""

TOOL_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*')
"""What both Gemini and OpenAI accept in a function declaration name."""

_TOOL_NAME_REJECTED_CHARS = re.compile(r'[^A-Za-z0-9_-]')


def validate_tool_prefix(prefix: str) -> None:
    """Reject a prefix that cannot appear in a model-facing tool name.

    The length is not checked, because the tool half of the name comes from the
    server and is unknown until it is listed.

    Args:
        prefix: Candidate prefix.

    Raises:
        ValueError: If a name built from this prefix would be rejected by the
            model providers.
    """
    if TOOL_NAME_PATTERN.fullmatch(prefix):
        return
    raise ValueError(
        f'{prefix!r} cannot prefix an MCP tool name: a function declaration name must start with a '
        'letter or an underscore and hold only letters, digits, underscores and hyphens. Pass '
        'tool_prefix= to define_mcp_client with a name that can.'
    )


def mcp_tool_name(prefix: str, tool_name: str) -> str:
    """Return the Genkit tool name for an MCP tool.

    The prefix and the tool name join with an underscore, and every character
    Gemini or OpenAI rejects in a function declaration name becomes an
    underscore too. MCP allows a ``.`` in a tool name and some servers use a
    ``/``; either would make every generate call carrying the tool fail. The
    name is only what the model and the registry see, so the rewrite never
    reaches the server.

    Args:
        prefix: Namespace for this server's tools.
        tool_name: Tool name as listed by the server.

    Returns:
        The namespaced name sent to the model.
    """
    name = _TOOL_NAME_REJECTED_CHARS.sub('_', f'{prefix}_{tool_name}')
    if not TOOL_NAME_PATTERN.fullmatch(name):
        name = f'_{name}'
    return name


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
    metadata = dict(result.meta) if result.meta is not None else None
    if result.isError:
        error_text = ''.join(block.text for block in result.content if isinstance(block, TextContent))
        return MultipartToolResponse(
            output={'error': error_text},
            metadata={**(metadata or {}), 'mcp': {'isError': True}},
        )

    content: list[Part] = []
    text_output = ''
    for block in result.content:
        if isinstance(block, TextContent):
            text_output += block.text
        elif isinstance(block, (ImageContent, AudioContent)):
            content.append(_media_part(block.mimeType, block.data))
        elif isinstance(block, ResourceLink):
            content.append(Part(resource=Resource(uri=str(block.uri))))
        elif isinstance(block, EmbeddedResource):
            resource = block.resource
            if isinstance(resource, TextResourceContents):
                if text_output:
                    text_output += '\n\n'
                text_output += f'Resource ({resource.uri}):\n{resource.text}'
            elif isinstance(resource, BlobResourceContents):
                content.append(_media_part(resource.mimeType or 'application/octet-stream', resource.blob))

    output: object | None = result.structuredContent
    if output is None and text_output:
        output = _text_output(text_output)
    return MultipartToolResponse(
        output=output,
        content=content or None,
        metadata=metadata,
    )


def _media_part(mime_type: str, data: str) -> Part:
    """Wrap base64 MCP binary content in the Genkit media wire format."""
    return Part.from_media(url=f'data:{mime_type};base64,{data}', content_type=mime_type)


def _text_output(text: str) -> object:
    """Decode object and array text while preserving scalar-looking text."""
    if text.lstrip().startswith(('{', '[')):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def mcp_tool_to_action(
    tool: McpTool,
    prefix: str,
    call: McpToolCall,
) -> Action:
    """Build a Genkit ``tool.v2`` action that invokes one MCP tool.

    The returned action carries its own name under the ``name`` metadata key,
    which is what a dynamic action provider matches a wildcard selector against,
    and the server's own tool name under ``metadata['mcp']['tool']``.

    Args:
        tool: Tool as listed by the server.
        prefix: Namespace for the Genkit tool name.
        call: Coroutine function invoking the tool on the live session.

    Returns:
        An unregistered action, for a dynamic action provider to hand out.
    """
    tool_name = mcp_tool_name(prefix, tool.name)
    if tool_name != f'{prefix}_{tool.name}':
        logger.warning(
            'MCP tool name rewritten: the model providers reject the server name',
            tool=tool_name,
            server_tool=tool.name,
        )
    if len(tool_name) > MAX_TOOL_NAME_LENGTH:
        logger.warning(
            'MCP tool name is longer than most model providers accept',
            tool=tool_name,
            server_tool=tool.name,
        )
    mcp_meta = dict(tool.meta) if tool.meta is not None else {}

    async def handler(input: dict[str, Any], ctx: ActionRunContext) -> MultipartToolResponse:
        context_mcp = ctx.context.get('mcp', {})
        meta = context_mcp.get('_meta') if isinstance(context_mcp, dict) else None
        try:
            result = await call(tool.name, input, meta)
        except McpToolResultError as error:
            result = CallToolResult(
                content=[TextContent(type='text', text=str(error))],
                isError=True,
            )
        return call_result_to_multipart(result)

    action = genkit.tool(
        handler,
        name=tool_name,
        description=tool.description or '',
        input_schema=tool.inputSchema,
    ).action()
    action.metadata['name'] = action.name
    action.metadata['mcp'] = {'prefix': prefix, 'tool': tool.name, '_meta': mcp_meta}
    return action


def mcp_tools_to_actions(
    tools: Sequence[McpTool],
    prefix: str,
    call: McpToolCall,
) -> list[Action]:
    """Build one Genkit ``tool.v2`` action per tool in a server listing.

    Args:
        tools: Tools as listed by the server.
        prefix: Namespace for the Genkit tool names.
        call: Coroutine function invoking a tool on the live session.

    Returns:
        Unregistered actions in listing order, for a dynamic action provider to
        hand out.

    Raises:
        ValueError: If two tools in the listing share a Genkit tool name once
            the characters the model providers reject are rewritten.
    """
    actions: list[Action] = []
    server_tool_by_name: dict[str, str] = {}
    for tool in tools:
        action = mcp_tool_to_action(tool, prefix, call)
        if action.name in server_tool_by_name:
            raise ValueError(
                f'MCP tools {server_tool_by_name[action.name]!r} and {tool.name!r} both map to the '
                f'Genkit tool name {action.name!r} once the characters the model providers reject are '
                'rewritten. Rename one of them on the server.'
            )
        server_tool_by_name[action.name] = tool.name
        actions.append(action)
    return actions
