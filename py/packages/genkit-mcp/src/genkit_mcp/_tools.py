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
from collections.abc import Awaitable, Callable
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
from genkit import MultipartToolResponse
from genkit._core._action import Action, ActionRunContext
from genkit._core._typing import Media, MediaPart, Part, Resource, ResourcePart

McpToolCall = Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[CallToolResult]]

logger = structlog.get_logger(__name__)

MAX_TOOL_NAME_LENGTH = 64
"""Longest function declaration name both Gemini and OpenAI accept."""

TOOL_NAME_PATTERN = re.compile(r'[A-Za-z_][A-Za-z0-9_-]*')
"""What both Gemini and OpenAI accept in a function declaration name."""

SELECTOR_SEPARATORS = (':', '/')
"""Characters a tool selector puts around a provider name, so a name cannot hold them."""


def validate_provider_name(name: str) -> None:
    """Reject a provider name no tool selector could carry.

    Looser than the tool name rule a prefix has to meet: a provider name reaches
    no model, so only the selector constrains it.

    Args:
        name: Candidate name for a client or host.

    Raises:
        ValueError: If ``<name>:tool/<tool>`` would not parse as a selector.
    """
    if name and not any(separator in name for separator in SELECTOR_SEPARATORS):
        return
    raise ValueError(
        f'{name!r} cannot name an MCP provider: its tools are selected as '
        f"'{name}:tool/<tool>', so the name must not be empty and must hold no ':' or '/'."
    )


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
        'tool_prefix= with a name that can.'
    )


def mcp_tool_name(prefix: str, tool_name: str) -> str:
    """Return the Genkit tool name for an MCP tool.

    The separator is an underscore because Gemini and OpenAI reject a ``/`` in a
    function declaration name.

    Args:
        prefix: Namespace for this server's tools.
        tool_name: Tool name as listed by the server.

    Returns:
        The namespaced name sent to the model.
    """
    return f'{prefix}_{tool_name}'


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
            content.append(Part(root=ResourcePart(resource=Resource(uri=str(block.uri)))))
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
    return Part(
        root=MediaPart(
            media=Media(
                url=f'data:{mime_type};base64,{data}',
                content_type=mime_type,
            )
        )
    )


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
    which is what a dynamic action provider matches a wildcard selector against.

    Args:
        tool: Tool as listed by the server.
        prefix: Namespace for the Genkit tool name.
        call: Coroutine function invoking the tool on the live session.

    Returns:
        An unregistered action, for a dynamic action provider to hand out.
    """
    tool_name = mcp_tool_name(prefix, tool.name)
    if not TOOL_NAME_PATTERN.fullmatch(tool_name) or len(tool_name) > MAX_TOOL_NAME_LENGTH:
        logger.warning(
            'MCP tool name will be rejected by most model providers',
            tool=tool_name,
            server_tool=tool.name,
        )
    mcp_meta = dict(tool.meta) if tool.meta is not None else {}

    async def handler(input: dict[str, Any], ctx: ActionRunContext) -> MultipartToolResponse:
        context_mcp = ctx.context.get('mcp', {})
        meta = context_mcp.get('_meta') if isinstance(context_mcp, dict) else None
        try:
            result = await call(tool.name, input, meta)
        except RuntimeError as error:
            # The SDK reports a result which fails the server's own output schema
            # as a bare RuntimeError. Everything this package raises is an
            # McpClientError, which is not a RuntimeError and so propagates: a
            # dead subprocess must not reach the model as a retryable tool error.
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
    action.metadata['mcp'] = {'prefix': prefix, '_meta': mcp_meta}
    return action
