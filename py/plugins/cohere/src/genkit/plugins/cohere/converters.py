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

"""Type converters between Genkit and Cohere V2 API formats.

This module contains pure-function converters that translate between Genkit
types (messages, responses, tools, usage) and their Cohere V2 API
counterparts.  Keeping them in a dedicated module makes each converter
independently testable without requiring a live Cohere client.

Converter categories
====================

========================= ====================================================
Category                  Functions
========================= ====================================================
Messages                  :func:`convert_messages` — Genkit → Cohere V2
Responses                 :func:`convert_response` — Cohere V2 → Genkit
Usage                     :func:`convert_usage` — Cohere Usage → Genkit
Tools                     :func:`convert_tools` — Genkit tool defs → Cohere V2
Response format           :func:`get_response_format` — output config → dict
Tool-call argument parse  :func:`parse_tool_arguments` — raw args → dict|str
Streaming extractors      :func:`extract_content_delta_text`, etc.
========================= ====================================================
"""

from __future__ import annotations

import json
from typing import Any

from cohere.types import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    SystemChatMessageV2,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    ToolV2,
    ToolV2Function,
    Usage,
    UserChatMessageV2,
)

try:
    from cohere.types import TextAssistantMessageV2ContentItem
except ImportError:
    from cohere.types import TextAssistantMessageV2ContentOneItem as TextAssistantMessageV2ContentItem
from cohere.v2.types.v2chat_response import V2ChatResponse
from genkit.core.typing import (
    FinishReason,
    GenerateResponse,
    GenerationUsage,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)

# Cohere V2 message union type.
CohereV2Message = UserChatMessageV2 | AssistantChatMessageV2 | SystemChatMessageV2 | ToolChatMessageV2

# Cohere finish reason string → Genkit FinishReason.
FINISH_REASON_MAP: dict[str, FinishReason] = {
    'COMPLETE': FinishReason.STOP,
    'STOP_SEQUENCE': FinishReason.STOP,
    'MAX_TOKENS': FinishReason.LENGTH,
    'ERROR': FinishReason.OTHER,
    'ERROR_TOXIC': FinishReason.BLOCKED,
    'ERROR_LIMIT': FinishReason.LENGTH,
    'TOOL_CALL': FinishReason.STOP,
}


def convert_messages(messages: list[Message]) -> list[CohereV2Message]:
    """Convert Genkit messages to Cohere V2 message format.

    The Cohere V2 API accepts a flat list of typed message objects
    (not a union wrapper), each with a ``role`` field.

    Args:
        messages: List of Genkit messages.

    Returns:
        List of Cohere SDK V2 message objects.
    """
    cohere_messages: list[CohereV2Message] = []

    for msg in messages:
        text_parts: list[str] = []
        tool_calls: list[ToolCallV2] = []
        tool_responses: list[tuple[str, str]] = []  # (call_id, output)

        for part in msg.content:
            part_root = part.root
            if isinstance(part_root, TextPart):
                text_parts.append(part_root.text)
            elif isinstance(part_root, ToolRequestPart):
                tool_req = part_root.tool_request
                tool_calls.append(
                    ToolCallV2(
                        id=tool_req.ref or '',
                        type='function',
                        function=ToolCallV2Function(
                            name=tool_req.name,
                            arguments=json.dumps(tool_req.input) if tool_req.input else '{}',
                        ),
                    )
                )
            elif isinstance(part_root, ToolResponsePart):
                tool_resp = part_root.tool_response
                output = tool_resp.output
                if isinstance(output, dict):
                    output_str = json.dumps(output)
                else:
                    output_str = str(output) if output is not None else ''
                tool_responses.append((tool_resp.ref or '', output_str))

        content_str = '\n'.join(text_parts) if text_parts else ''

        if msg.role == Role.SYSTEM:
            cohere_messages.append(SystemChatMessageV2(content=content_str))
        elif msg.role == Role.USER:
            cohere_messages.append(UserChatMessageV2(content=content_str))
        elif msg.role == Role.MODEL:
            if tool_calls:
                cohere_messages.append(
                    AssistantChatMessageV2(
                        content=content_str if content_str else None,
                        tool_calls=tool_calls,
                    )
                )
            else:
                cohere_messages.append(AssistantChatMessageV2(content=content_str))
        elif msg.role == Role.TOOL:
            for call_id, output_str in tool_responses:
                cohere_messages.append(
                    ToolChatMessageV2(
                        tool_call_id=call_id,
                        content=output_str,
                    )
                )

    return cohere_messages


def parse_tool_arguments(arguments: str | dict[str, object] | int | float | bool | None) -> dict[str, object] | str:
    """Parse tool call arguments from various raw formats.

    The Cohere V2 API may return arguments as a JSON string, a dict,
    or some other type.  This helper normalises the value.

    Args:
        arguments: Raw tool-call arguments from the Cohere response.

    Returns:
        Parsed dict if valid JSON string, the dict itself if already a
        dict, otherwise the stringified value.
    """
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    if isinstance(arguments, dict):
        return arguments
    return str(arguments)


def convert_response(response: V2ChatResponse) -> GenerateResponse:
    """Convert a Cohere V2 response to a Genkit ``GenerateResponse``.

    Args:
        response: Cohere V2 chat response.

    Returns:
        Genkit GenerateResponse.
    """
    content: list[Part] = []

    msg: AssistantMessageResponse | None = response.message if response else None
    if msg and msg.content:
        for block in msg.content:
            if isinstance(block, TextAssistantMessageV2ContentItem) and block.text:
                content.append(Part(root=TextPart(text=block.text)))

    # Handle tool calls in the response.
    if msg and msg.tool_calls:
        for tool_call in msg.tool_calls:
            args: dict[str, Any] | str = {}
            func = tool_call.function
            if func and func.arguments:
                args = parse_tool_arguments(func.arguments)

            func_name = (func.name if func else None) or ''
            content.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref=tool_call.id or None,
                            name=func_name,
                            input=args,
                        )
                    )
                )
            )

    message = Message(role=Role.MODEL, content=content)

    usage = convert_usage(response.usage) if response.usage else None

    finish_reason = FinishReason.STOP
    if response.finish_reason:
        finish_reason = FINISH_REASON_MAP.get(str(response.finish_reason), FinishReason.OTHER)

    return GenerateResponse(
        message=message,
        finish_reason=finish_reason,
        usage=usage,
    )


def convert_usage(usage: Usage) -> GenerationUsage:
    """Convert Cohere ``Usage`` to Genkit ``GenerationUsage``.

    Prefers billed units when available, falling back to token counts.

    Args:
        usage: Cohere Usage object.

    Returns:
        Genkit GenerationUsage.
    """
    billed = usage.billed_units
    tokens = usage.tokens
    input_tokens = billed.input_tokens if billed and billed.input_tokens else None
    output_tokens = billed.output_tokens if billed and billed.output_tokens else None
    if tokens:
        input_tokens = input_tokens or getattr(tokens, 'input_tokens', None)
        output_tokens = output_tokens or getattr(tokens, 'output_tokens', None)
    total = (input_tokens or 0) + (output_tokens or 0) if input_tokens or output_tokens else None
    return GenerationUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total,
    )


def convert_tools(tools: list[ToolDefinition]) -> list[ToolV2]:
    """Convert Genkit tool definitions to Cohere V2 ``ToolV2`` objects.

    Cohere V2 uses JSON-Schema–based tool definitions (like OpenAI).

    Args:
        tools: List of Genkit tool definitions.

    Returns:
        List of Cohere V2 Tool objects.
    """
    cohere_tools: list[ToolV2] = []
    for tool in tools:
        parameters = tool.input_schema or {}
        cohere_tools.append(
            ToolV2(
                type='function',
                function=ToolV2Function(
                    name=tool.name,
                    description=tool.description or '',
                    parameters=parameters,
                ),
            )
        )
    return cohere_tools


def get_response_format(output: OutputConfig) -> dict[str, Any] | None:
    """Build the Cohere ``response_format`` parameter from output config.

    Args:
        output: Output configuration specifying desired format.

    Returns:
        Response format dict for the Cohere API, or ``None`` for default.
    """
    if output.format == 'json':
        if output.schema:
            return {
                'type': 'json_object',
                'json_schema': output.schema,
            }
        return {'type': 'json_object'}
    return None


def extract_content_delta_text(event: object) -> str:
    """Extract text from a content-delta stream event.

    Args:
        event: A content-delta stream event.

    Returns:
        The delta text, or empty string if not available.
    """
    delta = getattr(event, 'delta', None)
    if delta is None:
        return ''
    msg = getattr(delta, 'message', None)
    if msg is None:
        return ''
    content = getattr(msg, 'content', None)
    if content is None:
        return ''
    return str(getattr(content, 'text', '') or '')


def extract_tool_call_start(event: object) -> tuple[str, str]:
    """Extract tool call id and name from a tool-call-start event.

    Args:
        event: A tool-call-start stream event.

    Returns:
        Tuple of ``(tool_call_id, function_name)``.
    """
    delta = getattr(event, 'delta', None)
    if delta is None:
        return ('', '')
    msg = getattr(delta, 'message', None)
    if msg is None:
        return ('', '')
    tc = getattr(msg, 'tool_calls', None)
    if tc is None:
        return ('', '')
    tc_id = str(getattr(tc, 'id', '') or '')
    func = getattr(tc, 'function', None)
    tc_name = str(getattr(func, 'name', '') or '') if func else ''
    return (tc_id, tc_name)


def extract_tool_call_delta_args(event: object) -> str:
    """Extract argument fragment from a tool-call-delta event.

    Args:
        event: A tool-call-delta stream event.

    Returns:
        The argument string fragment, or empty string.
    """
    delta = getattr(event, 'delta', None)
    if delta is None:
        return ''
    msg = getattr(delta, 'message', None)
    if msg is None:
        return ''
    tc = getattr(msg, 'tool_calls', None)
    if tc is None:
        return ''
    func = getattr(tc, 'function', None)
    if func is None:
        return ''
    return str(getattr(func, 'arguments', '') or '')


def extract_finish_reason(event: object) -> str:
    """Extract finish reason from a message-end event.

    Args:
        event: A message-end stream event.

    Returns:
        The finish reason string, or empty string.
    """
    delta = getattr(event, 'delta', None)
    if delta is None:
        return ''
    fr = getattr(delta, 'finish_reason', None)
    return str(fr) if fr else ''
