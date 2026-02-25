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

"""Cloudflare Workers AI format conversion utilities.

Pure-function helpers for converting between Genkit types and the Cloudflare
Workers AI chat completion API format. Extracted from the model module for
independent unit testing.

See: https://developers.cloudflare.com/workers-ai/
"""

import json
from typing import Any

from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig
from genkit.types import (
    GenerationCommonConfig,
    GenerationUsage,
    Message,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponsePart,
)

__all__ = [
    'build_usage',
    'normalize_config',
    'parse_sse_line',
    'parse_tool_calls',
    'to_cloudflare_messages_sync',
    'to_cloudflare_role',
    'to_cloudflare_tool',
    'wrap_non_object_schema',
]


def to_cloudflare_role(role: Role | str) -> str:
    """Convert a Genkit role to a Cloudflare role string.

    Args:
        role: Genkit Role enum or string.

    Returns:
        Cloudflare-compatible role string.
    """
    if isinstance(role, str):
        role_str = role.lower()
    else:
        role_str = role.value.lower() if hasattr(role, 'value') else str(role).lower()

    role_mapping = {
        'user': 'user',
        'model': 'assistant',
        'system': 'system',
        'tool': 'tool',
    }
    return role_mapping.get(role_str, 'user')


def wrap_non_object_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Wrap a non-object schema in an object wrapper.

    Cloudflare expects tool parameters to be an object schema. If the input
    schema is a primitive type, wrap it in ``{'type': 'object', ...}``.

    Args:
        schema: Input JSON schema or None.

    Returns:
        Object-type JSON schema suitable for Cloudflare tools.
    """
    params = schema or {'type': 'object', 'properties': {}}
    if params.get('type') != 'object':
        params = {
            'type': 'object',
            'properties': {'input': params},
            'required': ['input'],
        }
    return params


def to_cloudflare_tool(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a Genkit tool definition to Cloudflare format.

    Args:
        tool: Genkit ToolDefinition.

    Returns:
        Cloudflare-compatible tool specification.
    """
    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description or '',
            'parameters': wrap_non_object_schema(tool.input_schema),
        },
    }


def to_cloudflare_messages_sync(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert Genkit messages to Cloudflare API message format (sync, no media fetch).

    This handles text, tool-request, and tool-response parts. For MediaPart
    that requires async URL fetching, the caller must handle them separately.

    Args:
        messages: List of Genkit messages.

    Returns:
        List of Cloudflare-compatible message dictionaries.
    """
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = to_cloudflare_role(msg.role)
        text_content = ''

        for part in msg.content:
            root = part.root if isinstance(part, Part) else part

            if isinstance(root, TextPart):
                text_content += root.text

            elif isinstance(root, ToolRequestPart):
                tool_req = root.tool_request
                tool_call_obj = {
                    'name': tool_req.name,
                    'arguments': tool_req.input if isinstance(tool_req.input, dict) else {'input': tool_req.input},
                }
                result.append({'role': 'assistant', 'content': json.dumps(tool_call_obj)})
                continue

            elif isinstance(root, ToolResponsePart):
                tool_resp = root.tool_response
                output = tool_resp.output
                output_str = json.dumps(output) if isinstance(output, dict) else str(output)
                result.append({'role': 'tool', 'name': tool_resp.name, 'content': output_str})
                continue

        if text_content:
            result.append({'role': role, 'content': text_content})

    return result


def parse_tool_calls(tool_calls: list[dict[str, Any]]) -> list[Part]:
    """Parse Cloudflare tool call dicts into Genkit ToolRequestParts.

    Args:
        tool_calls: List of tool call dicts from the Cloudflare response.

    Returns:
        List of Genkit Part objects containing ToolRequestParts.
    """
    parts: list[Part] = []
    for tc in tool_calls:
        parts.append(
            Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(
                        name=tc.get('name', ''),
                        input=tc.get('arguments', {}),
                    )
                )
            )
        )
    return parts


def build_usage(usage_data: dict[str, Any]) -> GenerationUsage:
    """Build GenerationUsage from Cloudflare usage data.

    Args:
        usage_data: Usage dict from the Cloudflare response.

    Returns:
        GenerationUsage with token counts.
    """
    return GenerationUsage(
        input_tokens=usage_data.get('prompt_tokens', 0),
        output_tokens=usage_data.get('completion_tokens', 0),
        total_tokens=usage_data.get('total_tokens', 0),
    )


def parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a Server-Sent Events data line.

    Returns None for non-data lines, empty lines, and the ``[DONE]`` sentinel.

    Args:
        line: Raw SSE line.

    Returns:
        Parsed JSON dict or None.
    """
    stripped = line.strip()
    if not stripped or not stripped.startswith('data: '):
        return None

    payload = stripped[6:]
    if payload == '[DONE]':
        return None

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def normalize_config(config: object) -> CloudflareConfig:
    """Normalize config to CloudflareConfig.

    Handles dicts with camelCase keys, GenerationCommonConfig, and
    CloudflareConfig passthrough.

    Args:
        config: Request configuration.

    Returns:
        Normalized CloudflareConfig instance.
    """
    if config is None:
        return CloudflareConfig()

    if isinstance(config, CloudflareConfig):
        return config

    if isinstance(config, GenerationCommonConfig):
        return CloudflareConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            top_p=config.top_p,
            stop_sequences=config.stop_sequences,
        )

    if isinstance(config, dict):
        mapped: dict[str, Any] = {}
        key_map: dict[str, str] = {
            'maxOutputTokens': 'max_output_tokens',
            'maxTokens': 'max_output_tokens',
            'topP': 'top_p',
            'topK': 'top_k',
            'stopSequences': 'stop_sequences',
            'repetitionPenalty': 'repetition_penalty',
            'frequencyPenalty': 'frequency_penalty',
            'presencePenalty': 'presence_penalty',
        }
        for key, value in config.items():
            str_key = str(key)
            mapped_key = key_map.get(str_key, str_key)
            mapped[mapped_key] = value
        return CloudflareConfig(**mapped)

    return CloudflareConfig()
