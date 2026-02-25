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

"""Ollama format conversion utilities.

Pure-function helpers for converting between Genkit types and Ollama
API types. Extracted from the model module for independent unit testing.

Functions that require the ``ollama`` SDK (e.g., building ``ollama.Message``
objects) remain in the model class. This module focuses on data
transformations that can be tested without SDK dependencies.

See: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from typing import Any, Literal, cast

from genkit.types import (
    GenerationCommonConfig,
    GenerationUsage,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

__all__ = [
    'build_prompt',
    'build_request_options_dict',
    'build_response_parts',
    'get_usage_info',
    'strip_data_uri_prefix',
    'to_ollama_role',
]


def to_ollama_role(role: Role) -> Literal['user', 'assistant', 'system', 'tool']:
    """Convert a Genkit Role to an Ollama role string.

    Args:
        role: Genkit Role enum.

    Returns:
        Ollama-compatible role string.

    Raises:
        ValueError: If the role is not recognized.
    """
    match role:
        case Role.USER:
            return 'user'
        case Role.MODEL:
            return 'assistant'
        case Role.TOOL:
            return 'tool'
        case Role.SYSTEM:
            return 'system'
        case _:
            raise ValueError(f'Unknown role: {role}')


def build_prompt(messages: list[Message]) -> str:
    """Build a plain-text prompt from messages for the generate API.

    Only text parts are included; non-text parts are skipped.

    Args:
        messages: List of Genkit messages.

    Returns:
        Concatenated text content.
    """
    parts: list[str] = []
    for message in messages:
        for text_part in message.content:
            if isinstance(text_part.root, TextPart):
                parts.append(text_part.root.text)
    return ''.join(parts)


def build_request_options_dict(
    config: GenerationCommonConfig | dict[str, object] | None,
) -> dict[str, Any]:
    """Build options dict from config for the Ollama API.

    Maps Genkit ``GenerationCommonConfig`` fields to Ollama option names.

    Args:
        config: Request configuration.

    Returns:
        Dict of Ollama options.
    """
    if config is None:
        return {}

    if isinstance(config, GenerationCommonConfig):
        result: dict[str, Any] = {}
        if config.top_k is not None:
            result['top_k'] = config.top_k
        if config.top_p is not None:
            result['topP'] = config.top_p
        if config.stop_sequences is not None:
            result['stop'] = config.stop_sequences
        if config.temperature is not None:
            result['temperature'] = config.temperature
        if config.max_output_tokens is not None:
            result['num_predict'] = config.max_output_tokens
        return result

    if isinstance(config, dict):
        return cast(dict[str, Any], config)

    return {}


def build_response_parts(
    content: str | None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[Part]:
    """Build Genkit Parts from Ollama response content and tool calls.

    Args:
        content: Text content from the response.
        tool_calls: Optional tool calls from the response.

    Returns:
        List of Genkit Part objects.
    """
    parts: list[Part] = []

    if content:
        parts.append(Part(root=TextPart(text=content)))

    if tool_calls:
        for tc in tool_calls:
            function = tc.get('function', {})
            parts.append(
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            name=function.get('name', ''),
                            input=function.get('arguments', {}),
                        )
                    )
                )
            )

    return parts


def get_usage_info(
    basic_usage: GenerationUsage,
    prompt_eval_count: int | None,
    eval_count: int | None,
) -> GenerationUsage:
    """Update basic usage with token counts from Ollama API response.

    Args:
        basic_usage: Base usage with character/image counts.
        prompt_eval_count: Input token count from Ollama.
        eval_count: Output token count from Ollama.

    Returns:
        Updated GenerationUsage with token counts.
    """
    basic_usage.input_tokens = prompt_eval_count or 0
    basic_usage.output_tokens = eval_count or 0
    basic_usage.total_tokens = basic_usage.input_tokens + basic_usage.output_tokens
    return basic_usage


def strip_data_uri_prefix(url: str) -> str:
    """Strip the ``data:...;base64,`` prefix from a data URI.

    The Ollama client's ``Image`` type only accepts raw base64 strings,
    not full data URIs. This extracts the base64 payload.

    Args:
        url: A data URI string.

    Returns:
        The base64 payload after the comma.

    Raises:
        ValueError: If the URL doesn't contain a comma.
    """
    comma_idx = url.index(',')
    return url[comma_idx + 1 :]
