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

"""Mistral AI format conversion utilities.

Pure-function helpers for converting between Genkit types and Mistral
SDK types. Extracted from the model module for independent unit testing.

SDK-dependent conversions (e.g., producing ``Tool`` or ``AssistantMessage``
objects) remain in the model class because they require the ``mistralai``
SDK. This module focuses on data transformations that can be tested
without mocking the SDK.

See: https://docs.mistral.ai/api/
"""

import json
from typing import Any, cast

from genkit.types import (
    FinishReason,
    GenerationCommonConfig,
    GenerationUsage,
    Part,
    ToolRequest,
    ToolRequestPart,
)

__all__ = [
    'CONFIG_KEYS',
    'FINISH_REASON_MAP',
    'build_tool_request_part',
    'build_usage',
    'extract_mistral_text',
    'map_finish_reason',
    'normalize_config',
    'parse_tool_call_args',
]

FINISH_REASON_MAP: dict[str, FinishReason] = {
    'stop': FinishReason.STOP,
    'length': FinishReason.LENGTH,
    'tool_calls': FinishReason.STOP,
    'model_length': FinishReason.LENGTH,
    'error': FinishReason.OTHER,
}

CONFIG_KEYS = (
    'temperature',
    'max_tokens',
    'top_p',
    'random_seed',
    'stop',
    'presence_penalty',
    'frequency_penalty',
    'safe_prompt',
)


def map_finish_reason(finish_reason: str | None) -> FinishReason:
    """Map a Mistral finish reason to a Genkit FinishReason.

    Args:
        finish_reason: The finish reason string from the Mistral response.

    Returns:
        The corresponding Genkit FinishReason, or OTHER if unmapped.
    """
    if not finish_reason:
        return FinishReason.STOP
    return FINISH_REASON_MAP.get(finish_reason, FinishReason.OTHER)


def extract_mistral_text(content: object) -> str:
    """Extract text from a Mistral delta content value.

    Handles plain strings and lists of objects with a ``text`` attribute.
    The ``ThinkChunk`` and ``TextChunk`` SDK types both expose ``.text``,
    so this function works with any such object without importing the SDK.

    Args:
        content: The delta content — may be str, object with .text,
            or a list of such items.

    Returns:
        Concatenated text extracted from the content.
    """
    if isinstance(content, str):
        return content
    if hasattr(content, 'text'):
        return str(content.text)
    if isinstance(content, list):
        return ''.join(extract_mistral_text(item) for item in content)
    return ''


def parse_tool_call_args(func_args: object) -> dict[str, Any] | str:
    """Parse tool call arguments from Mistral response.

    Args:
        func_args: The function arguments — may be str, dict, or other.

    Returns:
        Parsed dict, raw string, or empty dict.
    """
    if not func_args:
        return {}
    if isinstance(func_args, dict):
        return cast(dict[str, Any], func_args)
    if isinstance(func_args, str):
        try:
            return json.loads(func_args)
        except json.JSONDecodeError:
            return func_args
    return str(func_args)


def build_tool_request_part(
    tool_call_id: str | None,
    function_name: str,
    func_args: object,
) -> Part:
    """Build a Genkit ToolRequestPart from Mistral tool call fields.

    Args:
        tool_call_id: The tool call ID.
        function_name: The function name.
        func_args: Raw function arguments (str, dict, or other).

    Returns:
        A Genkit Part containing a ToolRequestPart.
    """
    return Part(
        root=ToolRequestPart(
            tool_request=ToolRequest(
                ref=tool_call_id or None,
                name=function_name,
                input=parse_tool_call_args(func_args),
            )
        )
    )


def build_usage(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> GenerationUsage:
    """Build GenerationUsage from Mistral token counts.

    Args:
        prompt_tokens: Input/prompt token count.
        completion_tokens: Output/completion token count.
        total_tokens: Total token count.

    Returns:
        GenerationUsage with token counts.
    """
    return GenerationUsage(
        input_tokens=prompt_tokens or 0,
        output_tokens=completion_tokens or 0,
        total_tokens=total_tokens or 0,
    )


def normalize_config(config: object) -> dict[str, Any]:
    """Normalize config to a dict suitable for the Mistral API.

    Handles ``GenerationCommonConfig`` by mapping its fields to the
    Mistral API field names. Dicts are passed through. Other types
    return an empty dict.

    Args:
        config: Request configuration.

    Returns:
        Dict of Mistral API parameters.
    """
    if config is None:
        return {}

    if isinstance(config, dict):
        mapped: dict[str, Any] = {}
        key_map: dict[str, str] = {
            'maxOutputTokens': 'max_tokens',
            'maxTokens': 'max_tokens',
            'topP': 'top_p',
            'stopSequences': 'stop',
            'randomSeed': 'random_seed',
            'presencePenalty': 'presence_penalty',
            'frequencyPenalty': 'frequency_penalty',
            'safePrompt': 'safe_prompt',
        }
        for key, value in config.items():
            str_key = str(key)
            mapped_key = key_map.get(str_key, str_key)
            mapped[mapped_key] = value
        return mapped

    if isinstance(config, GenerationCommonConfig):
        result: dict[str, Any] = {}
        if config.temperature is not None:
            result['temperature'] = config.temperature
        if config.max_output_tokens is not None:
            result['max_tokens'] = config.max_output_tokens
        if config.top_p is not None:
            result['top_p'] = config.top_p
        if config.stop_sequences is not None:
            result['stop'] = config.stop_sequences
        return result

    return {}
