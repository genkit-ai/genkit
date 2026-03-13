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

"""xAI format conversion utilities.

Pure-function helpers for converting between Genkit types and xAI API
types. Protobuf-specific conversions remain in the model module; this
file provides the logic that can be tested without SDK dependencies.

See: https://docs.x.ai/docs/api-reference
"""

import json
from typing import Any

from genkit.types import (
    FinishReason,
    GenerationUsage,
    Part,
    ToolRequest,
    ToolRequestPart,
)

__all__ = [
    'DEFAULT_MAX_OUTPUT_TOKENS',
    'FINISH_REASON_MAP',
    'build_generation_usage',
    'map_finish_reason',
    'parse_tool_input',
    'to_genkit_tool_request',
]

DEFAULT_MAX_OUTPUT_TOKENS = 4096

FINISH_REASON_MAP: dict[str, FinishReason] = {
    # OpenAI-style finish reasons (used by some SDK versions).
    'STOP': FinishReason.STOP,
    'LENGTH': FinishReason.LENGTH,
    'TOOL_CALLS': FinishReason.STOP,
    'CONTENT_FILTER': FinishReason.OTHER,
    # xAI protobuf enum names (REASON_-prefixed).
    'REASON_STOP': FinishReason.STOP,
    'REASON_LENGTH': FinishReason.LENGTH,
    'REASON_TOOL_CALLS': FinishReason.STOP,
    'REASON_CONTENT_FILTER': FinishReason.OTHER,
}


def map_finish_reason(reason: str | None) -> FinishReason:
    """Map an xAI finish reason string to a Genkit FinishReason.

    Args:
        reason: The finish reason string from the xAI response.

    Returns:
        The corresponding Genkit FinishReason.
    """
    if not reason:
        return FinishReason.UNKNOWN
    return FINISH_REASON_MAP.get(reason, FinishReason.UNKNOWN)


def parse_tool_input(arguments: object) -> Any:  # noqa: ANN401
    """Parse tool call arguments from an xAI response.

    Args:
        arguments: The function arguments — may be a JSON string or a dict.

    Returns:
        Parsed dict or original value if parsing fails.
    """
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return arguments
    return arguments


def to_genkit_tool_request(
    tool_call_id: str | None,
    function_name: str,
    arguments: object,
) -> Part:
    """Build a Genkit ToolRequestPart from xAI tool call fields.

    Args:
        tool_call_id: The tool call ID.
        function_name: The function name.
        arguments: Raw function arguments (str or dict).

    Returns:
        A Genkit Part containing a ToolRequestPart.
    """
    return Part(
        root=ToolRequestPart(
            tool_request=ToolRequest(
                ref=tool_call_id,
                name=function_name,
                input=parse_tool_input(arguments),
            )
        )
    )


def build_generation_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    basic_usage: GenerationUsage | None = None,
) -> GenerationUsage:
    """Build GenerationUsage from xAI token counts.

    Merges token counts with any character/image counts from a basic
    usage object.

    Args:
        prompt_tokens: Input/prompt token count.
        completion_tokens: Output/completion token count.
        total_tokens: Total token count.
        basic_usage: Optional basic usage with character/image counts.

    Returns:
        Combined GenerationUsage.
    """
    return GenerationUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        input_characters=basic_usage.input_characters if basic_usage else None,
        output_characters=basic_usage.output_characters if basic_usage else None,
        input_images=basic_usage.input_images if basic_usage else None,
        output_images=basic_usage.output_images if basic_usage else None,
    )
