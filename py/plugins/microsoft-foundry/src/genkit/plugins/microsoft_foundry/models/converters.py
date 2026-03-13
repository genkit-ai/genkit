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

"""Microsoft Foundry format conversion utilities.

Pure-function helpers for converting between Genkit types and OpenAI-compatible
chat completion API formats used by Microsoft Foundry / Azure OpenAI.
Extracted from the model module for independent unit testing.

See:
    - Microsoft Foundry: https://ai.azure.com/
    - Azure OpenAI: https://learn.microsoft.com/en-us/azure/ai-services/openai/
"""

import json
from typing import Any

from genkit.plugins.microsoft_foundry.typing import MicrosoftFoundryConfig, VisualDetailLevel
from genkit.types import (
    FinishReason,
    GenerationCommonConfig,
    GenerationUsage,
    MediaPart,
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
    'FINISH_REASON_MAP',
    'build_usage',
    'extract_text',
    'from_openai_tool_calls',
    'map_finish_reason',
    'normalize_config',
    'parse_tool_call_args',
    'to_openai_messages',
    'to_openai_role',
    'to_openai_tool',
]

# Mapping from OpenAI finish reasons to Genkit finish reasons.
FINISH_REASON_MAP: dict[str, FinishReason] = {
    'stop': FinishReason.STOP,
    'length': FinishReason.LENGTH,
    'tool_calls': FinishReason.STOP,
    'content_filter': FinishReason.BLOCKED,
    'function_call': FinishReason.STOP,
}


def map_finish_reason(finish_reason: str) -> FinishReason:
    """Map an OpenAI finish reason string to a Genkit FinishReason.

    Args:
        finish_reason: The finish reason from the API response.

    Returns:
        The corresponding Genkit FinishReason, or UNKNOWN if unmapped.
    """
    return FINISH_REASON_MAP.get(finish_reason, FinishReason.UNKNOWN)


def to_openai_role(role: Role | str) -> str:
    """Convert a Genkit role to an OpenAI role string.

    Args:
        role: Genkit Role enum or string.

    Returns:
        OpenAI role string ('user', 'assistant', 'system', 'tool').
    """
    if isinstance(role, str):
        str_role_map = {
            'user': 'user',
            'model': 'assistant',
            'system': 'system',
            'tool': 'tool',
        }
        return str_role_map.get(role.lower(), 'user')

    role_map = {
        Role.USER: 'user',
        Role.MODEL: 'assistant',
        Role.SYSTEM: 'system',
        Role.TOOL: 'tool',
    }
    return role_map.get(role, 'user')


def extract_text(msg: Message) -> str:
    """Extract text content from a message.

    Concatenates all TextPart contents from the message's parts.

    Args:
        msg: Message to extract text from.

    Returns:
        Concatenated text content.
    """
    texts: list[str] = []
    for part in msg.content:
        root = part.root if isinstance(part, Part) else part
        if isinstance(root, TextPart):
            texts.append(root.text)
    return ''.join(texts)


def to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    """Convert a Genkit tool definition to OpenAI format.

    Args:
        tool: Genkit ToolDefinition.

    Returns:
        OpenAI-compatible tool definition dict.
    """
    parameters = tool.input_schema or {}
    if parameters:
        parameters = {**parameters, 'type': 'object'}

    return {
        'type': 'function',
        'function': {
            'name': tool.name,
            'description': tool.description or '',
            'parameters': parameters,
        },
    }


def parse_tool_call_args(args: str | None) -> dict[str, Any] | str:
    """Parse tool call arguments from a JSON string.

    Gracefully handles invalid JSON by returning the raw string.

    Args:
        args: JSON string of tool call arguments, or None.

    Returns:
        Parsed dict if valid JSON, otherwise the raw string or empty dict.
    """
    if not args:
        return {}
    try:
        return json.loads(args)
    except (json.JSONDecodeError, TypeError):
        return args


def from_openai_tool_calls(tool_calls: list[Any]) -> list[Part]:
    """Convert OpenAI tool call objects to Genkit ToolRequestParts.

    Args:
        tool_calls: List of tool call objects from the OpenAI response.

    Returns:
        List of Genkit Part objects containing ToolRequestParts.
    """
    parts: list[Part] = []
    for tc in tool_calls:
        func = getattr(tc, 'function', None)
        if func is None:
            continue

        func_args = getattr(func, 'arguments', None)
        func_name = getattr(func, 'name', 'unknown')
        args = parse_tool_call_args(func_args)

        parts.append(
            Part(
                root=ToolRequestPart(
                    tool_request=ToolRequest(
                        ref=tc.id,
                        name=func_name,
                        input=args,
                    )
                )
            )
        )
    return parts


def build_usage(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> GenerationUsage:
    """Build GenerationUsage from OpenAI token counts.

    Args:
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.
        total_tokens: Total tokens.

    Returns:
        GenerationUsage with token counts.
    """
    return GenerationUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def to_openai_messages(
    messages: list[Message],
    visual_detail_level: VisualDetailLevel = VisualDetailLevel.AUTO,
) -> list[dict[str, Any]]:
    """Convert Genkit messages to OpenAI chat message format.

    Args:
        messages: List of Genkit messages.
        visual_detail_level: Detail level for image processing.

    Returns:
        List of OpenAI-compatible message dictionaries.
    """
    openai_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = to_openai_role(msg.role)

        if role == 'system':
            openai_msgs.append({'role': 'system', 'content': extract_text(msg)})

        elif role == 'user':
            content_parts: list[dict[str, Any]] = []
            for part in msg.content:
                root = part.root if isinstance(part, Part) else part
                if isinstance(root, TextPart):
                    content_parts.append({'type': 'text', 'text': root.text})
                elif isinstance(root, MediaPart):
                    content_parts.append({
                        'type': 'image_url',
                        'image_url': {
                            'url': root.media.url,
                            'detail': visual_detail_level.value,
                        },
                    })
            openai_msgs.append({'role': 'user', 'content': content_parts})

        elif role == 'assistant':
            tool_calls: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for part in msg.content:
                root = part.root if isinstance(part, Part) else part
                if isinstance(root, TextPart):
                    text_parts.append(root.text)
                elif isinstance(root, ToolRequestPart):
                    tool_calls.append({
                        'id': root.tool_request.ref or '',
                        'type': 'function',
                        'function': {
                            'name': root.tool_request.name,
                            'arguments': json.dumps(root.tool_request.input),
                        },
                    })

            if tool_calls:
                openai_msgs.append({'role': 'assistant', 'tool_calls': tool_calls})
            else:
                openai_msgs.append({'role': 'assistant', 'content': ''.join(text_parts)})

        elif role == 'tool':
            for part in msg.content:
                root = part.root if isinstance(part, Part) else part
                if isinstance(root, ToolResponsePart):
                    output = root.tool_response.output
                    content = output if isinstance(output, str) else json.dumps(output)
                    openai_msgs.append({
                        'role': 'tool',
                        'tool_call_id': root.tool_response.ref or '',
                        'content': content,
                    })

    return openai_msgs


def normalize_config(config: object) -> MicrosoftFoundryConfig:
    """Normalize config to MicrosoftFoundryConfig.

    Handles dicts with camelCase keys, GenerationCommonConfig, and
    MicrosoftFoundryConfig passthrough.

    Args:
        config: Request configuration.

    Returns:
        Normalized MicrosoftFoundryConfig instance.
    """
    if config is None:
        return MicrosoftFoundryConfig()

    if isinstance(config, MicrosoftFoundryConfig):
        return config

    if isinstance(config, GenerationCommonConfig):
        max_tokens = int(config.max_output_tokens) if config.max_output_tokens is not None else None
        return MicrosoftFoundryConfig(
            temperature=config.temperature,
            max_tokens=max_tokens,
            top_p=config.top_p,
            stop=config.stop_sequences,
        )

    if isinstance(config, dict):
        mapped: dict[str, Any] = {}
        key_map: dict[str, str] = {
            'maxOutputTokens': 'max_tokens',
            'maxTokens': 'max_tokens',
            'maxCompletionTokens': 'max_completion_tokens',
            'topP': 'top_p',
            'stopSequences': 'stop',
            'frequencyPenalty': 'frequency_penalty',
            'presencePenalty': 'presence_penalty',
            'logitBias': 'logit_bias',
            'logProbs': 'logprobs',
            'topLogProbs': 'top_logprobs',
            'visualDetailLevel': 'visual_detail_level',
            'reasoningEffort': 'reasoning_effort',
            'parallelToolCalls': 'parallel_tool_calls',
            'responseFormat': 'response_format',
        }
        for key, value in config.items():
            str_key = str(key)
            mapped_key = key_map.get(str_key, str_key)
            mapped[mapped_key] = value
        return MicrosoftFoundryConfig(**mapped)

    return MicrosoftFoundryConfig()
