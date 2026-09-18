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

"""Normalize Genkit messages to the OpenTelemetry GenAI content schema.

Pure mapping, no OpenTelemetry dependency. Only used when content
capture is on — that's the leftover a later reader joins to when they
opt into seeing the prompt and reply.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel

from genkit._core._typing import Part, Role
from genkit.model import Message, ModelResponse


class NormalizedMessages(NamedTuple):
    """Conversation messages plus extracted system instructions."""

    messages: list[dict[str, Any]]
    system_instructions: list[dict[str, Any]]


def map_role(role: Role | str) -> str:
    """Map a Genkit role to the GenAI role name.

    Genkit ``model`` becomes ``assistant``; other roles pass through.
    """
    value = role.value if isinstance(role, Role) else str(role)
    if value == 'model':
        return 'assistant'
    return value


def part_json(part: object) -> dict[str, Any]:
    """JSON shape of a Part, discriminated the way the wire does."""
    root = part.root if isinstance(part, Part) else part
    if isinstance(root, BaseModel):
        return root.model_dump(by_alias=True, exclude_none=True)
    if isinstance(root, Mapping):
        return {str(key): value for key, value in root.items()}
    return {}


def map_part(part: object) -> dict[str, Any]:
    """Convert a single Genkit part to a GenAI content part map.

    Unknown part kinds fall back to a generic ``text`` part whose
    content is JSON, so nothing is silently dropped from captured
    content.
    """
    payload = part_json(part)

    text = payload.get('text')
    if text is not None:
        return {'type': 'text', 'content': text}

    reasoning = payload.get('reasoning')
    if reasoning is not None:
        return {'type': 'reasoning', 'content': reasoning}

    tool_request = payload.get('toolRequest')
    if isinstance(tool_request, Mapping):
        mapped: dict[str, Any] = {
            'type': 'tool_call',
            'name': tool_request.get('name'),
            'arguments': tool_request.get('input'),
        }
        if tool_request.get('ref') is not None:
            mapped['id'] = tool_request['ref']
        return mapped

    tool_response = payload.get('toolResponse')
    if isinstance(tool_response, Mapping):
        mapped = {
            'type': 'tool_call_response',
            'response': tool_response.get('output'),
        }
        if tool_response.get('ref') is not None:
            mapped['id'] = tool_response['ref']
        return mapped

    media = payload.get('media')
    if isinstance(media, Mapping):
        mapped = {
            'type': 'media',
            'content': media.get('url'),
        }
        if media.get('contentType') is not None:
            mapped['content_type'] = media['contentType']
        return mapped

    return {'type': 'text', 'content': json.dumps(payload)}


def is_tool_request_part(part: object) -> bool:
    """True when ``part`` is a tool-request part."""
    return isinstance(part_json(part).get('toolRequest'), Mapping)


def map_message(message: Message) -> dict[str, Any]:
    """Convert a single Genkit message to a GenAI message map."""
    return {
        'role': map_role(message.role),
        'parts': [map_part(part) for part in message.content],
    }


def normalize_messages(messages: Sequence[Message]) -> NormalizedMessages:
    """Split system instructions out of the conversation leftover.

    System messages become ``gen_ai.system_instructions`` (their parts
    directly). Everything else stays in ``gen_ai.input.messages``.
    """
    system: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for message in messages:
        role = message.role.value if isinstance(message.role, Role) else str(message.role)
        if role == 'system':
            system.extend(map_part(part) for part in message.content)
        else:
            rest.append(map_message(message))
    return NormalizedMessages(messages=rest, system_instructions=system)


def map_output_message(message: Message, finish_reason: str) -> dict[str, Any]:
    """Map a response message and attach the mapped finish reason."""
    return {**map_message(message), 'finish_reason': finish_reason}


def resolve_response_message(response: ModelResponse) -> Message | None:
    """Resolve the effective response message.

    Prefers the top-level ``message``. Falls back to
    ``candidates[0].message`` so a leftover that only arrived in the
    older candidate shape still gets captured. Missing both → ``None``.
    """
    if response.message is not None:
        return response.message
    candidates = response.candidates
    if not candidates:
        return None
    first = candidates[0]
    message = first.message
    if isinstance(message, Message):
        return message
    if isinstance(message, Mapping):
        return Message(message)
    if message is not None:
        return Message(message)
    return None
