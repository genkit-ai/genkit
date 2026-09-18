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

from __future__ import annotations

import json

from genkit_otel._gen_ai_message_mapping import (
    map_message,
    map_output_message,
    map_part,
    map_role,
    normalize_messages,
    resolve_response_message,
)

from genkit import FinishReason
from genkit._core._typing import (
    Candidate,
    CustomPart,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.model import Message, ModelResponse


def test_map_role_remaps_model_to_assistant() -> None:
    assert map_role(Role.MODEL) == 'assistant'


def test_map_role_passes_through_others() -> None:
    assert map_role(Role.USER) == 'user'
    assert map_role(Role.TOOL) == 'tool'
    assert map_role(Role.SYSTEM) == 'system'


def test_map_text_part() -> None:
    assert map_part(Part(root=TextPart(text='hi'))) == {'type': 'text', 'content': 'hi'}


def test_map_tool_request_part() -> None:
    part = Part(
        root=ToolRequestPart(
            tool_request=ToolRequest(ref='call_1', name='weather', input={'city': 'sf'}),
        )
    )
    assert map_part(part) == {
        'type': 'tool_call',
        'id': 'call_1',
        'name': 'weather',
        'arguments': {'city': 'sf'},
    }


def test_map_tool_response_part() -> None:
    part = Part(
        root=ToolResponsePart(
            tool_response=ToolResponse(ref='call_1', name='weather', output={'temp': 20}),
        )
    )
    assert map_part(part) == {
        'type': 'tool_call_response',
        'id': 'call_1',
        'response': {'temp': 20},
    }


def test_normalize_splits_system_instructions() -> None:
    result = normalize_messages([
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be helpful'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hello'))]),
    ])
    assert result.system_instructions == [{'type': 'text', 'content': 'Be helpful'}]
    assert result.messages == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'Hi'}]},
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Hello'}]},
    ]


def test_map_output_message_attaches_finish_reason() -> None:
    message = Message(role=Role.MODEL, content=[Part(root=TextPart(text='Done'))])
    result = map_output_message(message, 'stop')
    assert result['role'] == 'assistant'
    assert result['finish_reason'] == 'stop'
    assert result['parts'] == [{'type': 'text', 'content': 'Done'}]


def test_map_part_opaque_fallback_is_json() -> None:
    part = Part(root=CustomPart(custom={'foo': 'bar', 'n': 1}))
    mapped = map_part(part)
    assert mapped['type'] == 'text'
    assert json.loads(mapped['content']) == {'custom': {'foo': 'bar', 'n': 1}}


def test_resolve_prefers_top_level_message() -> None:
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='top'))]),
    )
    message = resolve_response_message(response)
    assert message is not None
    assert map_message(message)['parts'] == [{'type': 'text', 'content': 'top'}]


def test_resolve_falls_back_to_candidates() -> None:
    response = ModelResponse(
        finish_reason=FinishReason.STOP,
        candidates=[
            Candidate(
                index=0,
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='from candidate'))]),
            )
        ],
    )
    assert response.message is None
    message = resolve_response_message(response)
    assert message is not None
    assert map_message(message)['parts'] == [{'type': 'text', 'content': 'from candidate'}]


def test_resolve_returns_none_when_neither_present() -> None:
    response = ModelResponse(finish_reason=FinishReason.STOP)
    assert resolve_response_message(response) is None
