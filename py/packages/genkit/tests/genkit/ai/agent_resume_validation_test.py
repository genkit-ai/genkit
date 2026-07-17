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

"""Tests for ``validate_resume_against_history``.

A resumed agent turn answers tool requests the model actually made. These tests
pin the guardrail: every ``respond``/``restart`` must reference a tool request
recorded in history, restart inputs must match the interrupted request exactly
(anti-forgery), and only ``model`` messages count as the source of truth.
"""

from __future__ import annotations

import pytest

from genkit._ai._agents._base import validate_resume_against_history
from genkit._core._error import GenkitError
from genkit._core._typing import (
    MessageData,
    Part,
    Resume,
    Role,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


def model_message_with_tools(*requests: ToolRequest) -> MessageData:
    return MessageData(
        role=Role.MODEL,
        content=[Part(root=ToolRequestPart(tool_request=tr)) for tr in requests],
    )


def restart(name: str, *, ref: str | None = None, input: object = None) -> ToolRequestPart:
    return ToolRequestPart(tool_request=ToolRequest(name=name, ref=ref, input=input))


def respond(name: str, *, ref: str | None = None) -> ToolResponsePart:
    return ToolResponsePart(tool_response=ToolResponse(name=name, ref=ref))


def test_valid_respond_passes() -> None:
    history = [model_message_with_tools(ToolRequest(name='get_weather', ref='1', input={'city': 'sf'}))]
    validate_resume_against_history(Resume(respond=[respond('get_weather', ref='1')]), history)


def test_valid_restart_with_matching_input_passes() -> None:
    history = [model_message_with_tools(ToolRequest(name='book', ref='1', input={'seat': '3A'}))]
    validate_resume_against_history(Resume(restart=[restart('book', ref='1', input={'seat': '3A'})]), history)


def test_empty_resume_passes() -> None:
    validate_resume_against_history(Resume(), [])


def test_respond_unknown_tool_raises() -> None:
    history = [model_message_with_tools(ToolRequest(name='get_weather', ref='1'))]
    with pytest.raises(GenkitError) as exc:
        validate_resume_against_history(Resume(respond=[respond('nope', ref='1')]), history)
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert 'not found' in str(exc.value).lower()


def test_restart_unknown_tool_raises() -> None:
    history = [model_message_with_tools(ToolRequest(name='book', ref='1', input={'seat': '3A'}))]
    with pytest.raises(GenkitError) as exc:
        validate_resume_against_history(Resume(restart=[restart('other', ref='1', input={'seat': '3A'})]), history)
    assert exc.value.status == 'INVALID_ARGUMENT'


def test_restart_with_tampered_input_raises() -> None:
    history = [model_message_with_tools(ToolRequest(name='book', ref='1', input={'seat': '3A'}))]
    with pytest.raises(GenkitError) as exc:
        validate_resume_against_history(Resume(restart=[restart('book', ref='1', input={'seat': '1F'})]), history)
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert 'modified inputs' in str(exc.value).lower()


def test_restart_input_match_is_order_insensitive() -> None:
    history = [model_message_with_tools(ToolRequest(name='book', ref='1', input={'a': 1, 'b': 2}))]
    # Same dict, keys in a different order — must still count as unchanged.
    validate_resume_against_history(Resume(restart=[restart('book', ref='1', input={'b': 2, 'a': 1})]), history)


def test_searches_entire_history_not_just_last_message() -> None:
    history = [
        model_message_with_tools(ToolRequest(name='get_weather', ref='1', input={'city': 'sf'})),
        MessageData(role=Role.USER, content=[]),
        model_message_with_tools(ToolRequest(name='book', ref='2', input={'seat': '3A'})),
    ]
    validate_resume_against_history(Resume(respond=[respond('get_weather', ref='1')]), history)


def test_ref_mismatch_raises() -> None:
    history = [model_message_with_tools(ToolRequest(name='book', ref='1', input={}))]
    with pytest.raises(GenkitError) as exc:
        validate_resume_against_history(Resume(respond=[respond('book', ref='2')]), history)
    assert exc.value.status == 'INVALID_ARGUMENT'


def test_tool_request_in_non_model_message_does_not_count() -> None:
    # A tool request only counts if the *model* asked for it; a matching name in a
    # user message must not satisfy the resume.
    history = [
        MessageData(
            role=Role.USER,
            content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='book', ref='1', input={})))],
        )
    ]
    with pytest.raises(GenkitError) as exc:
        validate_resume_against_history(Resume(respond=[respond('book', ref='1')]), history)
    assert exc.value.status == 'INVALID_ARGUMENT'
