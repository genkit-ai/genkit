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

"""ModelRequest rejects keywords it doesn't know instead of silently dropping them."""

import pytest
from pydantic import ValidationError

from genkit import Part
from genkit._core._model import Message, ModelRequest, OutputConfig
from genkit._core._typing import Role


def _messages() -> list[Message]:
    return [Message(role=Role.USER, content=[Part.from_text('hi')])]


def test_model_request_output_format_keyword_raises() -> None:
    """ModelRequest(..., output_format='json') raises and points at output=OutputConfig(...)."""
    with pytest.raises(ValidationError, match=r'output=OutputConfig'):
        ModelRequest(messages=_messages(), output_format='json')  # type: ignore[call-arg]


def test_model_request_output_format_camel_keyword_raises() -> None:
    """ModelRequest(..., outputFormat='json') raises the same error as the snake_case name."""
    with pytest.raises(ValidationError, match=r'output=OutputConfig'):
        ModelRequest(messages=_messages(), outputFormat='json')  # type: ignore[call-arg]


def test_model_request_misspelled_config_raises() -> None:
    """ModelRequest(..., confg={...}) raises instead of running with no config."""
    with pytest.raises(ValidationError) as exc:
        ModelRequest(messages=_messages(), confg={'temperature': 0.2})  # type: ignore[call-arg]
    assert exc.value.errors()[0]['type'] == 'extra_forbidden'
    assert exc.value.errors()[0]['loc'] == ('confg',)


def test_model_request_unknown_keyword_raises() -> None:
    """ModelRequest(..., futureField=1) raises extra_forbidden."""
    with pytest.raises(ValidationError) as exc:
        ModelRequest(messages=_messages(), futureField=1)  # type: ignore[call-arg]
    assert exc.value.errors()[0]['type'] == 'extra_forbidden'


def test_model_request_tool_choice_accepts_camel_and_snake() -> None:
    """toolChoice='required' and tool_choice='required' both land on tool_choice."""
    camel = ModelRequest(messages=_messages(), toolChoice='required')  # type: ignore[call-arg]
    snake = ModelRequest(messages=_messages(), tool_choice='required')
    assert camel.tool_choice == 'required'
    assert snake.tool_choice == 'required'


def test_model_request_output_config_sets_format() -> None:
    """output=OutputConfig(format='json') reads back as output_format == 'json'."""
    req = ModelRequest(messages=_messages(), output=OutputConfig(format='json'))
    assert req.output_format == 'json'


def test_model_request_validate_unknown_top_level_key_raises() -> None:
    """model_validate on incoming JSON with futureField raises extra_forbidden."""
    with pytest.raises(ValidationError) as exc:
        ModelRequest.model_validate({'messages': [{'role': 'user', 'content': [{'text': 'hi'}]}], 'futureField': 1})
    assert exc.value.errors()[0]['type'] == 'extra_forbidden'


def test_model_request_validate_unknown_part_field_raises() -> None:
    """model_validate rejects an unknown key on a part inside a message."""
    with pytest.raises(ValidationError):
        ModelRequest.model_validate({'messages': [{'role': 'user', 'content': [{'text': 'hi', 'futureField': 1}]}]})


def test_model_request_copy_output_format_raises() -> None:
    """model_copy(update={'output_format': 'json'}) raises and leaves the original's format alone."""
    req = ModelRequest(messages=_messages(), output=OutputConfig(format='text'))
    with pytest.raises(ValidationError, match=r'output=OutputConfig'):
        req.model_copy(update={'output_format': 'json'})
    assert req.output_format == 'text'


def test_model_request_copy_output_config_updates_format() -> None:
    """model_copy(update={'output': OutputConfig(format='json')}) reads back output_format == 'json'."""
    req = ModelRequest(messages=_messages(), output=OutputConfig(format='text'))
    copied = req.model_copy(update={'output': OutputConfig(format='json')})
    assert copied.output_format == 'json'
    assert req.output_format == 'text'
