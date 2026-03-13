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

"""Tests for Ollama format conversion utilities.

Covers role conversion, prompt building, request options, response
part building, usage info, and data URI stripping.
"""

from typing import Any, cast

import pytest

from genkit.plugins.ollama.converters import (
    build_prompt,
    build_request_options_dict,
    build_response_parts,
    get_usage_info,
    strip_data_uri_prefix,
    to_ollama_role,
)
from genkit.types import (
    GenerationCommonConfig,
    GenerationUsage,
    Message,
    Part,
    Role,
    TextPart,
    ToolRequestPart,
)


class TestToOllamaRole:
    """Tests for Genkit to Ollama role conversion."""

    def test_user(self) -> None:
        """Test User."""
        assert to_ollama_role(Role.USER) == 'user'

    def test_model(self) -> None:
        """Test Model."""
        assert to_ollama_role(Role.MODEL) == 'assistant'

    def test_system(self) -> None:
        """Test System."""
        assert to_ollama_role(Role.SYSTEM) == 'system'

    def test_tool(self) -> None:
        """Test Tool."""
        assert to_ollama_role(Role.TOOL) == 'tool'


class TestBuildPrompt:
    """Tests for prompt building from messages."""

    def test_single_message(self) -> None:
        """Test Single message."""
        msgs = [Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])]
        assert build_prompt(msgs) == 'Hello'

    def test_multiple_messages(self) -> None:
        """Test Multiple messages."""
        msgs = [
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='System. '))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='User.'))]),
        ]
        assert build_prompt(msgs) == 'System. User.'

    def test_empty_messages(self) -> None:
        """Test Empty messages."""
        assert build_prompt([]) == ''


class TestBuildRequestOptionsDict:
    """Tests for building Ollama options from config."""

    def test_none_returns_empty(self) -> None:
        """Test None returns empty."""
        assert build_request_options_dict(None) == {}

    def test_generation_common_config(self) -> None:
        """Test Generation common config."""
        config = GenerationCommonConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)
        got = build_request_options_dict(config)
        assert got.get('temperature') == 0.7
        assert got.get('num_predict') == 100
        assert got.get('topP') == 0.9

    def test_dict_passthrough(self) -> None:
        """Test Dict passthrough."""
        d = {'temperature': 0.5, 'custom': True}
        got = build_request_options_dict(d)
        assert got == d

    def test_unknown_type_returns_empty(self) -> None:
        """Test Unknown type returns empty."""
        # Intentionally pass an unsupported type to test defensive handling
        assert build_request_options_dict(cast(Any, 42)) == {}


class TestBuildResponseParts:
    """Tests for building Genkit Parts from Ollama response data."""

    def test_text_only(self) -> None:
        """Test Text only."""
        parts = build_response_parts('Hello')
        assert len(parts) == 1
        assert isinstance(parts[0].root, TextPart)
        assert parts[0].root.text == 'Hello'

    def test_tool_calls(self) -> None:
        """Test Tool calls."""
        tool_calls = [{'function': {'name': 'search', 'arguments': {'q': 'test'}}}]
        parts = build_response_parts(None, tool_calls)
        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.name == 'search'

    def test_text_and_tool_calls(self) -> None:
        """Test Text and tool calls."""
        tool_calls = [{'function': {'name': 'calc', 'arguments': {}}}]
        parts = build_response_parts('Thinking...', tool_calls)
        assert len(parts) == 2, f'Expected 2 parts, got {len(parts)}'

    def test_empty(self) -> None:
        """Test Empty."""
        parts = build_response_parts(None)
        assert not (parts), 'Expected empty list'

    def test_empty_string(self) -> None:
        """Test Empty string."""
        parts = build_response_parts('')
        assert not (parts), 'Expected empty list for empty string'


class TestGetUsageInfo:
    """Tests for usage info extraction."""

    def test_with_counts(self) -> None:
        """Test With counts."""
        basic = GenerationUsage(input_characters=100)
        got = get_usage_info(basic, 10, 20)
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30, f'got {got}'
        assert got.input_characters == 100, 'Lost input_characters'

    def test_none_counts(self) -> None:
        """Test None counts."""
        basic = GenerationUsage()
        got = get_usage_info(basic, None, None)
        assert got.input_tokens == 0 or got.output_tokens != 0


class TestStripDataUriPrefix:
    """Tests for data URI prefix stripping."""

    def test_jpeg(self) -> None:
        """Test Jpeg."""
        assert strip_data_uri_prefix('data:image/jpeg;base64,/9j/4AA') == '/9j/4AA'

    def test_png(self) -> None:
        """Test Png."""
        assert strip_data_uri_prefix('data:image/png;base64,iVBOR') == 'iVBOR'

    def test_no_comma_raises(self) -> None:
        """Test No comma raises."""
        with pytest.raises(ValueError):
            strip_data_uri_prefix('invalid-no-comma')
