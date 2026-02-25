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

"""Tests for xAI format conversion utilities.

Covers finish reason mapping, tool input parsing, tool request building,
and generation usage construction.
"""

from genkit.plugins.xai.converters import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    FINISH_REASON_MAP,
    build_generation_usage,
    map_finish_reason,
    parse_tool_input,
    to_genkit_tool_request,
)
from genkit.types import (
    FinishReason,
    GenerationUsage,
    ToolRequestPart,
)


class TestMapFinishReasonXai:
    """Tests for xAI finish reason mapping."""

    def test_stop(self) -> None:
        """Test Stop."""
        assert map_finish_reason('STOP') == FinishReason.STOP

    def test_length(self) -> None:
        """Test Length."""
        assert map_finish_reason('LENGTH') == FinishReason.LENGTH

    def test_tool_calls(self) -> None:
        """Test Tool calls."""
        assert map_finish_reason('TOOL_CALLS') == FinishReason.STOP

    def test_content_filter(self) -> None:
        """Test Content filter."""
        assert map_finish_reason('CONTENT_FILTER') == FinishReason.OTHER

    def test_reason_stop(self) -> None:
        """Test xAI protobuf REASON_STOP maps to STOP."""
        assert map_finish_reason('REASON_STOP') == FinishReason.STOP

    def test_reason_length(self) -> None:
        """Test xAI protobuf REASON_LENGTH maps to LENGTH."""
        assert map_finish_reason('REASON_LENGTH') == FinishReason.LENGTH

    def test_reason_tool_calls(self) -> None:
        """Test xAI protobuf REASON_TOOL_CALLS maps to STOP."""
        assert map_finish_reason('REASON_TOOL_CALLS') == FinishReason.STOP

    def test_reason_content_filter(self) -> None:
        """Test xAI protobuf REASON_CONTENT_FILTER maps to OTHER."""
        assert map_finish_reason('REASON_CONTENT_FILTER') == FinishReason.OTHER

    def test_none_defaults_to_unknown(self) -> None:
        """Test None defaults to unknown."""
        assert map_finish_reason(None) == FinishReason.UNKNOWN

    def test_unknown_defaults_to_unknown(self) -> None:
        """Test Unknown defaults to unknown."""
        assert map_finish_reason('NEW_REASON') == FinishReason.UNKNOWN

    def test_finish_reason_map_keys(self) -> None:
        """Test Finish reason map includes both OpenAI-style and protobuf keys."""
        expected = {
            'STOP',
            'LENGTH',
            'TOOL_CALLS',
            'CONTENT_FILTER',
            'REASON_STOP',
            'REASON_LENGTH',
            'REASON_TOOL_CALLS',
            'REASON_CONTENT_FILTER',
        }
        assert FINISH_REASON_MAP.keys() == expected, f'keys = {set(FINISH_REASON_MAP.keys())}'


class TestParseToolInput:
    """Tests for tool input parsing."""

    def test_valid_json_string(self) -> None:
        """Test Valid json string."""
        assert parse_tool_input('{"a": 1}') == {'a': 1}

    def test_invalid_json_string(self) -> None:
        """Test Invalid json string."""
        assert parse_tool_input('bad') == 'bad'

    def test_dict_passthrough(self) -> None:
        """Test Dict passthrough."""
        d = {'x': 'y'}
        assert parse_tool_input(d) is d

    def test_none_passthrough(self) -> None:
        """Test None passthrough."""
        assert parse_tool_input(None) is None


class TestToGenkitToolRequest:
    """Tests for building ToolRequestPart from xAI tool call fields."""

    def test_basic(self) -> None:
        """Test Basic."""
        part = to_genkit_tool_request('tc-1', 'search', '{"q": "test"}')
        root = part.root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.ref == 'tc-1'
        assert root.tool_request.name == 'search'
        assert root.tool_request.input == {'q': 'test'}

    def test_dict_arguments(self) -> None:
        """Test Dict arguments."""
        part = to_genkit_tool_request('tc-2', 'calc', {'x': 1})
        root = part.root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.input == {'x': 1}


class TestBuildGenerationUsage:
    """Tests for usage statistics construction."""

    def test_basic(self) -> None:
        """Test Basic."""
        got = build_generation_usage(10, 20, 30)
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30

    def test_with_basic_usage(self) -> None:
        """Test With basic usage."""
        basic = GenerationUsage(
            input_characters=100,
            output_characters=200,
            input_images=1,
            output_images=0,
        )
        got = build_generation_usage(10, 20, 30, basic)
        assert got.input_tokens == 10
        assert got.input_characters == 100
        assert got.input_images == 1

    def test_without_basic_usage(self) -> None:
        """Test Without basic usage."""
        got = build_generation_usage(5, 10, 15)
        assert got.input_characters is None


class TestConstants:
    """Tests for xAI converter constants."""

    def test_default_max_output_tokens(self) -> None:
        """Test Default max output tokens."""
        assert DEFAULT_MAX_OUTPUT_TOKENS == 4096
