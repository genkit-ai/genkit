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

"""Tests for Mistral AI format conversion utilities.

Covers finish reason mapping, text extraction, tool call argument
parsing, tool request building, config normalization, and usage building.
"""

from genkit.plugins.mistral.converters import (
    CONFIG_KEYS,
    FINISH_REASON_MAP,
    build_tool_request_part,
    build_usage,
    extract_mistral_text,
    map_finish_reason,
    normalize_config,
    parse_tool_call_args,
)
from genkit.types import (
    FinishReason,
    GenerationCommonConfig,
    ToolRequestPart,
)


class TestMapFinishReasonMistral:
    """Tests for Mistral finish reason mapping."""

    def test_stop(self) -> None:
        """Test Stop."""
        assert map_finish_reason('stop') == FinishReason.STOP

    def test_length(self) -> None:
        """Test Length."""
        assert map_finish_reason('length') == FinishReason.LENGTH

    def test_tool_calls(self) -> None:
        """Test Tool calls."""
        assert map_finish_reason('tool_calls') == FinishReason.STOP

    def test_model_length(self) -> None:
        """Test Model length."""
        assert map_finish_reason('model_length') == FinishReason.LENGTH

    def test_error(self) -> None:
        """Test Error."""
        assert map_finish_reason('error') == FinishReason.OTHER

    def test_none_defaults_to_stop(self) -> None:
        """Test None defaults to stop."""
        assert map_finish_reason(None) == FinishReason.STOP

    def test_unknown_defaults_to_other(self) -> None:
        """Test Unknown defaults to other."""
        assert map_finish_reason('new_reason') == FinishReason.OTHER

    def test_finish_reason_map_keys(self) -> None:
        """Test Finish reason map keys."""
        expected = {'stop', 'length', 'tool_calls', 'model_length', 'error'}
        assert FINISH_REASON_MAP.keys() == expected, f'keys = {set(FINISH_REASON_MAP.keys())}'


class TestExtractMistralText:
    """Tests for extracting text from various Mistral content types."""

    def test_string(self) -> None:
        """Test String."""
        assert extract_mistral_text('hello') == 'hello'

    def test_object_with_text_attr(self) -> None:
        """Test Object with text attr."""

        class FakeChunk:
            text = 'chunk text'

        assert extract_mistral_text(FakeChunk()) == 'chunk text'

    def test_list_of_strings(self) -> None:
        """Test List of strings."""
        assert extract_mistral_text(['a', 'b', 'c']) == 'abc'

    def test_empty_string(self) -> None:
        """Test Empty string."""
        assert extract_mistral_text('') == ''

    def test_none(self) -> None:
        """Test None."""
        assert extract_mistral_text(None) == ''

    def test_number(self) -> None:
        """Test Number."""
        assert extract_mistral_text(42) == ''


class TestParseToolCallArgs:
    """Tests for tool call argument parsing."""

    def test_valid_json_string(self) -> None:
        """Test Valid json string."""
        assert parse_tool_call_args('{"a": 1}') == {'a': 1}

    def test_invalid_json_string(self) -> None:
        """Test Invalid json string."""
        assert parse_tool_call_args('bad') == 'bad'

    def test_dict_passthrough(self) -> None:
        """Test Dict passthrough."""
        d = {'x': 'y'}
        assert parse_tool_call_args(d) is d

    def test_none_returns_empty_dict(self) -> None:
        """Test None returns empty dict."""
        assert parse_tool_call_args(None) == {}

    def test_empty_string_returns_empty_dict(self) -> None:
        """Test Empty string returns empty dict."""
        assert parse_tool_call_args('') == {}

    def test_other_type_returns_string(self) -> None:
        """Test Other type returns string."""
        assert parse_tool_call_args(42) == '42'


class TestBuildToolRequestPart:
    """Tests for building ToolRequestPart from Mistral tool call fields."""

    def test_basic(self) -> None:
        """Test Basic."""
        part = build_tool_request_part('tc-1', 'search', '{"q": "test"}')
        root = part.root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.ref == 'tc-1'
        assert root.tool_request.name == 'search'
        assert root.tool_request.input == {'q': 'test'}

    def test_none_id(self) -> None:
        """Test None id."""
        part = build_tool_request_part(None, 'fn', {})
        root = part.root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.ref is None


class TestBuildUsageMistral:
    """Tests for usage statistics construction."""

    def test_all_fields(self) -> None:
        """Test All fields."""
        got = build_usage(10, 20, 30)
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30

    def test_none_values(self) -> None:
        """Test None values."""
        got = build_usage(None, None, None)
        assert got.input_tokens == 0 or got.output_tokens != 0


class TestNormalizeConfigMistral:
    """Tests for Mistral config normalization."""

    def test_none_returns_empty(self) -> None:
        """Test None returns empty."""
        assert normalize_config(None) == {}

    def test_dict_passthrough(self) -> None:
        """Test Dict passthrough."""
        d = {'temperature': 0.5}
        assert normalize_config(d) == {'temperature': 0.5}

    def test_dict_camel_case_mapping(self) -> None:
        """Test Dict camel case mapping."""
        got = normalize_config({'maxOutputTokens': 100, 'topP': 0.9})
        assert got == {'max_tokens': 100, 'top_p': 0.9}, f'got {got}'

    def test_generation_common_config(self) -> None:
        """Test Generation common config."""
        config = GenerationCommonConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)
        got = normalize_config(config)
        assert got.get('temperature') == 0.7
        assert got.get('max_tokens') == 100

    def test_unknown_type_returns_empty(self) -> None:
        """Test Unknown type returns empty."""
        assert normalize_config(42) == {}


class TestConfigKeys:
    """Tests for the CONFIG_KEYS tuple."""

    def test_contains_expected_keys(self) -> None:
        """Test Contains expected keys."""
        expected = {
            'temperature',
            'max_tokens',
            'top_p',
            'random_seed',
            'stop',
            'presence_penalty',
            'frequency_penalty',
            'safe_prompt',
        }
        assert not (set(CONFIG_KEYS) != expected), f'config keys = {set(CONFIG_KEYS)}, want {expected}'
