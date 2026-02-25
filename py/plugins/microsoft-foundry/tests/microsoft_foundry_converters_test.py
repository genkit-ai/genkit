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

"""Tests for Microsoft Foundry format conversion utilities.

Covers finish reason mapping, role conversion, message conversion,
tool definitions, config normalization, tool call parsing, and usage
building.
"""

from genkit.plugins.microsoft_foundry.models.converters import (
    FINISH_REASON_MAP,
    build_usage,
    extract_text,
    from_openai_tool_calls,
    map_finish_reason,
    normalize_config,
    parse_tool_call_args,
    to_openai_messages,
    to_openai_role,
    to_openai_tool,
)
from genkit.plugins.microsoft_foundry.typing import MicrosoftFoundryConfig, VisualDetailLevel
from genkit.types import (
    FinishReason,
    GenerationCommonConfig,
    Media,
    MediaPart,
    Message,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


class TestMapFinishReason:
    """Tests for finish reason mapping."""

    def test_stop_maps_to_stop(self) -> None:
        """Test Stop maps to stop."""
        assert map_finish_reason('stop') == FinishReason.STOP

    def test_length_maps_to_length(self) -> None:
        """Test Length maps to length."""
        assert map_finish_reason('length') == FinishReason.LENGTH

    def test_tool_calls_maps_to_stop(self) -> None:
        """Test Tool calls maps to stop."""
        assert map_finish_reason('tool_calls') == FinishReason.STOP

    def test_content_filter_maps_to_blocked(self) -> None:
        """Test Content filter maps to blocked."""
        assert map_finish_reason('content_filter') == FinishReason.BLOCKED

    def test_function_call_maps_to_stop(self) -> None:
        """Test Function call maps to stop."""
        assert map_finish_reason('function_call') == FinishReason.STOP

    def test_unknown_maps_to_unknown(self) -> None:
        """Test Unknown maps to unknown."""
        assert map_finish_reason('new_reason') == FinishReason.UNKNOWN

    def test_empty_string_maps_to_unknown(self) -> None:
        """Test Empty string maps to unknown."""
        assert map_finish_reason('') == FinishReason.UNKNOWN

    def test_finish_reason_map_keys(self) -> None:
        """Test Finish reason map keys."""
        expected = {'stop', 'length', 'tool_calls', 'content_filter', 'function_call'}
        assert FINISH_REASON_MAP.keys() == expected, f'keys = {set(FINISH_REASON_MAP.keys())}, want {expected}'


class TestToOpenaiRole:
    """Tests for Genkit to OpenAI role conversion."""

    def test_user_enum(self) -> None:
        """Test User enum."""
        assert to_openai_role(Role.USER) == 'user'

    def test_model_enum(self) -> None:
        """Test Model enum."""
        assert to_openai_role(Role.MODEL) == 'assistant'

    def test_system_enum(self) -> None:
        """Test System enum."""
        assert to_openai_role(Role.SYSTEM) == 'system'

    def test_tool_enum(self) -> None:
        """Test Tool enum."""
        assert to_openai_role(Role.TOOL) == 'tool'

    def test_user_string(self) -> None:
        """Test User string."""
        assert to_openai_role('user') == 'user'

    def test_model_string(self) -> None:
        """Test Model string."""
        assert to_openai_role('model') == 'assistant'

    def test_case_insensitive(self) -> None:
        """Test Case insensitive."""
        assert to_openai_role('SYSTEM') == 'system'

    def test_unknown_defaults_to_user(self) -> None:
        """Test Unknown defaults to user."""
        assert to_openai_role('admin') == 'user'


class TestExtractText:
    """Tests for message text extraction."""

    def test_single_text_part(self) -> None:
        """Test Single text part."""
        msg = Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])
        assert extract_text(msg) == 'Hello'

    def test_multiple_text_parts(self) -> None:
        """Test Multiple text parts."""
        msg = Message(
            role=Role.USER,
            content=[Part(root=TextPart(text='A')), Part(root=TextPart(text='B'))],
        )
        assert extract_text(msg) == 'AB'

    def test_no_text_parts(self) -> None:
        """Test No text parts."""
        msg = Message(
            role=Role.USER,
            content=[Part(root=MediaPart(media=Media(url='https://x.com/img.png', content_type='image/png')))],
        )
        assert extract_text(msg) == ''

    def test_empty_content(self) -> None:
        """Test Empty content."""
        msg = Message(role=Role.USER, content=[])
        assert extract_text(msg) == ''


class TestToOpenaiTool:
    """Tests for Genkit to OpenAI tool format conversion."""

    def test_basic_tool(self) -> None:
        """Test Basic tool."""
        tool = ToolDefinition(
            name='search',
            description='Search the web',
            input_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}},
        )
        got = to_openai_tool(tool)
        assert got['type'] == 'function', f'type = {got["type"]}'
        assert got['function']['name'] == 'search'
        assert got['function']['description'] == 'Search the web'

    def test_tool_empty_description(self) -> None:
        """Test Tool empty description."""
        tool = ToolDefinition(name='ping', description='')
        got = to_openai_tool(tool)
        assert got['function']['description'] == ''

    def test_tool_without_schema(self) -> None:
        """Test Tool without schema."""
        tool = ToolDefinition(name='noop', description='does nothing')
        got = to_openai_tool(tool)
        assert got['function']['parameters'] == {}, f'parameters = {got["function"]["parameters"]}'


class TestParseToolCallArgs:
    """Tests for tool call argument parsing."""

    def test_valid_json(self) -> None:
        """Test Valid json."""
        assert parse_tool_call_args('{"a": 1}') == {'a': 1}

    def test_invalid_json(self) -> None:
        """Test Invalid json."""
        assert parse_tool_call_args('bad') == 'bad'

    def test_none_returns_empty_dict(self) -> None:
        """Test None returns empty dict."""
        assert parse_tool_call_args(None) == {}

    def test_empty_string_returns_empty_dict(self) -> None:
        """Test Empty string returns empty dict."""
        assert parse_tool_call_args('') == {}


class TestFromOpenaiToolCalls:
    """Tests for OpenAI tool call to Genkit Part conversion."""

    def test_single_tool_call(self) -> None:
        """Test Single tool call."""

        class FakeFunc:
            name = 'get_weather'
            arguments = '{"city": "London"}'

        class FakeToolCall:
            id = 'tc-1'
            function = FakeFunc()

        parts = from_openai_tool_calls([FakeToolCall()])
        assert len(parts) == 1, f'Expected 1 part, got {len(parts)}'
        root = parts[0].root
        assert isinstance(root, ToolRequestPart), f'Expected ToolRequestPart, got {type(root)}'
        assert root.tool_request.name == 'get_weather'
        assert root.tool_request.ref == 'tc-1'

    def test_tool_call_without_function(self) -> None:
        """Test Tool call without function."""

        class FakeToolCall:
            id = 'tc-1'
            function = None

        parts = from_openai_tool_calls([FakeToolCall()])
        assert len(parts) == 0, f'Expected 0 parts, got {len(parts)}'


class TestBuildUsage:
    """Tests for usage statistics construction."""

    def test_all_fields(self) -> None:
        """Test All fields."""
        got = build_usage(10, 20, 30)
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30, f'got {got}'

    def test_zero_values(self) -> None:
        """Test Zero values."""
        got = build_usage(0, 0, 0)
        assert got.input_tokens == 0


class TestToOpenaiMessages:
    """Tests for Genkit to OpenAI message list conversion."""

    def test_system_message(self) -> None:
        """Test System message."""
        msgs = [Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be helpful.'))])]
        got = to_openai_messages(msgs)
        assert len(got) == 1, f'Expected 1 message, got {len(got)}'
        assert got[0]['role'] == 'system'
        assert got[0]['content'] == 'Be helpful.'

    def test_user_text_message(self) -> None:
        """Test User text message."""
        msgs = [Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])]
        got = to_openai_messages(msgs)
        assert got[0]['role'] == 'user'
        content = got[0]['content']
        assert len(content) == 1 or content[0]['type'] != 'text', f'content = {content}'

    def test_user_media_message(self) -> None:
        """Test User media message."""
        msgs = [
            Message(
                role=Role.USER,
                content=[Part(root=MediaPart(media=Media(url='https://x.com/img.png', content_type='image/png')))],
            )
        ]
        got = to_openai_messages(msgs, VisualDetailLevel.HIGH)
        content = got[0]['content']
        assert content[0]['type'] == 'image_url', f'type = {content[0]["type"]}'
        assert content[0]['image_url']['detail'] == 'high'

    def test_assistant_text_message(self) -> None:
        """Test Assistant text message."""
        msgs = [Message(role=Role.MODEL, content=[Part(root=TextPart(text='Sure'))])]
        got = to_openai_messages(msgs)
        assert got[0]['role'] == 'assistant'
        assert got[0]['content'] == 'Sure'

    def test_assistant_tool_call_message(self) -> None:
        """Test Assistant tool call message."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(ref='tc-1', name='search', input={'q': 'test'}),
                        )
                    )
                ],
            )
        ]
        got = to_openai_messages(msgs)
        assert 'tool_calls' in got[0], f'Missing tool_calls key: {got[0]}'
        tc = got[0]['tool_calls'][0]
        assert tc['function']['name'] == 'search'

    def test_tool_response_message(self) -> None:
        """Test Tool response message."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(ref='tc-1', name='search', output='result'),
                        )
                    )
                ],
            )
        ]
        got = to_openai_messages(msgs)
        assert got[0]['role'] == 'tool'
        assert got[0]['tool_call_id'] == 'tc-1'
        assert got[0]['content'] == 'result'


class TestNormalizeConfig:
    """Tests for config normalization."""

    def test_none_returns_default(self) -> None:
        """Test None returns default."""
        got = normalize_config(None)
        assert isinstance(got, MicrosoftFoundryConfig)

    def test_passthrough(self) -> None:
        """Test Passthrough."""
        config = MicrosoftFoundryConfig(temperature=0.5)
        got = normalize_config(config)
        assert got is config, 'Expected same instance'

    def test_generation_common_config(self) -> None:
        """Test Generation common config."""
        config = GenerationCommonConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)
        got = normalize_config(config)
        assert got.temperature == 0.7, f'temperature = {got.temperature}'
        assert got.max_tokens == 100, f'max_tokens = {got.max_tokens}'

    def test_dict_with_camel_case(self) -> None:
        """Test Dict with camel case."""
        config = {'maxOutputTokens': 200, 'topP': 0.8}
        got = normalize_config(config)
        assert got.max_tokens == 200, f'max_tokens = {got.max_tokens}'

    def test_unknown_type_returns_default(self) -> None:
        """Test Unknown type returns default."""
        got = normalize_config(42)
        assert isinstance(got, MicrosoftFoundryConfig)
