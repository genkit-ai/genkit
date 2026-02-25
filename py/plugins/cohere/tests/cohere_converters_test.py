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

"""Tests for Cohere converters — message, response, tool, and streaming helpers."""

from types import SimpleNamespace

from cohere.types import (
    AssistantChatMessageV2,
    SystemChatMessageV2,
    ToolChatMessageV2,
    Usage,
    UsageBilledUnits,
    UsageTokens,
    UserChatMessageV2,
)

from genkit.core.typing import (
    FinishReason,
    Message,
    OutputConfig,
    Part,
    Role,
    TextPart,
    ToolDefinition,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)
from genkit.plugins.cohere.converters import (
    FINISH_REASON_MAP,
    convert_messages,
    convert_tools,
    convert_usage,
    extract_content_delta_text,
    extract_finish_reason,
    extract_tool_call_delta_args,
    extract_tool_call_start,
    get_response_format,
    parse_tool_arguments,
)


class TestConvertMessages:
    """Tests for convert_messages."""

    def test_user_message(self) -> None:
        """Test user message."""
        msgs = [Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], UserChatMessageV2)
        assert result[0].content == 'Hi'

    def test_system_message(self) -> None:
        """Test system message."""
        msgs = [Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be helpful'))])]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], SystemChatMessageV2)
        assert result[0].content == 'Be helpful'

    def test_model_text_message(self) -> None:
        """Test model text message."""
        msgs = [Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hello!'))])]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], AssistantChatMessageV2)
        assert result[0].content == 'Hello!'

    def test_model_message_with_tool_calls(self) -> None:
        """Test model message with tool calls."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref='call-1',
                                name='get_weather',
                                input={'location': 'NYC'},
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], AssistantChatMessageV2)
        assert result[0].tool_calls is not None
        assert len(result[0].tool_calls) == 1
        tc_func = result[0].tool_calls[0].function
        assert tc_func is not None
        assert tc_func.name == 'get_weather'

    def test_model_message_with_text_and_tool_calls(self) -> None:
        """Test model message with text and tool calls."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(root=TextPart(text='Calling tool...')),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref='call-1',
                                name='lookup',
                                input={'q': 'test'},
                            )
                        )
                    ),
                ],
            )
        ]
        result = convert_messages(msgs)
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, AssistantChatMessageV2)
        assert msg.content == 'Calling tool...'
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_response_message(self) -> None:
        """Test tool response message."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(
                                ref='call-1',
                                name='get_weather',
                                output={'temp': 72},
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert len(result) == 1
        assert isinstance(result[0], ToolChatMessageV2)
        assert result[0].tool_call_id == 'call-1'

    def test_tool_response_with_string_output(self) -> None:
        """Test tool response with string output."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(
                                ref='call-2',
                                name='search',
                                output='result text',
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert isinstance(result[0], ToolChatMessageV2)
        assert result[0].content == 'result text'

    def test_tool_response_with_none_output(self) -> None:
        """Test tool response with none output."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(
                                ref='call-3',
                                name='noop',
                                output=None,
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert isinstance(result[0], ToolChatMessageV2)
        assert result[0].content == ''

    def test_multiple_tool_responses_expand(self) -> None:
        """Test multiple tool responses expand."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(root=ToolResponsePart(tool_response=ToolResponse(ref='c1', name='a', output='r1'))),
                    Part(root=ToolResponsePart(tool_response=ToolResponse(ref='c2', name='b', output='r2'))),
                ],
            )
        ]
        result = convert_messages(msgs)
        assert len(result) == 2
        assert all(isinstance(m, ToolChatMessageV2) for m in result)

    def test_multi_text_parts_joined(self) -> None:
        """Test multi text parts joined."""
        msgs = [
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='Part one')),
                    Part(root=TextPart(text='Part two')),
                ],
            )
        ]
        result = convert_messages(msgs)
        assert result[0].content == 'Part one\nPart two'

    def test_empty_messages_list(self) -> None:
        """Test empty messages list."""
        assert convert_messages([]) == []

    def test_empty_content_message(self) -> None:
        """Test empty content message."""
        msgs = [Message(role=Role.USER, content=[])]
        result = convert_messages(msgs)
        assert isinstance(result[0], UserChatMessageV2)
        assert result[0].content == ''

    def test_tool_request_with_none_input(self) -> None:
        """Test tool request with none input."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref='call-x',
                                name='no_args_tool',
                                input=None,
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert isinstance(result[0], AssistantChatMessageV2)
        assert result[0].tool_calls is not None
        tc_func = result[0].tool_calls[0].function
        assert tc_func is not None
        assert tc_func.arguments == '{}'

    def test_tool_request_with_none_ref(self) -> None:
        """Test tool request with none ref."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(
                                ref=None,
                                name='tool_name',
                                input={'a': 1},
                            )
                        )
                    )
                ],
            )
        ]
        result = convert_messages(msgs)
        assert isinstance(result[0], AssistantChatMessageV2)
        assert result[0].tool_calls is not None
        tc = result[0].tool_calls[0]
        assert tc.id == ''


class TestParseToolArguments:
    """Tests for parse_tool_arguments."""

    def test_valid_json_string(self) -> None:
        """Test valid json string."""
        assert parse_tool_arguments('{"a": 1}') == {'a': 1}

    def test_invalid_json_string(self) -> None:
        """Test invalid json string."""
        assert parse_tool_arguments('not json') == 'not json'

    def test_dict_passthrough(self) -> None:
        """Test dict passthrough."""
        d = {'key': 'value'}
        assert parse_tool_arguments(d) == d

    def test_other_type_stringified(self) -> None:
        """Test other type stringified."""
        assert parse_tool_arguments(42) == '42'

    def test_empty_json_string(self) -> None:
        """Test empty json string."""
        assert parse_tool_arguments('{}') == {}

    def test_json_array_string(self) -> None:
        """Test json array string."""
        assert parse_tool_arguments('[1, 2]') == [1, 2]

    def test_none_stringified(self) -> None:
        """Test none stringified."""
        assert parse_tool_arguments(None) == 'None'

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert parse_tool_arguments('') == ''


class TestConvertUsage:
    """Tests for convert_usage."""

    def test_billed_units_only(self) -> None:
        """Test billed units only."""
        usage = Usage(
            billed_units=UsageBilledUnits(input_tokens=10, output_tokens=20),
            tokens=None,
        )
        result = convert_usage(usage)
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.total_tokens == 30

    def test_tokens_only(self) -> None:
        """Test tokens only."""
        usage = Usage(
            billed_units=None,
            tokens=UsageTokens(input_tokens=5, output_tokens=15),
        )
        result = convert_usage(usage)
        assert result.input_tokens == 5
        assert result.output_tokens == 15
        assert result.total_tokens == 20

    def test_billed_takes_precedence(self) -> None:
        """Test billed takes precedence."""
        usage = Usage(
            billed_units=UsageBilledUnits(input_tokens=100, output_tokens=200),
            tokens=UsageTokens(input_tokens=5, output_tokens=15),
        )
        result = convert_usage(usage)
        assert result.input_tokens == 100
        assert result.output_tokens == 200

    def test_no_counts_returns_none_total(self) -> None:
        """Test no counts returns none total."""
        usage = Usage(
            billed_units=None,
            tokens=None,
        )
        result = convert_usage(usage)
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None

    def test_partial_billed_fills_from_tokens(self) -> None:
        """Test partial billed fills from tokens."""
        usage = Usage(
            billed_units=UsageBilledUnits(input_tokens=10, output_tokens=None),
            tokens=UsageTokens(input_tokens=5, output_tokens=25),
        )
        result = convert_usage(usage)
        assert result.input_tokens == 10
        assert result.output_tokens == 25
        assert result.total_tokens == 35

    def test_zero_tokens(self) -> None:
        """Test zero tokens."""
        usage = Usage(
            billed_units=UsageBilledUnits(input_tokens=0, output_tokens=0),
            tokens=None,
        )
        result = convert_usage(usage)
        # 0 is falsy, so billed_units with 0 fall through.
        assert result.total_tokens is None


class TestConvertTools:
    """Tests for convert_tools."""

    def test_single_tool(self) -> None:
        """Test single tool."""
        tools = [
            ToolDefinition(
                name='get_weather',
                description='Get weather',
                input_schema={'type': 'object', 'properties': {'loc': {'type': 'string'}}},
            )
        ]
        result = convert_tools(tools)
        assert len(result) == 1
        assert result[0].type == 'function'
        func = result[0].function
        assert func is not None
        assert func.name == 'get_weather'
        assert func.description == 'Get weather'

    def test_tool_with_no_description(self) -> None:
        """Test tool with no description."""
        tools = [ToolDefinition(name='noop', description='', input_schema=None)]
        result = convert_tools(tools)
        func = result[0].function
        assert func is not None
        assert func.description == ''
        assert func.parameters == {}

    def test_empty_tools_list(self) -> None:
        """Test empty tools list."""
        assert convert_tools([]) == []

    def test_multiple_tools(self) -> None:
        """Test multiple tools."""
        tools = [
            ToolDefinition(name='a', description='desc_a', input_schema={'type': 'object'}),
            ToolDefinition(name='b', description='desc_b', input_schema={'type': 'object'}),
        ]
        result = convert_tools(tools)
        assert len(result) == 2
        names = set()
        for t in result:
            assert t.function is not None
            names.add(t.function.name)
        assert names == {'a', 'b'}


class TestGetResponseFormat:
    """Tests for get_response_format."""

    def test_json_with_schema(self) -> None:
        """Test json with schema."""
        output = OutputConfig(
            format='json',
            schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
        )
        result = get_response_format(output)
        assert result == {
            'type': 'json_object',
            'json_schema': {'type': 'object', 'properties': {'name': {'type': 'string'}}},
        }

    def test_json_without_schema(self) -> None:
        """Test json without schema."""
        output = OutputConfig(format='json')
        result = get_response_format(output)
        assert result == {'type': 'json_object'}

    def test_text_format_returns_none(self) -> None:
        """Test text format returns none."""
        output = OutputConfig(format='text')
        assert get_response_format(output) is None

    def test_no_format_returns_none(self) -> None:
        """Test no format returns none."""
        output = OutputConfig()
        assert get_response_format(output) is None


class TestFinishReasonMap:
    """Tests for the finish reason mapping."""

    def test_complete_maps_to_stop(self) -> None:
        """Test complete maps to stop."""
        assert FINISH_REASON_MAP['COMPLETE'] == FinishReason.STOP

    def test_stop_sequence_maps_to_stop(self) -> None:
        """Test stop sequence maps to stop."""
        assert FINISH_REASON_MAP['STOP_SEQUENCE'] == FinishReason.STOP

    def test_max_tokens_maps_to_length(self) -> None:
        """Test max tokens maps to length."""
        assert FINISH_REASON_MAP['MAX_TOKENS'] == FinishReason.LENGTH

    def test_error_maps_to_other(self) -> None:
        """Test error maps to other."""
        assert FINISH_REASON_MAP['ERROR'] == FinishReason.OTHER

    def test_error_toxic_maps_to_blocked(self) -> None:
        """Test error toxic maps to blocked."""
        assert FINISH_REASON_MAP['ERROR_TOXIC'] == FinishReason.BLOCKED

    def test_error_limit_maps_to_length(self) -> None:
        """Test error limit maps to length."""
        assert FINISH_REASON_MAP['ERROR_LIMIT'] == FinishReason.LENGTH

    def test_tool_call_maps_to_stop(self) -> None:
        """Test tool call maps to stop."""
        assert FINISH_REASON_MAP['TOOL_CALL'] == FinishReason.STOP

    def test_unknown_reason_not_in_map(self) -> None:
        """Test unknown reason not in map."""
        assert 'UNKNOWN_REASON' not in FINISH_REASON_MAP


class TestExtractContentDeltaText:
    """Tests for extract_content_delta_text."""

    def test_full_path(self) -> None:
        """Test full path."""
        event = SimpleNamespace(delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text='hello'))))
        assert extract_content_delta_text(event) == 'hello'

    def test_missing_delta(self) -> None:
        """Test missing delta."""
        event = SimpleNamespace()
        assert extract_content_delta_text(event) == ''

    def test_missing_message(self) -> None:
        """Test missing message."""
        event = SimpleNamespace(delta=SimpleNamespace())
        assert extract_content_delta_text(event) == ''

    def test_missing_content(self) -> None:
        """Test missing content."""
        event = SimpleNamespace(delta=SimpleNamespace(message=SimpleNamespace()))
        assert extract_content_delta_text(event) == ''

    def test_none_text(self) -> None:
        """Test none text."""
        event = SimpleNamespace(delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text=None))))
        assert extract_content_delta_text(event) == ''

    def test_empty_text(self) -> None:
        """Test empty text."""
        event = SimpleNamespace(delta=SimpleNamespace(message=SimpleNamespace(content=SimpleNamespace(text=''))))
        assert extract_content_delta_text(event) == ''


class TestExtractToolCallStart:
    """Tests for extract_tool_call_start."""

    def test_full_path(self) -> None:
        """Test full path."""
        event = SimpleNamespace(
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id='tc-1',
                        function=SimpleNamespace(name='get_weather'),
                    )
                )
            )
        )
        assert extract_tool_call_start(event) == ('tc-1', 'get_weather')

    def test_missing_delta(self) -> None:
        """Test missing delta."""
        event = SimpleNamespace()
        assert extract_tool_call_start(event) == ('', '')

    def test_missing_message(self) -> None:
        """Test missing message."""
        event = SimpleNamespace(delta=SimpleNamespace())
        assert extract_tool_call_start(event) == ('', '')

    def test_missing_tool_calls(self) -> None:
        """Test missing tool calls."""
        event = SimpleNamespace(delta=SimpleNamespace(message=SimpleNamespace()))
        assert extract_tool_call_start(event) == ('', '')

    def test_missing_function(self) -> None:
        """Test missing function."""
        event = SimpleNamespace(
            delta=SimpleNamespace(message=SimpleNamespace(tool_calls=SimpleNamespace(id='tc-2', function=None)))
        )
        assert extract_tool_call_start(event) == ('tc-2', '')

    def test_none_id(self) -> None:
        """Test none id."""
        event = SimpleNamespace(
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id=None,
                        function=SimpleNamespace(name='f'),
                    )
                )
            )
        )
        assert extract_tool_call_start(event) == ('', 'f')


class TestExtractToolCallDeltaArgs:
    """Tests for extract_tool_call_delta_args."""

    def test_full_path(self) -> None:
        """Test full path."""
        event = SimpleNamespace(
            delta=SimpleNamespace(
                message=SimpleNamespace(tool_calls=SimpleNamespace(function=SimpleNamespace(arguments='{"a":1}')))
            )
        )
        assert extract_tool_call_delta_args(event) == '{"a":1}'

    def test_missing_delta(self) -> None:
        """Test missing delta."""
        event = SimpleNamespace()
        assert extract_tool_call_delta_args(event) == ''

    def test_missing_function(self) -> None:
        """Test missing function."""
        event = SimpleNamespace(
            delta=SimpleNamespace(message=SimpleNamespace(tool_calls=SimpleNamespace(function=None)))
        )
        assert extract_tool_call_delta_args(event) == ''

    def test_none_arguments(self) -> None:
        """Test none arguments."""
        event = SimpleNamespace(
            delta=SimpleNamespace(
                message=SimpleNamespace(tool_calls=SimpleNamespace(function=SimpleNamespace(arguments=None)))
            )
        )
        assert extract_tool_call_delta_args(event) == ''


class TestExtractFinishReason:
    """Tests for extract_finish_reason."""

    def test_complete(self) -> None:
        """Test complete."""
        event = SimpleNamespace(delta=SimpleNamespace(finish_reason='COMPLETE'))
        assert extract_finish_reason(event) == 'COMPLETE'

    def test_missing_delta(self) -> None:
        """Test missing delta."""
        event = SimpleNamespace()
        assert extract_finish_reason(event) == ''

    def test_none_finish_reason(self) -> None:
        """Test none finish reason."""
        event = SimpleNamespace(delta=SimpleNamespace(finish_reason=None))
        assert extract_finish_reason(event) == ''

    def test_max_tokens(self) -> None:
        """Test max tokens."""
        event = SimpleNamespace(delta=SimpleNamespace(finish_reason='MAX_TOKENS'))
        assert extract_finish_reason(event) == 'MAX_TOKENS'
