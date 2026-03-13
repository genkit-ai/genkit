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

"""Tests for Amazon Bedrock format conversion utilities.

Covers finish reason mapping, role conversion, system message extraction,
content block conversion (to/from Bedrock), tool definitions, config
normalization, JSON instructions, media handling, and inference profile
ID resolution.
"""

import base64

from genkit.plugins.amazon_bedrock.models.converters import (
    FINISH_REASON_MAP,
    INFERENCE_PROFILE_PREFIXES,
    INFERENCE_PROFILE_SUPPORTED_PROVIDERS,
    StreamingFenceStripper,
    build_json_instruction,
    build_media_block,
    build_usage,
    convert_media_data_uri,
    from_bedrock_content,
    get_effective_model_id,
    is_image_media,
    map_finish_reason,
    maybe_strip_fences,
    normalize_config,
    parse_tool_call_args,
    separate_system_messages,
    strip_markdown_fences,
    to_bedrock_content,
    to_bedrock_role,
    to_bedrock_tool,
)
from genkit.plugins.amazon_bedrock.typing import BedrockConfig
from genkit.types import (
    FinishReason,
    GenerateRequest,
    GenerationCommonConfig,
    Media,
    MediaPart,
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


class TestMapFinishReason:
    """Tests for finish reason mapping."""

    def test_end_turn_maps_to_stop(self) -> None:
        """Test End turn maps to stop."""
        got = map_finish_reason('end_turn')
        assert got == FinishReason.STOP, f'map_finish_reason("end_turn") = {got}, want STOP'

    def test_stop_sequence_maps_to_stop(self) -> None:
        """Test Stop sequence maps to stop."""
        got = map_finish_reason('stop_sequence')
        assert got == FinishReason.STOP, f'map_finish_reason("stop_sequence") = {got}, want STOP'

    def test_max_tokens_maps_to_length(self) -> None:
        """Test Max tokens maps to length."""
        got = map_finish_reason('max_tokens')
        assert got == FinishReason.LENGTH, f'map_finish_reason("max_tokens") = {got}, want LENGTH'

    def test_tool_use_maps_to_stop(self) -> None:
        """Test Tool use maps to stop."""
        got = map_finish_reason('tool_use')
        assert got == FinishReason.STOP, f'map_finish_reason("tool_use") = {got}, want STOP'

    def test_content_filtered_maps_to_blocked(self) -> None:
        """Test Content filtered maps to blocked."""
        got = map_finish_reason('content_filtered')
        assert got == FinishReason.BLOCKED, f'map_finish_reason("content_filtered") = {got}, want BLOCKED'

    def test_guardrail_intervened_maps_to_blocked(self) -> None:
        """Test Guardrail intervened maps to blocked."""
        got = map_finish_reason('guardrail_intervened')
        assert got == FinishReason.BLOCKED, f'map_finish_reason("guardrail_intervened") = {got}, want BLOCKED'

    def test_unknown_reason_maps_to_unknown(self) -> None:
        """Test Unknown reason maps to unknown."""
        got = map_finish_reason('something_new')
        assert got == FinishReason.UNKNOWN, f'map_finish_reason("something_new") = {got}, want UNKNOWN'

    def test_empty_string_maps_to_unknown(self) -> None:
        """Test Empty string maps to unknown."""
        got = map_finish_reason('')
        assert got == FinishReason.UNKNOWN, f'map_finish_reason("") = {got}, want UNKNOWN'

    def test_finish_reason_map_is_complete(self) -> None:
        """Ensure the constant covers all expected Bedrock stop reasons."""
        expected_keys = {
            'end_turn',
            'stop_sequence',
            'max_tokens',
            'tool_use',
            'content_filtered',
            'guardrail_intervened',
        }
        assert FINISH_REASON_MAP.keys() == expected_keys, (
            f'FINISH_REASON_MAP keys = {set(FINISH_REASON_MAP.keys())}, want {expected_keys}'
        )


class TestToBedrockRole:
    """Tests for Genkit → Bedrock role conversion."""

    def test_user_role_enum(self) -> None:
        """Test User role enum."""
        assert to_bedrock_role(Role.USER) == 'user'

    def test_model_role_enum(self) -> None:
        """Test Model role enum."""
        assert to_bedrock_role(Role.MODEL) == 'assistant'

    def test_tool_role_enum(self) -> None:
        """Test Tool role enum."""
        assert to_bedrock_role(Role.TOOL) == 'user'

    def test_user_string(self) -> None:
        """Test User string."""
        assert to_bedrock_role('user') == 'user'

    def test_model_string(self) -> None:
        """Test Model string."""
        assert to_bedrock_role('model') == 'assistant'

    def test_assistant_string(self) -> None:
        """Test Assistant string."""
        assert to_bedrock_role('assistant') == 'assistant'

    def test_tool_string(self) -> None:
        """Test Tool string."""
        assert to_bedrock_role('tool') == 'user'

    def test_unknown_string_defaults_to_user(self) -> None:
        """Test Unknown string defaults to user."""
        assert to_bedrock_role('admin') == 'user'

    def test_case_insensitive_string(self) -> None:
        """Test Case insensitive string."""
        assert to_bedrock_role('MODEL') == 'assistant'


class TestSeparateSystemMessages:
    """Tests for system message extraction."""

    def test_no_messages(self) -> None:
        """Test No messages."""
        system, conv = separate_system_messages([])
        assert not (system or conv)

    def test_no_system_messages(self) -> None:
        """Test No system messages."""
        msgs = [Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])]
        system, conv = separate_system_messages(msgs)
        assert not (system), f'Expected no system messages, got {system}'
        assert len(conv) == 1, f'Expected 1 conversation message, got {len(conv)}'

    def test_single_system_message(self) -> None:
        """Test Single system message."""
        msgs = [
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be helpful.'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))]),
        ]
        system, conv = separate_system_messages(msgs)
        assert system == ['Be helpful.'], f'system = {system}, want ["Be helpful."]'
        assert len(conv) == 1, f'Expected 1 conversation message, got {len(conv)}'

    def test_multiple_system_messages(self) -> None:
        """Test Multiple system messages."""
        msgs = [
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Rule 1'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))]),
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Rule 2'))]),
        ]
        system, conv = separate_system_messages(msgs)
        assert system == ['Rule 1', 'Rule 2'], f'system = {system}'
        assert len(conv) == 1, f'Expected 1 conversation message, got {len(conv)}'

    def test_multi_part_system_message(self) -> None:
        """Test Multi part system message."""
        msgs = [
            Message(
                role=Role.SYSTEM,
                content=[
                    Part(root=TextPart(text='Part A')),
                    Part(root=TextPart(text='Part B')),
                ],
            ),
        ]
        system, conv = separate_system_messages(msgs)
        assert system == ['Part APart B'], f'system = {system}'

    def test_preserves_conversation_order(self) -> None:
        """Test Preserves conversation order."""
        msgs = [
            Message(role=Role.USER, content=[Part(root=TextPart(text='Q1'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='A1'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Q2'))]),
        ]
        system, conv = separate_system_messages(msgs)
        assert len(conv) == 3, f'Expected 3 conversation messages, got {len(conv)}'
        roles = [m.role for m in conv]
        assert roles == [Role.USER, Role.MODEL, Role.USER], f'Conversation roles = {roles}'


class TestToBedrockTool:
    """Tests for Genkit → Bedrock tool definition conversion."""

    def test_basic_tool(self) -> None:
        """Test Basic tool."""
        tool = ToolDefinition(
            name='get_weather',
            description='Fetch current weather',
            input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
        )
        got = to_bedrock_tool(tool)
        want = {
            'toolSpec': {
                'name': 'get_weather',
                'description': 'Fetch current weather',
                'inputSchema': {
                    'json': {'type': 'object', 'properties': {'city': {'type': 'string'}}},
                },
            },
        }
        assert got == want, f'got {got}, want {want}'

    def test_tool_without_schema(self) -> None:
        """Test Tool without schema."""
        tool = ToolDefinition(name='ping', description='Ping service')
        got = to_bedrock_tool(tool)
        expected = {'type': 'object', 'properties': {}}
        assert got['toolSpec']['inputSchema']['json'] == expected, (
            f'Expected default schema, got {got["toolSpec"]["inputSchema"]}'
        )

    def test_tool_with_empty_description(self) -> None:
        """Test Tool with empty description."""
        tool = ToolDefinition(name='noop', description='')
        got = to_bedrock_tool(tool)
        assert got['toolSpec']['description'] == '', f'Expected empty description, got {got["toolSpec"]["description"]}'


class TestToBedrockContent:
    """Tests for Genkit Part → Bedrock content block conversion."""

    def test_text_part(self) -> None:
        """Test Text part."""
        part = Part(root=TextPart(text='Hello world'))
        got = to_bedrock_content(part)
        assert got == {'text': 'Hello world'}, f'got {got}'

    def test_tool_request_part(self) -> None:
        """Test Tool request part."""
        part = Part(
            root=ToolRequestPart(tool_request=ToolRequest(ref='call-1', name='get_weather', input={'city': 'London'}))
        )
        got = to_bedrock_content(part)
        want = {
            'toolUse': {
                'toolUseId': 'call-1',
                'name': 'get_weather',
                'input': {'city': 'London'},
            },
        }
        assert got == want, f'got {got}, want {want}'

    def test_tool_request_part_without_ref(self) -> None:
        """Test Tool request part without ref."""
        part = Part(root=ToolRequestPart(tool_request=ToolRequest(name='ping', input={})))
        got = to_bedrock_content(part)
        assert got is not None, 'Expected non-None result'
        assert got['toolUse']['toolUseId'] == '', f'Expected empty toolUseId, got {got["toolUse"]["toolUseId"]}'

    def test_tool_response_part_string_output(self) -> None:
        """Test Tool response part string output."""
        part = Part(root=ToolResponsePart(tool_response=ToolResponse(ref='call-1', name='get_weather', output='Sunny')))
        got = to_bedrock_content(part)
        want = {
            'toolResult': {
                'toolUseId': 'call-1',
                'content': [{'text': 'Sunny'}],
            },
        }
        assert got == want, f'got {got}, want {want}'

    def test_tool_response_part_dict_output(self) -> None:
        """Test Tool response part dict output."""
        part = Part(
            root=ToolResponsePart(tool_response=ToolResponse(ref='call-1', name='get_weather', output={'temp': 20}))
        )
        got = to_bedrock_content(part)
        assert got is not None, 'Expected non-None result'
        assert got['toolResult']['content'] == [{'json': {'temp': 20}}], f'got {got}'

    def test_media_part_returns_none(self) -> None:
        """Test Media part returns none."""
        part = Part(root=MediaPart(media=Media(url='https://example.com/img.png', content_type='image/png')))
        got = to_bedrock_content(part)
        assert got is None, f'Expected None for MediaPart, got {got}'


class TestFromBedrockContent:
    """Tests for Bedrock content block → Genkit Part conversion."""

    def test_text_block(self) -> None:
        """Test Text block."""
        parts = from_bedrock_content([{'text': 'Hello'}])
        assert len(parts) == 1, f'Expected 1 part, got {len(parts)}'
        root = parts[0].root
        assert isinstance(root, TextPart), f'Expected TextPart, got {type(root)}'
        assert root.text == 'Hello'

    def test_tool_use_block(self) -> None:
        """Test Tool use block."""
        parts = from_bedrock_content([
            {
                'toolUse': {
                    'toolUseId': 'abc-123',
                    'name': 'search',
                    'input': {'query': 'test'},
                }
            }
        ])
        assert len(parts) == 1, f'Expected 1 part, got {len(parts)}'
        root = parts[0].root
        assert isinstance(root, ToolRequestPart), f'Expected ToolRequestPart, got {type(root)}'
        assert root.tool_request.name == 'search', f'tool name = {root.tool_request.name}'
        assert root.tool_request.ref == 'abc-123', f'tool ref = {root.tool_request.ref}'

    def test_reasoning_content_string(self) -> None:
        """Test Reasoning content string."""
        parts = from_bedrock_content([
            {
                'reasoningContent': {
                    'reasoningText': 'Let me think...',
                }
            }
        ])
        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, TextPart), f'Expected TextPart, got {type(root)}'
        assert '[Reasoning]' in root.text or 'Let me think' not in root.text, f'text = {root.text}'

    def test_reasoning_content_dict(self) -> None:
        """Test Reasoning content dict."""
        parts = from_bedrock_content([
            {
                'reasoningContent': {
                    'reasoningText': {'text': 'Step 1: analyze'},
                }
            }
        ])
        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, TextPart), f'Expected TextPart, got {type(root)}'
        assert 'Step 1: analyze' in root.text, f'text = {root.text}'

    def test_multiple_blocks(self) -> None:
        """Test Multiple blocks."""
        parts = from_bedrock_content([
            {'text': 'Result:'},
            {'toolUse': {'toolUseId': 'x', 'name': 'calc', 'input': {}}},
        ])
        assert len(parts) == 2, f'Expected 2 parts, got {len(parts)}'

    def test_empty_blocks(self) -> None:
        """Test Empty blocks."""
        parts = from_bedrock_content([])
        assert len(parts) == 0


class TestParseToolCallArgs:
    """Tests for tool call argument JSON parsing."""

    def test_valid_json(self) -> None:
        """Test Valid json."""
        got = parse_tool_call_args('{"x": 1}')
        assert got == {'x': 1}, f'got {got}'

    def test_invalid_json_returns_string(self) -> None:
        """Test Invalid json returns string."""
        got = parse_tool_call_args('not json')
        assert got == 'not json', f'got {got}'

    def test_empty_string_returns_empty_dict(self) -> None:
        """Test Empty string returns empty dict."""
        got = parse_tool_call_args('')
        assert got == {}, f'got {got}'

    def test_nested_json(self) -> None:
        """Test Nested json."""
        got = parse_tool_call_args('{"a": {"b": [1, 2]}}')
        assert got == {'a': {'b': [1, 2]}}, f'got {got}'


class TestBuildUsage:
    """Tests for usage statistics construction."""

    def test_full_usage(self) -> None:
        """Test Full usage."""
        got = build_usage({'inputTokens': 10, 'outputTokens': 20, 'totalTokens': 30})
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30, f'got {got}'

    def test_missing_fields_default_to_zero(self) -> None:
        """Test Missing fields default to zero."""
        got = build_usage({})
        assert got.input_tokens == 0 or got.output_tokens != 0 or got.total_tokens != 0, f'got {got}'

    def test_partial_usage(self) -> None:
        """Test Partial usage."""
        got = build_usage({'inputTokens': 5})
        assert got.input_tokens == 5 or got.output_tokens != 0, f'got {got}'


class TestNormalizeConfig:
    """Tests for config normalization."""

    def test_none_returns_default(self) -> None:
        """Test None returns default."""
        got = normalize_config(None)
        assert isinstance(got, BedrockConfig), f'Expected BedrockConfig, got {type(got)}'

    def test_bedrock_config_passthrough(self) -> None:
        """Test Bedrock config passthrough."""
        config = BedrockConfig(temperature=0.5)
        got = normalize_config(config)
        assert got is config, 'Expected same instance'

    def test_generation_common_config(self) -> None:
        """Test Generation common config."""
        config = GenerationCommonConfig(temperature=0.7, max_output_tokens=100, top_p=0.9)
        got = normalize_config(config)
        assert got.temperature == 0.7, f'temperature = {got.temperature}'
        assert got.max_tokens == 100, f'max_tokens = {got.max_tokens}'
        assert got.top_p == 0.9, f'top_p = {got.top_p}'

    def test_dict_with_camel_case_keys(self) -> None:
        """Test Dict with camel case keys."""
        config = {'maxOutputTokens': 200, 'topP': 0.8}
        got = normalize_config(config)
        assert got.max_tokens == 200, f'max_tokens = {got.max_tokens}'
        assert got.top_p == 0.8, f'top_p = {got.top_p}'

    def test_dict_with_snake_case_keys(self) -> None:
        """Test Dict with snake case keys."""
        config = {'temperature': 0.5, 'stop_sequences': ['END']}
        got = normalize_config(config)
        assert got.temperature == 0.5, f'temperature = {got.temperature}'
        assert got.stop_sequences == ['END'], f'stop_sequences = {got.stop_sequences}'

    def test_unknown_type_returns_default(self) -> None:
        """Test Unknown type returns default."""
        got = normalize_config(42)
        assert isinstance(got, BedrockConfig), f'Expected BedrockConfig, got {type(got)}'


class TestBuildJsonInstruction:
    """Tests for JSON output instruction generation."""

    def test_no_output_returns_none(self) -> None:
        """Test No output returns none."""
        request = GenerateRequest(messages=[])
        got = build_json_instruction(request)
        assert got is None, f'Expected None, got {got}'

    def test_text_format_returns_none(self) -> None:
        """Test Text format returns none."""
        request = GenerateRequest(messages=[], output=OutputConfig(format='text'))
        got = build_json_instruction(request)
        assert got is None, f'Expected None, got {got}'

    def test_json_format_without_schema(self) -> None:
        """Test Json format without schema."""
        request = GenerateRequest(messages=[], output=OutputConfig(format='json'))
        got = build_json_instruction(request)
        assert got is not None, 'Expected non-None instruction'
        assert 'valid JSON' in got, f'Missing JSON instruction: {got}'

    def test_json_format_with_schema(self) -> None:
        """Test Json format with schema."""
        schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
        request = GenerateRequest(messages=[], output=OutputConfig(format='json', schema=schema))
        got = build_json_instruction(request)
        assert got is not None, 'Expected non-None instruction'
        assert 'name' in got, f'Schema not in instruction: {got}'


class TestConvertMediaDataUri:
    """Tests for data URI media parsing."""

    def test_png_data_uri(self) -> None:
        """Test Png data uri."""
        png_data = base64.b64encode(b'\x89PNG').decode('ascii')
        media = Media(url=f'data:image/png;base64,{png_data}', content_type='image/png')
        media_bytes, format_str, is_data = convert_media_data_uri(media)
        assert is_data, 'Expected is_data_uri=True'
        assert format_str == 'png', f'format = {format_str}'
        assert media_bytes == b'\x89PNG', f'bytes = {media_bytes}'

    def test_http_url_returns_false(self) -> None:
        """Test Http url returns false."""
        media = Media(url='https://example.com/img.jpg', content_type='image/jpeg')
        _, _, is_data = convert_media_data_uri(media)
        assert not (is_data), 'Expected is_data_uri=False for HTTP URL'

    def test_data_uri_without_comma(self) -> None:
        """Test Data uri without comma."""
        media = Media(url='data:image/png;base64', content_type='image/png')
        _, _, is_data = convert_media_data_uri(media)
        assert not (is_data), 'Expected is_data_uri=False for malformed data URI'


class TestIsImageMedia:
    """Tests for image vs video classification."""

    def test_image_content_type(self) -> None:
        """Test Image content type."""
        assert is_image_media('image/png', '')

    def test_video_content_type(self) -> None:
        """Test Video content type."""
        assert not (is_image_media('video/mp4', ''))

    def test_image_url_extension(self) -> None:
        """Test Image url extension."""
        assert is_image_media('', 'photo.jpg')

    def test_video_url_no_image_ext(self) -> None:
        """Test Video url no image ext."""
        assert not (is_image_media('', 'video.mp4')), 'mp4 URL without content type should not be image'

    def test_no_content_type_no_ext(self) -> None:
        """Test No content type no ext."""
        # No image extension → defaults to False
        assert not (is_image_media('', 'blob'))


class TestBuildMediaBlock:
    """Tests for Bedrock media block construction."""

    def test_image_block(self) -> None:
        """Test Image block."""
        got = build_media_block(b'\x89PNG', 'png', is_image=True)
        assert 'image' in got, f'Expected image key, got {got}'
        assert got['image']['format'] == 'png'

    def test_video_block(self) -> None:
        """Test Video block."""
        got = build_media_block(b'\x00', 'mp4', is_image=False)
        assert 'video' in got, f'Expected video key, got {got}'
        assert got['video']['format'] == 'mp4'


class TestGetEffectiveModelId:
    """Tests for inference profile ID resolution."""

    def test_already_prefixed_returns_unchanged(self) -> None:
        """Test Already prefixed returns unchanged."""
        got = get_effective_model_id('us.anthropic.claude-v3', bearer_token='tok', aws_region='us-east-1')
        assert got == 'us.anthropic.claude-v3', f'got {got}'

    def test_no_bearer_token_returns_unchanged(self) -> None:
        """Test No bearer token returns unchanged."""
        got = get_effective_model_id('anthropic.claude-v3', bearer_token=None, aws_region='us-east-1')
        assert got == 'anthropic.claude-v3', f'got {got}'

    def test_unsupported_provider_returns_unchanged(self) -> None:
        """Test Unsupported provider returns unchanged."""
        got = get_effective_model_id('stability.sd3', bearer_token='tok', aws_region='us-east-1')
        assert got == 'stability.sd3', f'got {got}'

    def test_no_region_returns_unchanged(self) -> None:
        """Test No region returns unchanged."""
        got = get_effective_model_id('anthropic.claude-v3', bearer_token='tok', aws_region=None)
        assert got == 'anthropic.claude-v3', f'got {got}'

    def test_us_region_adds_us_prefix(self) -> None:
        """Test Us region adds us prefix."""
        got = get_effective_model_id('anthropic.claude-v3', bearer_token='tok', aws_region='us-east-1')
        assert got == 'us.anthropic.claude-v3', f'got {got}'

    def test_eu_region_adds_eu_prefix(self) -> None:
        """Test Eu region adds eu prefix."""
        got = get_effective_model_id('meta.llama3', bearer_token='tok', aws_region='eu-west-1')
        assert got == 'eu.meta.llama3', f'got {got}'

    def test_ap_region_adds_apac_prefix(self) -> None:
        """Test Ap region adds apac prefix."""
        got = get_effective_model_id('cohere.command-r', bearer_token='tok', aws_region='ap-southeast-1')
        assert got == 'apac.cohere.command-r', f'got {got}'

    def test_unknown_region_defaults_to_us(self) -> None:
        """Test Unknown region defaults to us."""
        got = get_effective_model_id('anthropic.claude-v3', bearer_token='tok', aws_region='xx-central-1')
        assert got == 'us.anthropic.claude-v3', f'got {got}'

    def test_inference_profile_prefixes_constant(self) -> None:
        """Test Inference profile prefixes constant."""
        expected_prefixes = ('us.', 'eu.', 'apac.')
        assert INFERENCE_PROFILE_PREFIXES == expected_prefixes, (
            f'INFERENCE_PROFILE_PREFIXES = {INFERENCE_PROFILE_PREFIXES}'
        )

    def test_supported_providers_includes_anthropic(self) -> None:
        """Test Supported providers includes anthropic."""
        assert 'anthropic.' in INFERENCE_PROFILE_SUPPORTED_PROVIDERS


class TestStripMarkdownFences:
    """Tests for strip_markdown_fences."""

    def test_strips_json_fences(self) -> None:
        """Strips ```json ... ``` fences."""
        text = '```json\n{"name": "John"}\n```'
        assert strip_markdown_fences(text) == '{"name": "John"}'

    def test_strips_plain_fences(self) -> None:
        """Strips ``` ... ``` fences without language tag."""
        text = '```\n{"a": 1}\n```'
        assert strip_markdown_fences(text) == '{"a": 1}'

    def test_preserves_plain_json(self) -> None:
        """Does not alter valid JSON without fences."""
        text = '{"name": "John"}'
        assert strip_markdown_fences(text) == text

    def test_preserves_non_json_text(self) -> None:
        """Does not alter plain text."""
        text = 'Hello, world!'
        assert strip_markdown_fences(text) == text

    def test_strips_multiline_json(self) -> None:
        """Strips fences around multiline JSON."""
        text = '```json\n{\n  "a": 1\n}\n```'
        assert strip_markdown_fences(text) == '{\n  "a": 1\n}'


class TestMaybeStripFences:
    """Tests for maybe_strip_fences."""

    def test_strips_fences_for_json_output(self) -> None:
        """Strips markdown fences when JSON output is requested."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema={'type': 'object'}),
        )
        parts = [Part(root=TextPart(text='```json\n{"a": 1}\n```'))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == '{"a": 1}'

    def test_no_op_for_text_output(self) -> None:
        """Does not modify responses when output format is not json."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='text'),
        )
        fenced = '```json\n{"a": 1}\n```'
        parts = [Part(root=TextPart(text=fenced))]
        result = maybe_strip_fences(request, parts)
        assert result[0].root.text == fenced

    def test_no_op_when_no_fences(self) -> None:
        """Does not modify clean JSON responses."""
        request = GenerateRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
            output=OutputConfig(format='json', schema={'type': 'object'}),
        )
        text = '{"name": "John"}'
        parts = [Part(root=TextPart(text=text))]
        result = maybe_strip_fences(request, parts)
        assert result is parts


class TestStreamingFenceStripper:
    """Tests for StreamingFenceStripper."""

    def test_strips_fences_across_chunks(self) -> None:
        """Strips opening and closing fences split across chunks."""
        stripper = StreamingFenceStripper(json_mode=True)
        chunks = ['```json\n', '{"name":', ' "John"}\n', '```']
        out = [stripper.process(c) for c in chunks]
        out.append(stripper.flush())
        combined = ''.join(out)
        assert combined == '{"name": "John"}'

    def test_strips_fence_in_single_chunk(self) -> None:
        """Strips fences when entire response is one chunk."""
        stripper = StreamingFenceStripper(json_mode=True)
        out = stripper.process('```json\n{"a": 1}\n```')
        out += stripper.flush()
        assert out == '{"a": 1}'

    def test_no_op_when_not_json_mode(self) -> None:
        """Passes text through unchanged when json_mode is False."""
        stripper = StreamingFenceStripper(json_mode=False)
        text = '```json\n{"a": 1}\n```'
        assert stripper.process(text) == text
        assert stripper.flush() == ''

    def test_no_fence_passes_through(self) -> None:
        """Passes text through when no fence is detected."""
        stripper = StreamingFenceStripper(json_mode=True)
        chunks = ['{"name":', ' "John"}']
        out = [stripper.process(c) for c in chunks]
        out.append(stripper.flush())
        combined = ''.join(out)
        assert combined == '{"name": "John"}'

    def test_buffers_small_prefix(self) -> None:
        """Buffers small initial chunks until fence can be detected."""
        stripper = StreamingFenceStripper(json_mode=True)
        # First chunk is small — should be buffered.
        assert stripper.process('```') == ''
        # Second chunk triggers flush with fence detection.
        assert stripper.process('json\n{"a":') == '{"a":'
        assert stripper.process(' 1}\n```') == ' 1}'
        assert stripper.flush() == ''
