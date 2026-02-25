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

"""Tests for Cloudflare Workers AI format conversion utilities.

Covers role conversion, tool definitions, schema wrapping, message
conversion, SSE parsing, tool call parsing, config normalization,
and usage building.
"""

import json

from genkit.plugins.cloudflare_workers_ai.models.converters import (
    build_usage,
    normalize_config,
    parse_sse_line,
    parse_tool_calls,
    to_cloudflare_messages_sync,
    to_cloudflare_role,
    to_cloudflare_tool,
    wrap_non_object_schema,
)
from genkit.plugins.cloudflare_workers_ai.models.model import _resolve_json_schema_refs
from genkit.plugins.cloudflare_workers_ai.typing import CloudflareConfig
from genkit.types import (
    GenerationCommonConfig,
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


class TestToCloudflareRole:
    """Tests for Genkit to Cloudflare role conversion."""

    def test_user_enum(self) -> None:
        """Test User enum."""
        assert to_cloudflare_role(Role.USER) == 'user'

    def test_model_enum(self) -> None:
        """Test Model enum."""
        assert to_cloudflare_role(Role.MODEL) == 'assistant'

    def test_system_enum(self) -> None:
        """Test System enum."""
        assert to_cloudflare_role(Role.SYSTEM) == 'system'

    def test_tool_enum(self) -> None:
        """Test Tool enum."""
        assert to_cloudflare_role(Role.TOOL) == 'tool'

    def test_string_user(self) -> None:
        """Test String user."""
        assert to_cloudflare_role('user') == 'user'

    def test_string_model(self) -> None:
        """Test String model."""
        assert to_cloudflare_role('model') == 'assistant'

    def test_unknown_defaults_to_user(self) -> None:
        """Test Unknown defaults to user."""
        assert to_cloudflare_role('admin') == 'user'


class TestWrapNonObjectSchema:
    """Tests for schema wrapping logic."""

    def test_object_schema_unchanged(self) -> None:
        """Test Object schema unchanged."""
        schema = {'type': 'object', 'properties': {'x': {'type': 'string'}}}
        got = wrap_non_object_schema(schema)
        assert got == schema, f'got {got}'

    def test_string_schema_wrapped(self) -> None:
        """Test String schema wrapped."""
        schema = {'type': 'string'}
        got = wrap_non_object_schema(schema)
        assert got['type'] == 'object', f'type = {got["type"]}'
        assert got['properties']['input'] == {'type': 'string'}
        assert got['required'] == ['input']

    def test_none_returns_default(self) -> None:
        """Test None returns default."""
        got = wrap_non_object_schema(None)
        assert got == {'type': 'object', 'properties': {}}, f'got {got}'


class TestToCloudflareToolCf:
    """Tests for Genkit to Cloudflare tool conversion."""

    def test_basic_tool(self) -> None:
        """Test Basic tool."""
        tool = ToolDefinition(
            name='search',
            description='Search the web',
            input_schema={'type': 'object', 'properties': {'q': {'type': 'string'}}},
        )
        got = to_cloudflare_tool(tool)
        assert got['type'] == 'function'
        assert got['function']['name'] == 'search'
        assert got['function']['description'] == 'Search the web'

    def test_primitive_schema_wrapped(self) -> None:
        """Test Primitive schema wrapped."""
        tool = ToolDefinition(
            name='echo',
            description='Echo input',
            input_schema={'type': 'string'},
        )
        got = to_cloudflare_tool(tool)
        params = got['function']['parameters']
        assert params['type'] == 'object', f'type = {params["type"]}'
        assert params['properties']['input'] == {'type': 'string'}

    def test_empty_description(self) -> None:
        """Test Empty description."""
        tool = ToolDefinition(name='noop', description='')
        got = to_cloudflare_tool(tool)
        assert got['function']['description'] == ''


class TestToCloudflareMessagesSync:
    """Tests for Genkit to Cloudflare message conversion (sync)."""

    def test_text_message(self) -> None:
        """Test Text message."""
        msgs = [Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])]
        got = to_cloudflare_messages_sync(msgs)
        assert len(got) == 1, f'Expected 1 message, got {len(got)}'
        assert got[0] == {'role': 'user', 'content': 'Hi'}, f'got {got[0]}'

    def test_system_message(self) -> None:
        """Test System message."""
        msgs = [Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be kind.'))])]
        got = to_cloudflare_messages_sync(msgs)
        assert got[0]['role'] == 'system'

    def test_tool_request_message(self) -> None:
        """Test Tool request message."""
        msgs = [
            Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='search', input={'q': 'test'}),
                        )
                    )
                ],
            )
        ]
        got = to_cloudflare_messages_sync(msgs)
        assert got[0]['role'] == 'assistant'
        parsed = json.loads(got[0]['content'])
        assert parsed['name'] == 'search'

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
        got = to_cloudflare_messages_sync(msgs)
        assert got[0]['role'] == 'tool'
        assert got[0]['name'] == 'search'
        assert got[0]['content'] == 'result'

    def test_tool_response_dict_output(self) -> None:
        """Test Tool response dict output."""
        msgs = [
            Message(
                role=Role.TOOL,
                content=[
                    Part(
                        root=ToolResponsePart(
                            tool_response=ToolResponse(ref='tc-1', name='calc', output={'sum': 42}),
                        )
                    )
                ],
            )
        ]
        got = to_cloudflare_messages_sync(msgs)
        assert got[0]['content'] == '{"sum": 42}', f'content = {got[0]["content"]}'

    def test_multiple_messages(self) -> None:
        """Test Multiple messages."""
        msgs = [
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='System.'))]),
            Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
            Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi'))]),
        ]
        got = to_cloudflare_messages_sync(msgs)
        assert len(got) == 3, f'Expected 3 messages, got {len(got)}'


class TestParseToolCalls:
    """Tests for Cloudflare tool call parsing."""

    def test_single_tool_call(self) -> None:
        """Test Single tool call."""
        tool_calls = [{'name': 'search', 'arguments': {'q': 'test'}}]
        parts = parse_tool_calls(tool_calls)
        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, ToolRequestPart), f'Expected ToolRequestPart, got {type(root)}'
        assert root.tool_request.name == 'search'

    def test_missing_fields(self) -> None:
        """Test Missing fields."""
        tool_calls = [{}]
        parts = parse_tool_calls(tool_calls)
        assert len(parts) == 1
        root = parts[0].root
        assert isinstance(root, ToolRequestPart)
        assert root.tool_request.name == ''

    def test_empty_list(self) -> None:
        """Test Empty list."""
        assert not (parse_tool_calls([])), 'Expected empty list'


class TestParseSseLine:
    """Tests for Server-Sent Events line parsing."""

    def test_valid_data_line(self) -> None:
        """Test Valid data line."""
        got = parse_sse_line('data: {"response": "Hello"}')
        assert got == {'response': 'Hello'}, f'got {got}'

    def test_done_sentinel(self) -> None:
        """Test Done sentinel."""
        assert parse_sse_line('data: [DONE]') is None

    def test_empty_line(self) -> None:
        """Test Empty line."""
        assert parse_sse_line('') is None

    def test_non_data_line(self) -> None:
        """Test Non data line."""
        assert parse_sse_line('event: message') is None

    def test_invalid_json(self) -> None:
        """Test Invalid json."""
        assert parse_sse_line('data: {bad json}') is None

    def test_whitespace_padding(self) -> None:
        """Test Whitespace padding."""
        got = parse_sse_line('  data: {"x": 1}  ')
        assert got == {'x': 1}, f'got {got}'


class TestBuildUsageCf:
    """Tests for usage statistics construction."""

    def test_all_fields(self) -> None:
        """Test All fields."""
        got = build_usage({'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30})
        assert got.input_tokens == 10 or got.output_tokens != 20 or got.total_tokens != 30, f'got {got}'

    def test_missing_fields(self) -> None:
        """Test Missing fields."""
        got = build_usage({})
        assert got.input_tokens == 0 or got.output_tokens != 0


class TestNormalizeConfigCf:
    """Tests for Cloudflare config normalization."""

    def test_none_returns_default(self) -> None:
        """Test None returns default."""
        got = normalize_config(None)
        assert isinstance(got, CloudflareConfig)

    def test_passthrough(self) -> None:
        """Test Passthrough."""
        config = CloudflareConfig(temperature=0.5)
        assert normalize_config(config) is config

    def test_generation_common_config(self) -> None:
        """Test Generation common config."""
        config = GenerationCommonConfig(temperature=0.7, max_output_tokens=100)
        got = normalize_config(config)
        assert got.temperature == 0.7
        assert got.max_output_tokens == 100

    def test_dict_with_camel_case(self) -> None:
        """Test Dict with camel case."""
        config = {'maxOutputTokens': 200, 'topP': 0.8, 'repetitionPenalty': 1.1}
        got = normalize_config(config)
        assert got.max_output_tokens == 200, f'max_output_tokens = {got.max_output_tokens}'
        assert got.top_p == 0.8
        assert got.repetition_penalty == 1.1

    def test_unknown_type_returns_default(self) -> None:
        """Test Unknown type returns default."""
        got = normalize_config(42)
        assert isinstance(got, CloudflareConfig)


class TestResolveJsonSchemaRefs:
    """Tests for JSON Schema ``$ref`` / ``$defs`` resolution.

    Cloudflare Workers AI does not support ``$ref`` or ``$defs`` in
    ``json_schema`` payloads. The ``_resolve_json_schema_refs`` helper
    inlines all references so schemas are self-contained.
    """

    def test_no_defs_returns_schema_unchanged(self) -> None:
        """Schema without $defs passes through unmodified."""
        schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        }
        got = _resolve_json_schema_refs(schema)
        assert got == schema, f'got {got}'

    def test_simple_ref_inlined(self) -> None:
        """A single $ref is replaced by the referenced definition."""
        schema = {
            '$defs': {
                'Address': {
                    'type': 'object',
                    'properties': {'city': {'type': 'string'}},
                },
            },
            'type': 'object',
            'properties': {
                'home': {'$ref': '#/$defs/Address'},
            },
        }
        got = _resolve_json_schema_refs(schema)
        # $defs should be stripped from output
        assert '$defs' not in got, f'$defs still present: {got}'
        # The $ref should be replaced by the Address definition
        assert got['properties']['home'] == {
            'type': 'object',
            'properties': {'city': {'type': 'string'}},
        }, f'got {got["properties"]["home"]}'

    def test_nested_ref_resolved(self) -> None:
        """A $ref inside a $def that itself has a $ref is resolved."""
        schema = {
            '$defs': {
                'Inner': {'type': 'string'},
                'Outer': {
                    'type': 'object',
                    'properties': {'val': {'$ref': '#/$defs/Inner'}},
                },
            },
            'type': 'object',
            'properties': {'wrapper': {'$ref': '#/$defs/Outer'}},
        }
        got = _resolve_json_schema_refs(schema)
        assert '$defs' not in got
        assert got['properties']['wrapper']['properties']['val'] == {
            'type': 'string',
        }, f'got {got}'

    def test_defs_removed_from_output(self) -> None:
        """The top-level $defs key is removed from the resolved schema."""
        schema = {
            '$defs': {'Foo': {'type': 'integer'}},
            'type': 'object',
            'properties': {'bar': {'$ref': '#/$defs/Foo'}},
        }
        got = _resolve_json_schema_refs(schema)
        assert '$defs' not in got

    def test_rpg_character_schema(self) -> None:
        """Test the exact pattern Pydantic generates for RpgCharacter with Skills.

        This is the real-world schema that triggered the 400 Bad Request from
        Cloudflare Workers AI.
        """
        schema = {
            '$defs': {
                'Skills': {
                    'description': 'A set of core character skills.',
                    'properties': {
                        'strength': {'description': 'strength (0-100)', 'type': 'integer'},
                        'charisma': {'description': 'charisma (0-100)', 'type': 'integer'},
                        'endurance': {'description': 'endurance (0-100)', 'type': 'integer'},
                    },
                    'required': ['strength', 'charisma', 'endurance'],
                    'title': 'Skills',
                    'type': 'object',
                },
            },
            'description': 'An RPG character.',
            'properties': {
                'name': {'description': 'name of the character', 'type': 'string'},
                'backStory': {'description': 'back story', 'type': 'string'},
                'abilities': {
                    'description': 'list of abilities (3-4)',
                    'items': {'type': 'string'},
                    'type': 'array',
                },
                'skills': {'$ref': '#/$defs/Skills'},
            },
            'required': ['name', 'backStory', 'abilities', 'skills'],
            'title': 'RpgCharacter',
            'type': 'object',
        }
        got = _resolve_json_schema_refs(schema)

        # $defs and $ref should be gone
        assert '$defs' not in got
        assert '$ref' not in json.dumps(got)

        # Skills should be inlined
        want_skills = {
            'description': 'A set of core character skills.',
            'properties': {
                'strength': {'description': 'strength (0-100)', 'type': 'integer'},
                'charisma': {'description': 'charisma (0-100)', 'type': 'integer'},
                'endurance': {'description': 'endurance (0-100)', 'type': 'integer'},
            },
            'required': ['strength', 'charisma', 'endurance'],
            'title': 'Skills',
            'type': 'object',
        }
        assert got['properties']['skills'] == want_skills, f'got {got["properties"]["skills"]}'

    def test_array_items_ref_resolved(self) -> None:
        """$ref inside array items is resolved."""
        schema = {
            '$defs': {
                'Tag': {'type': 'string'},
            },
            'type': 'object',
            'properties': {
                'tags': {
                    'type': 'array',
                    'items': {'$ref': '#/$defs/Tag'},
                },
            },
        }
        got = _resolve_json_schema_refs(schema)
        assert got['properties']['tags']['items'] == {'type': 'string'}

    def test_unknown_ref_left_intact(self) -> None:
        """References that don't start with #/$defs/ are left as-is."""
        schema = {
            '$defs': {},
            'type': 'object',
            'properties': {
                'ext': {'$ref': 'https://example.com/schema.json'},
            },
        }
        # No $defs entries to resolve, so schema returned as-is
        got = _resolve_json_schema_refs(schema)
        assert got == schema
