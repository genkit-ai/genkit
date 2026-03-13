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

"""Tests for Hugging Face model implementation."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from genkit.core.typing import (
    GenerateRequest,
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
from genkit.plugins.huggingface.models import HuggingFaceModel


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock InferenceClient."""
    return MagicMock()


@pytest.fixture
def model(mock_client: MagicMock) -> HuggingFaceModel:
    """Create a HuggingFaceModel with mocked client."""
    with patch('genkit.plugins.huggingface.models.InferenceClient', return_value=mock_client):
        return HuggingFaceModel(
            model='meta-llama/Llama-3.3-70B-Instruct',
            token='test-token',
        )


def test_model_initialization(model: HuggingFaceModel) -> None:
    """Test model initialization."""
    assert model.name == 'meta-llama/Llama-3.3-70B-Instruct'


def test_model_with_provider() -> None:
    """Test model initialization with provider."""
    with patch('genkit.plugins.huggingface.models.InferenceClient'):
        model = HuggingFaceModel(
            model='meta-llama/Llama-3.3-70B-Instruct',
            token='test-token',
            provider='groq',
        )
        assert model.provider == 'groq'


def test_get_model_info(model: HuggingFaceModel) -> None:
    """Test get_model_info returns expected structure."""
    info = model.get_model_info()
    assert info is not None
    assert 'name' in info
    assert 'supports' in info


def test_convert_messages_text_only(model: HuggingFaceModel) -> None:
    """Test converting simple text messages."""
    messages = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
        Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there!'))]),
    ]

    hf_messages = model._convert_messages(messages)

    assert len(hf_messages) == 2
    assert hf_messages[0]['role'] == 'user'
    assert hf_messages[0]['content'] == 'Hello'
    assert hf_messages[1]['role'] == 'assistant'
    assert hf_messages[1]['content'] == 'Hi there!'


def test_convert_messages_system_role(model: HuggingFaceModel) -> None:
    """Test converting system messages."""
    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='You are helpful.'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))]),
    ]

    hf_messages = model._convert_messages(messages)

    assert len(hf_messages) == 2
    assert hf_messages[0]['role'] == 'system'
    assert hf_messages[0]['content'] == 'You are helpful.'


def test_convert_messages_with_tool_request(model: HuggingFaceModel) -> None:
    """Test converting messages with tool requests."""
    messages = [
        Message(
            role=Role.MODEL,
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(
                            ref='call_123',
                            name='get_weather',
                            input={'city': 'Paris'},
                        )
                    )
                )
            ],
        ),
    ]

    hf_messages = model._convert_messages(messages)

    assert len(hf_messages) == 1
    assert hf_messages[0]['role'] == 'assistant'
    assert 'tool_calls' in hf_messages[0]
    assert len(hf_messages[0]['tool_calls']) == 1
    assert hf_messages[0]['tool_calls'][0]['id'] == 'call_123'
    assert hf_messages[0]['tool_calls'][0]['function']['name'] == 'get_weather'


def test_convert_messages_with_tool_response(model: HuggingFaceModel) -> None:
    """Test converting messages with tool responses."""
    messages = [
        Message(
            role=Role.TOOL,
            content=[
                Part(
                    root=ToolResponsePart(
                        tool_response=ToolResponse(
                            ref='call_123',
                            name='get_weather',
                            output={'temperature': 20},
                        )
                    )
                )
            ],
        ),
    ]

    hf_messages = model._convert_messages(messages)

    assert len(hf_messages) == 1
    assert hf_messages[0]['role'] == 'tool'
    assert hf_messages[0]['tool_call_id'] == 'call_123'


def test_convert_tools(model: HuggingFaceModel) -> None:
    """Test converting tool definitions."""
    tools = [
        ToolDefinition(
            name='get_weather',
            description='Get weather for a city',
            input_schema={
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        ),
    ]

    hf_tools = model._convert_tools(tools)

    assert len(hf_tools) == 1
    assert hf_tools[0]['type'] == 'function'
    assert hf_tools[0]['function']['name'] == 'get_weather'
    assert hf_tools[0]['function']['description'] == 'Get weather for a city'
    assert 'additionalProperties' in hf_tools[0]['function']['parameters']


def test_get_response_format_json(model: HuggingFaceModel) -> None:
    """Test response format for JSON output without schema."""
    from genkit.core.typing import OutputConfig

    output = OutputConfig(format='json')
    result = model._get_response_format(output)

    assert result == {'type': 'json_object'}


def test_get_response_format_json_with_schema(model: HuggingFaceModel) -> None:
    """Test response format for JSON with schema uses OpenAI-compatible json_schema format."""
    from genkit.core.typing import OutputConfig

    schema = {'type': 'object', 'title': 'Person', 'properties': {'name': {'type': 'string'}}}
    output = OutputConfig(format='json', schema=schema)
    result = model._get_response_format(output)

    assert result == {
        'type': 'json_schema',
        'json_schema': {
            'name': 'Person',
            'schema': schema,
            'strict': True,
        },
    }


def test_get_response_format_text(model: HuggingFaceModel) -> None:
    """Test response format for text output returns None."""
    from genkit.core.typing import OutputConfig

    output = OutputConfig(format='text')
    result = model._get_response_format(output)

    assert result is None


def test_get_response_format_json_schema_without_title(model: HuggingFaceModel) -> None:
    """Test that schemas without a title fall back to 'Response' as the name."""
    from genkit.core.typing import OutputConfig

    schema = {'type': 'object', 'properties': {'name': {'type': 'string'}}}
    output = OutputConfig(format='json', schema=schema)
    result = model._get_response_format(output)

    assert result is not None
    assert result['type'] == 'json_schema'
    assert result['json_schema']['name'] == 'Response'
    assert result['json_schema']['schema'] == schema
    assert result['json_schema']['strict'] is True


@patch('genkit.plugins.huggingface.models.InferenceClient')
@pytest.mark.asyncio
async def test_generate_simple_request(mock_client_class: MagicMock) -> None:
    """Test simple generate request."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = 'Hello, world!'
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_client.chat_completion.return_value = mock_response

    model = HuggingFaceModel(
        model='meta-llama/Llama-3.3-70B-Instruct',
        token='test-token',
    )

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))])],
    )

    response = await model.generate(request)

    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert len(response.message.content) == 1
    assert response.message.content[0].root.text == 'Hello, world!'
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


@patch('genkit.plugins.huggingface.models.InferenceClient')
@pytest.mark.asyncio
async def test_generate_with_tool_calls(mock_client_class: MagicMock) -> None:
    """Test generate with tool calls in response."""
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    # Mock tool call in response
    mock_tool_call = MagicMock()
    mock_tool_call.id = 'call_abc'
    mock_tool_call.function = MagicMock()
    mock_tool_call.function.name = 'get_weather'
    mock_tool_call.function.arguments = '{"city": "Paris"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage = None
    mock_client.chat_completion.return_value = mock_response

    model = HuggingFaceModel(
        model='meta-llama/Llama-3.3-70B-Instruct',
        token='test-token',
    )

    request = GenerateRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Weather in Paris?'))])],
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather',
                input_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
            )
        ],
    )

    response = await model.generate(request)

    assert response.message is not None
    assert len(response.message.content) == 1
    tool_part = response.message.content[0].root
    assert isinstance(tool_part, ToolRequestPart)
    assert tool_part.tool_request.name == 'get_weather'
    assert tool_part.tool_request.ref == 'call_abc'


def test_to_generate_fn(model: HuggingFaceModel) -> None:
    """Test to_generate_fn returns callable."""
    fn = model.to_generate_fn()
    assert callable(fn)
    assert fn == model.generate


class TestResolveSchemaRefs:
    """Tests for HuggingFaceModel._resolve_schema_refs."""

    def test_no_defs_returns_schema_unchanged(self) -> None:
        """Schema without $defs is returned as-is."""
        schema: dict = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
            'title': 'Simple',
        }
        result = HuggingFaceModel._resolve_schema_refs(schema)
        assert result == schema

    def test_single_ref_is_inlined(self) -> None:
        """A single $ref is replaced with the corresponding $defs entry."""
        schema: dict = {
            '$defs': {
                'Skills': {
                    'type': 'object',
                    'properties': {
                        'strength': {'type': 'integer'},
                    },
                    'required': ['strength'],
                    'title': 'Skills',
                },
            },
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'skills': {'$ref': '#/$defs/Skills'},
            },
            'required': ['name', 'skills'],
            'title': 'Character',
        }
        result = HuggingFaceModel._resolve_schema_refs(schema)

        # $defs should be stripped from the result.
        assert '$defs' not in result
        # The $ref should be replaced with the inlined definition.
        assert result['properties']['skills'] == {
            'type': 'object',
            'properties': {'strength': {'type': 'integer'}},
            'required': ['strength'],
            'title': 'Skills',
        }
        # Other fields should be preserved.
        assert result['title'] == 'Character'
        assert result['properties']['name'] == {'type': 'string'}

    def test_nested_refs_are_resolved(self) -> None:
        """$refs inside $defs entries are recursively resolved."""
        schema: dict = {
            '$defs': {
                'Inner': {
                    'type': 'object',
                    'properties': {'value': {'type': 'integer'}},
                    'title': 'Inner',
                },
                'Outer': {
                    'type': 'object',
                    'properties': {'inner': {'$ref': '#/$defs/Inner'}},
                    'title': 'Outer',
                },
            },
            'type': 'object',
            'properties': {'data': {'$ref': '#/$defs/Outer'}},
            'title': 'Root',
        }
        result = HuggingFaceModel._resolve_schema_refs(schema)

        assert '$defs' not in result
        # Outer.inner should be fully resolved to Inner's definition.
        outer = result['properties']['data']
        assert outer['properties']['inner'] == {
            'type': 'object',
            'properties': {'value': {'type': 'integer'}},
            'title': 'Inner',
        }

    def test_unknown_ref_preserved(self) -> None:
        """An unresolvable $ref is kept so the backend surfaces the error."""
        schema: dict = {
            '$defs': {},
            'type': 'object',
            'properties': {
                'item': {'$ref': '#/$defs/Unknown'},
            },
            'title': 'Root',
        }
        result = HuggingFaceModel._resolve_schema_refs(schema)

        # Unknown ref kept as-is.
        assert result['properties']['item'] == {'$ref': '#/$defs/Unknown'}

    def test_ref_in_array_items_is_inlined(self) -> None:
        """$ref inside array items is resolved."""
        schema: dict = {
            '$defs': {
                'Tag': {
                    'type': 'object',
                    'properties': {'label': {'type': 'string'}},
                    'title': 'Tag',
                },
            },
            'type': 'object',
            'properties': {
                'tags': {
                    'type': 'array',
                    'items': {'$ref': '#/$defs/Tag'},
                },
            },
            'title': 'Root',
        }
        result = HuggingFaceModel._resolve_schema_refs(schema)

        assert '$defs' not in result
        assert result['properties']['tags']['items'] == {
            'type': 'object',
            'properties': {'label': {'type': 'string'}},
            'title': 'Tag',
        }

    def test_original_schema_is_not_mutated(self) -> None:
        """The input schema dict must not be modified in-place."""
        schema: dict = {
            '$defs': {
                'Foo': {'type': 'string'},
            },
            'type': 'object',
            'properties': {'foo': {'$ref': '#/$defs/Foo'}},
            'title': 'Root',
        }
        original = copy.deepcopy(schema)
        HuggingFaceModel._resolve_schema_refs(schema)

        assert schema == original
