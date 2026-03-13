# Copyright 2025 Google LLC
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


"""Tests for the Gemini model implementation."""

import base64
import sys
from unittest.mock import AsyncMock, MagicMock, patch

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

import pytest
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from genkit.ai import ActionRunContext
from genkit.core.schema import to_json_schema
from genkit.plugins.google_genai.models.gemini import (
    DEFAULT_SUPPORTS_MODEL,
    GeminiModel,
    GoogleAIGeminiVersion,
    VertexAIGeminiVersion,
    google_model_info,
    is_image_model,
    is_tts_model,
)
from genkit.types import (
    GenerateRequest,
    GenerateResponse,
    MediaPart,
    Message,
    ModelInfo,
    Part,
    Role,
    TextPart,
    ToolDefinition,
)

ALL_VERSIONS = list(GoogleAIGeminiVersion) + list(VertexAIGeminiVersion)
IMAGE_GENERATION_VERSIONS = [GoogleAIGeminiVersion.GEMINI_2_0_FLASH_EXP]


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in ALL_VERSIONS])
async def test_generate_text_response(mocker: MockerFixture, version: str) -> None:
    """Test the generate method for text responses."""
    response_text = 'request answer'
    request_text = 'response question'

    request = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
        ]
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text=response_text)]))
    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.return_value = resp

    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ActionRunContext()
    response = await gemini.generate(request, ctx)

    # Determine expected config based on model type
    if is_tts_model(version):
        expected_config = genai.types.GenerateContentConfig(response_modalities=['AUDIO'])
    elif is_image_model(version):
        expected_config = genai.types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    else:
        expected_config = None

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.generate_content(
            model=version,
            contents=[genai.types.Content(parts=[genai.types.Part(text=request_text)], role=Role.USER)],
            config=expected_config,
        )
    ])
    assert isinstance(response, GenerateResponse)
    assert response.message is not None
    assert response.message.content[0].root.text == response_text


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in ALL_VERSIONS])
async def test_generate_stream_text_response(mocker: MockerFixture, version: str) -> None:
    """Test the generate method for text responses."""
    response_text = 'request answer'
    request_text = 'response question'

    request = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
        ]
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text=response_text)]))

    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content_stream.__aiter__.side_effect = [resp]
    on_chunk_mock = mocker.MagicMock()
    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ActionRunContext(on_chunk=on_chunk_mock)
    response = await gemini.generate(request, ctx)

    # Determine expected config based on model type
    if is_tts_model(version):
        expected_config = genai.types.GenerateContentConfig(response_modalities=['AUDIO'])
    elif is_image_model(version):
        expected_config = genai.types.GenerateContentConfig(response_modalities=['TEXT', 'IMAGE'])
    else:
        expected_config = None

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.generate_content_stream(
            model=version,
            contents=[genai.types.Content(parts=[genai.types.Part(text=request_text)], role=Role.USER)],
            config=expected_config,
        )
    ])
    assert isinstance(response, GenerateResponse)
    assert response.message is not None
    assert response.message.content == []


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in IMAGE_GENERATION_VERSIONS])
async def test_generate_media_response(mocker: MockerFixture, version: str) -> None:
    """Test generate method for media responses."""
    request_text = 'response question'
    response_byte_string = b'\x89PNG\r\n\x1a\n'
    response_mimetype = 'image/png'
    modalities = ['Text', 'Image']

    request = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
        ],
        config={'response_modalities': modalities},
    )

    candidate = genai.types.Candidate(
        content=genai.types.Content(
            parts=[
                genai.types.Part(inline_data=genai.types.Blob(data=response_byte_string, mime_type=response_mimetype))
            ]
        )
    )
    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.return_value = resp

    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ActionRunContext()
    response = await gemini.generate(request, ctx)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.generate_content(
            model=version,
            contents=[genai.types.Content(parts=[genai.types.Part(text=request_text)], role=Role.USER)],
            config=genai.types.GenerateContentConfig(response_modalities=modalities),
        )
    ])
    assert isinstance(response, GenerateResponse)
    assert response.message is not None

    content = response.message.content[0]
    assert isinstance(content.root, MediaPart)

    assert content.root.media.content_type == response_mimetype

    # Verify the data URL contains the correct base64-encoded content
    # Data URLs have format: data:<mimetype>;base64,<data>
    data_url = content.root.media.url
    assert data_url.startswith(f'data:{response_mimetype};base64,')
    encoded_data = data_url.split(',', 1)[1]
    assert base64.b64decode(encoded_data) == response_byte_string


def test_convert_schema_property(mocker: MockerFixture) -> None:
    """Test _convert_schema_property."""
    googleai_client_mock = mocker.AsyncMock()
    gemini = GeminiModel('abc', googleai_client_mock)

    class Simple(BaseModel):
        foo: str = Field(description='foo field')
        bar: int = Field(description='bar field')
        # Note: baz: list[str] | None generates anyOf schema which is not supported by _convert_schema_property yet

    assert gemini._convert_schema_property(to_json_schema(Simple)) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                description='foo field',
            ),
            'bar': genai_types.Schema(
                type=genai_types.Type.INTEGER,
                description='bar field',
            ),
        },
        required=['foo', 'bar'],
    )

    class Nested(BaseModel):
        baz: int = Field(description='baz field')

    class WithNested(BaseModel):
        foo: str = Field(description='foo field')
        bar: Nested = Field(description='bar field')

    assert gemini._convert_schema_property(to_json_schema(WithNested)) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                description='foo field',
            ),
            'bar': genai_types.Schema(
                type=genai_types.Type.OBJECT,
                description='bar field',
                properties={
                    'baz': genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        description='baz field',
                    ),
                },
                required=['baz'],
            ),
        },
        required=['foo', 'bar'],
    )

    class TestEnum(StrEnum):
        FOO = 'foo'
        BAR = 'bar'

    class WitEnum(BaseModel):
        foo: TestEnum = Field(description='foo field')

    assert gemini._convert_schema_property(to_json_schema(WitEnum)) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                description='foo field',
                enum=['foo', 'bar'],
            ),
        },
        required=['foo'],
    )


@pytest.mark.asyncio
async def test_generate_with_system_instructions(mocker: MockerFixture) -> None:
    """Test Generate using system instructions."""
    response_text = 'request answer'
    request_text = 'response question'
    system_instruction = 'system instruction text'
    version = GoogleAIGeminiVersion.GEMINI_2_0_FLASH

    request = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
            Message(
                role=Role.SYSTEM,
                content=[
                    Part(root=TextPart(text=system_instruction)),
                ],
            ),
        ]
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text=response_text)]))
    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    expected_system_instruction = genai.types.Content(parts=[genai.types.Part(text=system_instruction)])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.return_value = resp

    gemini = GeminiModel(version, googleai_client_mock)
    ctx = ActionRunContext()

    response = await gemini.generate(request, ctx)

    googleai_client_mock.assert_has_calls([
        mocker.call.aio.models.generate_content(
            model=version,
            contents=[genai.types.Content(parts=[genai.types.Part(text=request_text)], role=Role.USER)],
            config=genai.types.GenerateContentConfig(system_instruction=expected_system_instruction),
        )
    ])
    assert isinstance(response, GenerateResponse)
    assert response.message is not None
    assert response.message.content[0].root.text == response_text


# Unit tests


@pytest.mark.parametrize(
    'input, expected',
    [
        (
            'lazaro',
            ModelInfo(
                label='Google AI - lazaro',
                supports=DEFAULT_SUPPORTS_MODEL,
            ),
        ),
        (
            'gemini-4-0-pro-delux-max',
            ModelInfo(
                label='Google AI - gemini-4-0-pro-delux-max',
                supports=DEFAULT_SUPPORTS_MODEL,
            ),
        ),
    ],
)
def test_google_model_info(input: str, expected: ModelInfo) -> None:
    """Tests for google_model_info."""
    model_info = google_model_info(input)

    assert model_info == expected


@pytest.fixture
def gemini_model_instance() -> GeminiModel:
    """Common initialization of GeminiModel."""
    version = 'version'
    mock_client = MagicMock(spec=genai.Client)

    return GeminiModel(
        version=version,
        client=mock_client,
    )


def test_gemini_model__init__() -> None:
    """Test for init gemini model."""
    version = 'version'
    mock_client = MagicMock(spec=genai.Client)

    model = GeminiModel(
        version=version,
        client=mock_client,
    )

    assert isinstance(model, GeminiModel)
    assert model._version == version
    assert model._client == mock_client


@patch('genkit.plugins.google_genai.models.gemini.GeminiModel._create_tool')
def test_gemini_model__get_tools(
    mock_create_tool: MagicMock,
    gemini_model_instance: GeminiModel,
) -> None:
    """Unit test for GeminiModel._get_tools."""
    mock_create_tool.return_value = genai_types.Tool()

    request_tools = [
        ToolDefinition(
            name='tool_1',
            description='model tool description',
            input_schema={},
            output_schema={
                'type': 'object',
                'properties': {
                    'test': {'type': 'string', 'description': 'test field'},
                },
            },
            metadata={'date': 'today'},
        ),
        ToolDefinition(
            name='tool_2',
            description='model tool description',
            input_schema={},
            output_schema={
                'type': 'object',
                'properties': {
                    'test': {'type': 'string', 'description': 'test field'},
                },
            },
            metadata={'date': 'today'},
        ),
    ]

    request = GenerateRequest(
        tools=request_tools,
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='test text')),
                ],
            ),
        ],
    )

    tools = gemini_model_instance._get_tools(request)

    assert len(tools) == len(request_tools)
    for tool in tools:
        assert isinstance(tool, genai_types.Tool)


@patch('genkit.plugins.google_genai.models.gemini.GeminiModel._convert_schema_property')
def test_gemini_model__create_tool(
    mock_convert_schema_property: MagicMock,
    gemini_model_instance: GeminiModel,
) -> None:
    """Unit tests for GeminiModel._create_tool."""
    tool_defined = ToolDefinition(
        name='model_tool',
        description='model tool description',
        input_schema={
            'type': 'str',
            'description': 'test field',
        },
        output_schema={
            'type': 'object',
            'properties': {
                'test': {'type': 'string', 'description': 'test field'},
            },
        },
        metadata={'date': 'today'},
    )

    mock_convert_schema_property.return_value = genai_types.Schema()

    gemini_tool = gemini_model_instance._create_tool(
        tool_defined,
    )

    assert isinstance(gemini_tool, genai_types.Tool)


@pytest.mark.parametrize(
    'input_schema, defs, expected_schema',
    [
        # Test Case 1: None input_schema
        (
            None,
            None,
            None,
        ),
        # Test Case 2: input_schema without 'type'
        (
            {'description': 'A simple description'},
            None,
            None,
        ),
        # Test Case 3: Simple string type
        (
            {'type': 'STRING', 'description': 'A string field', 'required': ['field']},
            None,
            genai_types.Schema(description='A string field', required=['field'], type=genai_types.Type.STRING),
        ),
        # Test Case 4: String with enum
        (
            {'type': 'STRING', 'enum': ['A', 'B']},
            None,
            genai_types.Schema(type=genai_types.Type.STRING, enum=['A', 'B']),
        ),
        # Test Case 5: Array of strings
        (
            {'type': genai_types.Type.ARRAY, 'items': {'type': 'STRING'}},
            None,
            genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
            ),
        ),
        # Test Case 6: Empty object
        (
            {'type': 'OBJECT', 'properties': {}},
            None,
            genai_types.Schema(type=genai_types.Type.OBJECT, properties={}),
        ),
        # Test Case 7: Object with simple properties
        (
            {
                'type': 'OBJECT',
                'properties': {
                    'prop1': {'type': 'STRING'},
                    'prop2': {'type': 'NUMBER', 'description': 'Numeric field'},
                },
            },
            None,
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'prop1': genai_types.Schema(type=genai_types.Type.STRING),
                    'prop2': genai_types.Schema(type=genai_types.Type.NUMBER, description='Numeric field'),
                },
            ),
        ),
        # Test Case 8: Object with nested $ref
        (
            {
                'type': 'OBJECT',
                'properties': {'user': {'$ref': '#/$defs/User'}},
                '$defs': {'User': {'type': 'OBJECT', 'properties': {'name': {'type': 'STRING'}}}},
            },
            None,  # defs will be picked from input_schema['$defs']
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'user': genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={'name': genai_types.Schema(type=genai_types.Type.STRING)},
                    )
                },
            ),
        ),
        # Test Case 9: Object with nested $ref and existing defs
        (
            {
                'type': 'OBJECT',
                'properties': {'address': {'$ref': '#/$defs/Address'}},
            },
            {'Address': {'type': 'OBJECT', 'properties': {'street': {'type': 'STRING'}}}},
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'address': genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        properties={'street': genai_types.Schema(type=genai_types.Type.STRING)},
                    )
                },
            ),
        ),
        # Test Case 10: Object with $ref and description at the $ref level
        (
            {
                'type': 'OBJECT',
                'properties': {
                    'item': {
                        '$ref': '#/$defs/Item',
                        'description': 'A referenced item description',
                    }
                },
                '$defs': {'Item': {'type': 'STRING'}},
            },
            None,
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'item': genai_types.Schema(
                        type=genai_types.Type.STRING, description='A referenced item description'
                    )
                },
            ),
        ),
        # Test Case 11: Object with $ref at list field
        (
            {
                '$defs': {
                    'Product': {
                        'properties': {
                            'product_name': {
                                'title': 'Product Name',
                                'type': 'string',
                            },
                        },
                        'required': ['product_name'],
                        'title': 'Product',
                        'type': 'object',
                    },
                },
                'properties': {
                    'products': {
                        'items': {'$ref': '#/$defs/Product'},
                        'title': 'Products',
                        'type': 'array',
                    },
                },
                'required': ['products'],
                'title': 'Store',
                'type': 'object',
            },
            None,
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'products': genai_types.Schema(
                        items=genai_types.Schema(
                            properties={
                                'product_name': genai_types.Schema(
                                    type=genai_types.Type.STRING,
                                ),
                            },
                            required=['product_name'],
                            type=genai_types.Type.OBJECT,
                        ),
                        type=genai_types.Type.ARRAY,
                    ),
                },
                required=['products'],
            ),
        ),
    ],
)
def test_gemini_model__convert_schema_property(
    input_schema: dict[str, object] | None,
    defs: dict[str, object] | None,
    expected_schema: genai_types.Schema | None,
    gemini_model_instance: GeminiModel,
) -> None:
    """Unit tests for  GeminiModel._convert_schema_property with various valid schema inputs."""
    result_schema = gemini_model_instance._convert_schema_property(input_schema, defs)

    if expected_schema is None:
        assert result_schema is None
    else:

        def compare_schemas(s1: genai_types.Schema, s2: genai_types.Schema) -> None:
            assert s1.description == s2.description
            assert s1.required == s2.required
            assert s1.type == s2.type
            assert s1.enum == s2.enum

            if s1.items or s2.items:
                assert s1.items is not None and s2.items is not None
                compare_schemas(s1.items, s2.items)
            else:
                assert s1.items is None and s2.items is None

            if s1.properties or s2.properties:
                assert s1.properties is not None and s2.properties is not None
                assert set(s1.properties.keys()) == set(s2.properties.keys())
                for key in s1.properties:
                    compare_schemas(s1.properties[key], s2.properties[key])
            else:
                s1_props_len = len(s1.properties) if s1.properties else 0
                s2_props_len = len(s2.properties) if s2.properties else 0
                assert s1_props_len == 0 and s2_props_len == 0

        assert result_schema is not None
        compare_schemas(result_schema, expected_schema)


@pytest.mark.parametrize(
    'input_schema, defs',
    [
        # Test Case 11: Unresolvable $ref
        (
            {'type': 'OBJECT', 'properties': {'user': {'$ref': '#/$defs/NonExistent'}}},
            {'$defs': {'SomeOtherDef': {'type': 'STRING'}}},
        ),
        # Test Case 12: $ref with missing defs dict
        (
            {'type': 'OBJECT', 'properties': {'user': {'$ref': '#/$defs/NonExistent'}}},
            None,
        ),
    ],
)
def test_gemini_model__convert_schema_property_raises_exception(
    input_schema: dict[str, object],
    defs: dict[str, object] | None,
    gemini_model_instance: GeminiModel,
) -> None:
    """Test GeminiModel._convert_schema_property raises an exception for unresolvable schemas."""
    with pytest.raises(ValueError, match=r'Failed to resolve schema for .*'):
        gemini_model_instance._convert_schema_property(input_schema, defs)


@pytest.mark.asyncio
@patch(
    'genkit.plugins.google_genai.models.context_caching.utils.generate_cache_key',
    new_callable=MagicMock,
)
@patch(
    'genkit.plugins.google_genai.models.context_caching.utils.validate_context_cache_request',
    new_callable=MagicMock,
)
@pytest.mark.parametrize(
    'cache_key',
    [
        'key_not_cached',
        'key1',
    ],
)
async def test_gemini_model__retrieve_cached_content(
    mock_generate_cache_key: MagicMock,
    mock_validate_context_cache_request: MagicMock,
    cache_key: str,
    gemini_model_instance: GeminiModel,
) -> None:
    """Unit tests for GeminiModel._retrieve_cached_content."""
    # Mock cache utils
    mock_generate_cache_key.return_value = cache_key
    mock_validate_context_cache_request.return_value = None

    # Mock pager object
    class MockPage(AsyncMock):
        display_name: str

    async_mock_list = AsyncMock()
    mock_client = MagicMock()
    mock_client.aio.caches.list = async_mock_list

    async_mock_list.__aiter__.return_value = [MockPage(display_name='key1'), MockPage(display_name='key2')]

    # Mock update and create cache methods of google genai
    async_cache = AsyncMock()
    async_cache.return_value = genai_types.CachedContent()
    mock_client.aio.caches.update = async_cache
    mock_client.aio.caches.create = async_cache

    gemini_model_instance._client = mock_client

    request = GenerateRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text='request text')),
                ],
            ),
        ]
    )

    cache = await gemini_model_instance._retrieve_cached_content(
        request=request,
        model_name='gemini-1.5-flash-001',
        cache_config={},
        contents=[],
    )

    assert isinstance(cache, genai_types.CachedContent)
