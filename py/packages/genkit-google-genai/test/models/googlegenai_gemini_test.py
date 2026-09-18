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
from typing import Annotated, Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

if sys.version_info < (3, 11):
    from strenum import StrEnum
else:
    from enum import StrEnum

import pytest
from genkit_google_genai.models.gemini import (
    DEFAULT_SUPPORTS_MODEL,
    GeminiConfigSchema,
    GeminiImageConfigSchema,
    GeminiModel,
    GeminiTtsConfigSchema,
    GemmaConfigSchema,
    GoogleAIGeminiVersion,
    VertexAIGeminiVersion,
    _to_finish_reason,
    get_model_config_schema,
    google_model_info,
    is_image_model,
    is_tts_model,
)
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, Field
from pytest_mock import MockerFixture

from genkit import (
    ActionRunContext,
    Constrained,
    FinishReason,
    GenkitError,
    MediaPart,
    Message,
    ModelInfo,
    ModelRequest,
    ModelResponse,
    Part,
    Role,
    Supports,
    TextPart,
    ToolDefinition,
)
from genkit._core._model import OutputConfig
from genkit.plugin_api import to_json_schema

ALL_VERSIONS = list(GoogleAIGeminiVersion) + list(VertexAIGeminiVersion)
IMAGE_GENERATION_VERSIONS = [GoogleAIGeminiVersion.GEMINI_2_5_FLASH]


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in ALL_VERSIONS])
async def test_generate_text_response(mocker: MockerFixture, version: str) -> None:
    """Test the generate method for text responses."""
    response_text = 'request answer'
    request_text = 'response question'

    request = ModelRequest(
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
    assert isinstance(response, ModelResponse)
    assert response.message is not None
    assert response.message.content[0].root.text == response_text


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in ALL_VERSIONS])
async def test_generate_stream_text_response(mocker: MockerFixture, version: str) -> None:
    """Test the generate method for text responses."""
    response_text = 'request answer'
    request_text = 'response question'

    request = ModelRequest(
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

    ctx = ActionRunContext(streaming_callback=on_chunk_mock)
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
    assert isinstance(response, ModelResponse)
    assert response.message is not None
    assert response.message.content == []


@pytest.mark.asyncio
async def test_generate_stream_captures_finish_reason_and_usage(mocker: MockerFixture) -> None:
    """Test that streaming generate captures trailing finish_reason and usage_metadata."""
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ]
    )
    cand_1 = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='Hello')]))
    resp_1 = genai.types.GenerateContentResponse(candidates=[cand_1])

    cand_2 = genai.types.Candidate(
        content=genai.types.Content(parts=[genai.types.Part(text=' world!')]),
        finish_reason=genai.types.FinishReason.STOP,
    )
    usage_meta = genai.types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10,
        candidates_token_count=5,
        total_token_count=15,
    )
    resp_2 = genai.types.GenerateContentResponse(candidates=[cand_2], usage_metadata=usage_meta)

    googleai_client_mock = mocker.AsyncMock()

    async def mock_stream() -> Any:  # noqa: ANN401
        for r in [resp_1, resp_2]:
            yield r

    googleai_client_mock.aio.models.generate_content_stream.return_value = mock_stream()

    on_chunk_mock = mocker.MagicMock()
    gemini = GeminiModel('gemini-2.5-flash', googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=on_chunk_mock)

    response = await gemini.generate(request, ctx)
    assert response.finish_reason == FinishReason.STOP
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 15
    assert on_chunk_mock.call_count == 2


@pytest.mark.asyncio
async def test_generate_stream_without_finish_reason(mocker: MockerFixture) -> None:
    """Test that streaming generate defaults to FinishReason.UNKNOWN when no chunk carries a finish_reason."""
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='hi'))],
            ),
        ]
    )
    cand_1 = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='Hello')]))
    resp_1 = genai.types.GenerateContentResponse(candidates=[cand_1])

    googleai_client_mock = mocker.AsyncMock()

    async def mock_stream() -> Any:  # noqa: ANN401
        yield resp_1

    googleai_client_mock.aio.models.generate_content_stream.return_value = mock_stream()

    on_chunk_mock = mocker.MagicMock()
    gemini = GeminiModel('gemini-2.5-flash', googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=on_chunk_mock)

    response = await gemini.generate(request, ctx)
    assert response.finish_reason == FinishReason.UNKNOWN


@pytest.mark.asyncio
@pytest.mark.parametrize('version', [x for x in IMAGE_GENERATION_VERSIONS])
async def test_generate_media_response(mocker: MockerFixture, version: str) -> None:
    """Test generate method for media responses."""
    request_text = 'response question'
    response_byte_string = b'\x89PNG\r\n\x1a\n'
    response_mimetype = 'image/png'
    modalities = ['Text', 'Image']

    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=TextPart(text=request_text)),
                ],
            ),
        ],
        config=GeminiConfigSchema.model_validate({'response_modalities': modalities}),
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
    assert isinstance(response, ModelResponse)
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
        baz: list[str] | None = Field(None, description='baz field')

    assert gemini._convert_schema_property(to_json_schema(Simple)) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        title='Simple',
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                title='Foo',
                description='foo field',
            ),
            'bar': genai_types.Schema(
                type=genai_types.Type.INTEGER,
                title='Bar',
                description='bar field',
            ),
            'baz': genai_types.Schema(
                type=genai_types.Type.ARRAY,
                title='Baz',
                description='baz field',
                nullable=True,
                items=genai_types.Schema(type=genai_types.Type.STRING),
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
        title='WithNested',
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                title='Foo',
                description='foo field',
            ),
            'bar': genai_types.Schema(
                type=genai_types.Type.OBJECT,
                title='Nested',
                description='bar field',
                properties={
                    'baz': genai_types.Schema(
                        type=genai_types.Type.INTEGER,
                        title='Baz',
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
        title='WitEnum',
        properties={
            'foo': genai_types.Schema(
                type=genai_types.Type.STRING,
                title='TestEnum',
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
    version = GoogleAIGeminiVersion.GEMINI_2_5_FLASH

    request = ModelRequest(
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
    assert isinstance(response, ModelResponse)
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
        (
            'gemini-3-pro-image',
            ModelInfo(
                label='Google AI - Gemini 3 Pro Image',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            'gemini-3.1-flash-image',
            ModelInfo(
                label='Google AI - Gemini 3.1 Flash Image',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            'gemini-3.1-flash-image-preview',
            ModelInfo(
                label='Google AI - Gemini 3.1 Flash Image Preview',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            'gemini-3-pro-image-preview',
            ModelInfo(
                label='Google AI - Gemini 3 Pro Image Preview',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            'gemini-2.5-flash-image',
            ModelInfo(
                label='Google AI - Gemini 2.5 Flash Image',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            'gemini-2.5-flash-image-preview',
            ModelInfo(
                label='Google AI - Gemini 2.5 Flash Image Preview',
                supports=Supports(
                    multiturn=True,
                    media=True,
                    tools=True,
                    tool_choice=True,
                    system_role=True,
                    constrained=Constrained.ALL,
                ),
            ),
        ),
        (
            # An unregistered image model falls back to GENERIC_IMAGE_MODEL via
            # is_image_model(). That fallback must stay restrictive (single-turn,
            # no tools, output=['media']) because pure image-generation models are
            # not conversational/tool-capable.
            'gemini-2.0-flash-preview-image-generation',
            ModelInfo(
                label='Google AI - Gemini Image',
                supports=Supports(
                    multiturn=False,
                    media=True,
                    tools=False,
                    tool_choice=False,
                    system_role=True,
                    constrained=Constrained.ALL,
                    output=['media'],
                ),
            ),
        ),
        (
            'gemini-2.5-flash-preview-tts',
            ModelInfo(
                label='Google AI - Gemini TTS',
                supports=Supports(
                    multiturn=False,
                    media=False,
                    tools=False,
                    tool_choice=False,
                    system_role=False,
                    constrained=Constrained.NONE,
                    output=['media'],
                ),
            ),
        ),
    ],
)
def test_google_model_info(input: str, expected: ModelInfo) -> None:
    """Tests for google_model_info."""
    model_info = google_model_info(input)

    assert model_info == expected


@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-3.1-pro-preview',
        'gemini-3.1-pro-preview-customtools',
        'gemini-3.1-flash-lite-preview',
    ],
)
def test_gemini_3_1_models_register_real_capabilities(model_name: str) -> None:
    """Gemini 3.1 text models resolve to explicit ModelInfo, not the generic fallback.

    The generic fallback (DEFAULT_SUPPORTS_MODEL) leaves ``output`` unset, so asserting
    ``output == ['text', 'json']`` alongside tools/constrained proves these names are
    registered with real capability metadata matching the JS/Go registries.
    """
    model_info = google_model_info(model_name)

    assert model_info.label is not None
    assert model_info.label.startswith('Google AI - Gemini 3.1')
    assert model_info.supports is not None
    assert model_info.supports.tools is True
    assert model_info.supports.tool_choice is True
    assert model_info.supports.constrained == Constrained.ALL
    assert model_info.supports.output == ['text', 'json']


@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-3.1-pro-preview',
        'gemini-3.1-flash-lite',
        'gemini-3.5-flash',
        'gemini-3.6-flash',
        'gemini-3.7-flash',
    ],
)
def test_vertexai_gemini_3_x_text_models_register_real_capabilities(model_name: str) -> None:
    """VertexAI Gemini 3.1/3.5 text models resolve to explicit ModelInfo, not the generic fallback.

    These names are now first-class ``VertexAIGeminiVersion`` members. The generic fallback
    (DEFAULT_SUPPORTS_MODEL) leaves ``output`` unset, so asserting ``output == ['text', 'json']``
    alongside tools/constrained proves they carry real capability metadata matching the JS Vertex
    registry, not the fallback.
    """
    model_info = google_model_info(model_name)

    assert model_info.supports is not None
    assert model_info.supports.tools is True
    assert model_info.supports.tool_choice is True
    assert model_info.supports.constrained == Constrained.ALL
    assert model_info.supports.output == ['text', 'json']


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


@patch('genkit_google_genai.models.gemini.GeminiModel._create_tool')
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

    request = ModelRequest(
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


@patch('genkit_google_genai.models.gemini.GeminiModel._convert_schema_property')
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


def test_gemini_model__convert_schema_property_keeps_keywords(gemini_model_instance: GeminiModel) -> None:
    """anyOf, type lists, format, title, default, bounds and propertyOrdering survive conversion.

    A type list without null is not nullable, and an optional untyped property
    is left out.
    """
    schema: dict[str, object] = {
        'type': 'object',
        'title': 'Thing',
        'additionalProperties': False,
        'propertyOrdering': ['count', 'label', 'kind', 'ratio', 'tags'],
        'required': ['count'],
        'properties': {
            'count': {'type': 'integer', 'title': 'Count', 'default': 1, 'minimum': 0, 'maximum': 10},
            'label': {'type': ['string', 'null'], 'format': 'date-time'},
            'pick': {'type': ['string', 'integer']},
            'kind': {'anyOf': [{'type': 'string'}, {'type': 'integer'}], 'description': 'string or int'},
            'ratio': {'anyOf': [{'type': 'number'}, {'type': 'null'}], 'title': 'Ratio'},
            'tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
            'anything': {'title': 'Anything'},
        },
    }

    assert gemini_model_instance._convert_schema_property(schema) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        title='Thing',
        property_ordering=['count', 'label', 'kind', 'ratio', 'tags'],
        required=['count'],
        properties={
            'count': genai_types.Schema(
                type=genai_types.Type.INTEGER, title='Count', default=1, minimum=0.0, maximum=10.0
            ),
            'label': genai_types.Schema(type=genai_types.Type.STRING, format='date-time', nullable=True),
            'pick': genai_types.Schema(type=genai_types.Type.STRING),
            'kind': genai_types.Schema(
                any_of=[
                    genai_types.Schema(type=genai_types.Type.STRING),
                    genai_types.Schema(type=genai_types.Type.INTEGER),
                ],
                description='string or int',
            ),
            'ratio': genai_types.Schema(type=genai_types.Type.NUMBER, title='Ratio', nullable=True),
            'tags': genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(type=genai_types.Type.STRING),
                min_items=1,
                max_items=3,
            ),
        },
    )
    assert gemini_model_instance._convert_schema_property({}) is None


@pytest.mark.parametrize(
    'input_schema, keyword',
    [
        ({'type': 'integer', 'minimum': 'zero'}, 'minimum'),
        ({'type': 'object', 'propertyOrdering': 'ab'}, 'propertyOrdering'),
    ],
)
def test_gemini_model__convert_schema_property_rejects_bad_keyword_value(
    input_schema: dict[str, object],
    keyword: str,
    gemini_model_instance: GeminiModel,
) -> None:
    """A keyword value that cannot be coerced raises INVALID_ARGUMENT naming the keyword."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert keyword in exc_info.value.original_message


@pytest.mark.parametrize('raw_type', ['bogus', None, 123])
def test_gemini_model__convert_schema_property_rejects_unsupported_type(
    raw_type: object,
    gemini_model_instance: GeminiModel,
) -> None:
    """A type the Gemini Schema cannot express raises INVALID_ARGUMENT instead of going out as written."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property({'type': raw_type})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'is not supported' in exc_info.value.original_message


def test_gemini_model__convert_schema_property_discriminated_union(gemini_model_instance: GeminiModel) -> None:
    """A pydantic discriminated union (oneOf of $refs with const tags) becomes any_of of enum-tagged objects."""

    class Cat(BaseModel):
        kind: Literal['cat']
        meows: int

    class Dog(BaseModel):
        kind: Literal['dog']
        barks: bool

    class Pets(BaseModel):
        pet: Annotated[Cat | Dog, Field(discriminator='kind')]

    assert gemini_model_instance._convert_schema_property(to_json_schema(Pets)) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        title='Pets',
        required=['pet'],
        properties={
            'pet': genai_types.Schema(
                title='Pet',
                any_of=[
                    genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        title='Cat',
                        required=['kind', 'meows'],
                        properties={
                            'kind': genai_types.Schema(type=genai_types.Type.STRING, title='Kind', enum=['cat']),
                            'meows': genai_types.Schema(type=genai_types.Type.INTEGER, title='Meows'),
                        },
                    ),
                    genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        title='Dog',
                        required=['kind', 'barks'],
                        properties={
                            'kind': genai_types.Schema(type=genai_types.Type.STRING, title='Kind', enum=['dog']),
                            'barks': genai_types.Schema(type=genai_types.Type.BOOLEAN, title='Barks'),
                        },
                    ),
                ],
            ),
        },
    )


def test_gemini_model__convert_schema_property_non_string_const_keeps_type(gemini_model_instance: GeminiModel) -> None:
    """A const the Gemini enum cannot hold leaves the property typed but unconstrained."""
    assert gemini_model_instance._convert_schema_property({'type': 'integer', 'const': 1}) == genai_types.Schema(
        type=genai_types.Type.INTEGER
    )


def test_gemini_model__convert_schema_property_rejects_required_untyped_property(
    gemini_model_instance: GeminiModel,
) -> None:
    """A required property with no type cannot be left out, so it raises naming the property."""

    class Payload(BaseModel):
        data: Any
        note: Any = None

    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(to_json_schema(Payload))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'data' in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema',
    [
        {'allOf': [{'type': 'string'}]},
        {'type': 'object', 'properties': {'x': {'allOf': [{'type': 'string'}]}}},
        {'type': 'array', 'items': {'not': {'type': 'null'}}},
    ],
)
def test_gemini_model__convert_schema_property_rejects_untyped_constraint(
    input_schema: dict[str, object],
    gemini_model_instance: GeminiModel,
) -> None:
    """A node that constrains without a type raises instead of dropping out of its parent."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'no type' in exc_info.value.original_message


@pytest.mark.asyncio
@patch(
    'genkit_google_genai.models.gemini.generate_cache_key',
    new_callable=MagicMock,
)
@patch(
    'genkit_google_genai.models.gemini.validate_context_cache_request',
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

    request = ModelRequest(
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


# ---------------------------------------------------------------------------
# Config normalization
#
# Plugin-specific keys like ``code_execution`` carry a camelCase alias
# (``codeExecution``) on the wire so that the Python and JS SDKs share the
# same JSON. Callers can hand the plugin three different shapes for the same
# logical config and we have to fold all of them onto the canonical
# snake_case field name before downstream translation runs. These tests pin
# that contract so a future refactor can't quietly let an alias-form key
# leak through to the strict ``GenerateContentConfig``.
# ---------------------------------------------------------------------------


def test_gemini_model__normalize_config_dumps_schema_instance(
    gemini_model_instance: GeminiModel,
) -> None:
    """A typed family config dumps to snake_case for the SDK."""
    dumped = gemini_model_instance._normalize_config_to_dict(
        GeminiConfigSchema.model_validate({'code_execution': True})
    )

    assert dumped == {'code_execution': True}


def test_gemini_model__normalize_config_rejects_raw_dicts(
    gemini_model_instance: GeminiModel,
) -> None:
    """A dict at the dump leaf means Action never produced the family instance."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._normalize_config_to_dict({'code_execution': True})  # type: ignore[arg-type]

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert gemini_model_instance._version in str(exc_info.value)


@pytest.mark.asyncio
async def test_gemini_model__code_execution_translates_to_tool(
    gemini_model_instance: GeminiModel,
) -> None:
    """A typed ``code_execution`` flag becomes a tool and is not leaked to the SDK."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=GeminiConfigSchema.model_validate({'code_execution': True}),
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.tools is not None
    code_exec_tools = [t for t in cfg.tools if isinstance(t, genai_types.Tool) and t.code_execution is not None]
    assert len(code_exec_tools) == 1
    assert 'codeExecution' not in cfg.model_dump(exclude_none=True)
    assert 'code_execution' not in cfg.model_dump(exclude_none=True)


def test_gemini_model__normalize_config_dumps_gemma_instance() -> None:
    """Gemma's relaxed temperature dumps through without re-validation."""
    gemma_model = GeminiModel(version='gemma-2-27b-it', client=MagicMock(spec=genai.Client))

    dumped = gemma_model._normalize_config_to_dict(GemmaConfigSchema(temperature=3.0))

    assert dumped == {'temperature': 3.0}


@pytest.mark.asyncio
async def test_gemini_model__unknown_extra_rides_on_extra_body(
    gemini_model_instance: GeminiModel,
) -> None:
    """Leftover keys ride on extra_body so a newly supported field still reaches the API."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=GeminiConfigSchema.model_validate({'temperature': 0.5, 'fooBar': 1}),
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.temperature == 0.5
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body == {'generationConfig': {'fooBar': 1}}


# A pydantic-style schema exercising anyOf, type lists, $ref, $defs and numeric bounds.
_STRUCTURED_SCHEMA: dict[str, Any] = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    'type': 'object',
    'title': 'Thing',
    'properties': {
        'a': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None},
        'n': {'type': 'integer', 'minimum': 1, 'maximum': 5},
        't': {'type': ['string', 'null'], 'format': 'date-time'},
        'p': {'$ref': '#/$defs/P'},
    },
    '$defs': {'P': {'type': 'string', 'enum': ['x', 'y']}},
    'required': ['n'],
    'additionalProperties': False,
}

_STRUCTURED_RESPONSE_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    title='Thing',
    required=['n'],
    properties={
        'a': genai_types.Schema(type=genai_types.Type.STRING, nullable=True),
        'n': genai_types.Schema(type=genai_types.Type.INTEGER, minimum=1.0, maximum=5.0),
        't': genai_types.Schema(type=genai_types.Type.STRING, format='date-time', nullable=True),
        'p': genai_types.Schema(type=genai_types.Type.STRING, enum=['x', 'y']),
    },
)


def _output_request(
    *,
    format: str | None = None,
    json_schema: dict[str, Any] | None = None,
    constrained: bool | None = None,
    content_type: str | None = None,
) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        output=OutputConfig(format=format, json_schema=json_schema, constrained=constrained, content_type=content_type),
    )


@pytest.mark.asyncio
async def test_gemini_model__json_constrained_sends_schema_verbatim(gemini_model_instance: GeminiModel) -> None:
    """Constrained JSON output sends the JSON Schema untouched as response_json_schema."""
    request = _output_request(
        format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True, content_type='application/json'
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_json_schema == _STRUCTURED_SCHEMA
    assert cfg.response_schema is None


@pytest.mark.asyncio
async def test_gemini_model__tts_json_output_skips_constrained_config() -> None:
    """A TTS model receives neither response_mime_type nor a schema for JSON output."""
    model = GeminiModel(version='gemini-2.5-flash-preview-tts', client=MagicMock(spec=genai.Client))
    request = _output_request(format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True)

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type is None
    assert cfg.response_schema is None
    assert cfg.response_json_schema is None


@pytest.mark.asyncio
async def test_gemini_model__array_format_is_json_mode(gemini_model_instance: GeminiModel) -> None:
    """The array format reaches the API as JSON mode via its application/json content type."""
    schema = {'type': 'array', 'items': {'type': 'string'}}
    request = _output_request(format='array', json_schema=schema, constrained=True, content_type='application/json')

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_json_schema == schema


@pytest.mark.asyncio
async def test_gemini_model__content_type_alone_turns_on_json_mode(gemini_model_instance: GeminiModel) -> None:
    """A request carrying only the application/json content type still reaches JSON mode."""
    request = _output_request(content_type='application/json')

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'


@pytest.mark.asyncio
async def test_gemini_model__json_unconstrained_sets_mime_type_only(gemini_model_instance: GeminiModel) -> None:
    """Unconstrained JSON output turns on JSON mode without a schema."""
    request = _output_request(format='json', json_schema=_STRUCTURED_SCHEMA, constrained=False)

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_json_schema is None
    assert cfg.response_schema is None


@pytest.mark.asyncio
async def test_gemini_model__legacy_response_schema_sends_converted_schema() -> None:
    """With legacy_response_schema the converted schema goes out as response_schema."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=True)
    request = _output_request(format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True)

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_json_schema is None
    assert cfg.response_schema == _STRUCTURED_RESPONSE_SCHEMA


@pytest.mark.asyncio
async def test_gemini_model__legacy_untyped_output_schema_raises() -> None:
    """A schema with no type cannot be sent as response_schema and fails loudly."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=True)
    request = _output_request(format='json', json_schema={'description': 'anything'}, constrained=True)

    with pytest.raises(GenkitError) as exc_info:
        await model._genkit_to_googleai_cfg(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'response_schema' in exc_info.value.original_message


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_response_schema', [False, True])
@pytest.mark.parametrize('schema_type', ['string', 'enum'])
async def test_gemini_model__enum_output_sets_text_x_enum(schema_type: str, legacy_response_schema: bool) -> None:
    """Constrained enum output sends text/x.enum with a string schema listing the values."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=legacy_response_schema)
    request = _output_request(
        format='enum', json_schema={'type': schema_type, 'enum': ['a', 'b']}, constrained=True, content_type='text/enum'
    )

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'text/x.enum'
    assert cfg.response_schema == genai_types.Schema(type=genai_types.Type.STRING, enum=['a', 'b'])
    assert cfg.response_json_schema is None


@pytest.mark.asyncio
async def test_gemini_model__constrained_output_skipped_without_model_support() -> None:
    """A no-tools constrained model with tools attached gets no MIME type and no schema."""
    model = GeminiModel('gemini-2.5-flash-lite', MagicMock(spec=genai.Client))
    request = _output_request(format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True)
    request.tools = [ToolDefinition(name='t', description='d', input_schema={'type': 'object'})]

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type is None
    assert cfg.response_json_schema is None
    assert cfg.response_schema is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('response_mime_type', 'application/json'),
        ('response_schema', {'type': 'object'}),
        ('response_json_schema', {'type': 'object'}),
        ('responseMimeType', 'application/json'),
        ('responseSchema', {'type': 'object'}),
        ('responseJsonSchema', {'type': 'object'}),
    ],
)
async def test_gemini_model__config_output_keys_are_rejected(
    gemini_model_instance: GeminiModel, key: str, value: object
) -> None:
    """Config-level response fields are refused; output settings own them."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
        config=GeminiConfigSchema.model_validate({key: value}),
    )

    with pytest.raises(GenkitError) as exc_info:
        await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert key in exc_info.value.original_message
    assert 'output_schema=' in exc_info.value.original_message


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_response_schema', [False, True])
async def test_generate_json_output_pins_request_body(mocker: MockerFixture, legacy_response_schema: bool) -> None:
    """The client receives JSON mode plus the schema in one response field, and nothing else."""
    version = GoogleAIGeminiVersion.GEMINI_2_5_FLASH
    request = _output_request(
        format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True, content_type='application/json'
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='{"n": 1}')]))
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.return_value = genai.types.GenerateContentResponse(
        candidates=[candidate]
    )
    gemini = GeminiModel(version, googleai_client_mock, legacy_response_schema=legacy_response_schema)

    response = await gemini.generate(request, ActionRunContext())

    if legacy_response_schema:
        expected_config = genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=_STRUCTURED_RESPONSE_SCHEMA,
        )
    else:
        expected_config = genai.types.GenerateContentConfig(
            response_mime_type='application/json',
            response_json_schema=_STRUCTURED_SCHEMA,
        )
    googleai_client_mock.aio.models.generate_content.assert_called_once_with(
        model=version,
        contents=[genai.types.Content(parts=[genai.types.Part(text='hi')], role=Role.USER)],
        config=expected_config,
    )
    assert response.message is not None
    assert response.message.content[0].root.text == '{"n": 1}'


@pytest.mark.parametrize(
    ('version', 'expected_schema'),
    [
        ('gemini-2.5-flash-preview-tts', GeminiTtsConfigSchema),
        ('gemini-2.0-flash-preview-image-generation', GeminiImageConfigSchema),
        ('gemini-3-pro-image', GeminiImageConfigSchema),
        ('gemini-3.1-flash-image', GeminiImageConfigSchema),
        ('gemini-3.1-flash-image-preview', GeminiImageConfigSchema),
        ('gemini-3-pro-image-preview', GeminiImageConfigSchema),
        ('gemini-2.5-flash-image', GeminiImageConfigSchema),
        ('gemini-2.5-flash-image-preview', GeminiImageConfigSchema),
        ('gemma-2-27b-it', GemmaConfigSchema),
        ('gemini-2.0-flash-001', GeminiConfigSchema),
    ],
)
def test_get_model_config_schema_routes_by_model_family(
    version: str,
    expected_schema: type[GeminiConfigSchema],
) -> None:
    """Family routing lives on the name check, not a runtime re-validate."""
    assert get_model_config_schema(version) is expected_schema


@pytest.mark.asyncio
async def test_gemini_model__build_messages_maps_tool_role_to_user(
    gemini_model_instance: GeminiModel,
) -> None:
    """Messages with Role.TOOL are mapped to 'user' in Gemini request Content."""
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(root=TextPart(text='What is the weather in Seattle?'))]),
            Message(
                role=Role.MODEL,
                content=[Part(root=TextPart(text='I will check.'))],
            ),
            Message(
                role=Role.TOOL,
                content=[Part(root=TextPart(text='Sunny, 72°F in Seattle'))],
            ),
        ],
    )

    contents, cache = await gemini_model_instance._build_messages(request, model_name='gemini-2.5-flash')
    assert cache is None
    assert len(contents) == 3
    assert contents[0].role == 'user'
    assert contents[1].role == 'model'
    assert contents[2].role == 'user'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('code', 'status'),
    [
        (400, 'INVALID_ARGUMENT'),
        (503, 'UNAVAILABLE'),
    ],
)
async def test_streaming_generate_classifies_error_on_first_chunk(
    mocker: MockerFixture, code: int, status: str
) -> None:
    """The HTTP call is the first iteration, not the await that created the generator."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
    )

    async def failing_stream() -> Any:  # noqa: ANN401
        raise APIError(code, {'error': {'message': 'provider failed'}})
        if False:
            yield  # pragma: no cover

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content_stream.return_value = failing_stream()
    gemini = GeminiModel(GoogleAIGeminiVersion.GEMINI_2_5_FLASH, googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(request, ctx)
    assert raised.value.status == status


@pytest.mark.asyncio
async def test_streaming_generate_classifies_mid_stream_error(mocker: MockerFixture) -> None:
    """A 503 after the first chunk is still UNAVAILABLE so retry can wait it out."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
    )
    first = genai.types.GenerateContentResponse(
        candidates=[genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='Hello')]))]
    )

    async def mid_stream_fail() -> Any:  # noqa: ANN401
        yield first
        raise APIError(503, {'error': {'message': 'overloaded'}})

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content_stream.return_value = mid_stream_fail()
    gemini = GeminiModel(GoogleAIGeminiVersion.GEMINI_2_5_FLASH, googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(request, ctx)
    assert raised.value.status == 'UNAVAILABLE'


@pytest.mark.asyncio
async def test_generate_classifies_503_as_unavailable(mocker: MockerFixture) -> None:
    """A provider 503 must stay retryable, not collapse to INTERNAL."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='hi'))])],
    )
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.side_effect = APIError(503, {'error': {'message': 'overloaded'}})
    gemini = GeminiModel(GoogleAIGeminiVersion.GEMINI_2_5_FLASH, googleai_client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(request, ActionRunContext())
    assert raised.value.status == 'UNAVAILABLE'


def test_to_finish_reason_image_policy() -> None:
    """Image-policy refusals stay blocked so a leftover is not labeled a schema miss."""
    assert _to_finish_reason('IMAGE_SAFETY') == FinishReason.BLOCKED
    assert _to_finish_reason('IMAGE_PROHIBITED_CONTENT') == FinishReason.BLOCKED
    assert _to_finish_reason('IMAGE_RECITATION') == FinishReason.BLOCKED


def test_to_finish_reason_image_other_and_unexpected_tool() -> None:
    """No-image / unspecified image stop / bad tool call are other, not unknown."""
    assert _to_finish_reason('NO_IMAGE') == FinishReason.OTHER
    assert _to_finish_reason('IMAGE_OTHER') == FinishReason.OTHER
    assert _to_finish_reason('UNEXPECTED_TOOL_CALL') == FinishReason.OTHER
