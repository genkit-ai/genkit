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
from typing import Any, cast, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai._models import _gemini
from genkit_google_genai._models._gemini import (
    DEFAULT_SUPPORTS_MODEL,
    GeminiConfig,
    GeminiImageConfig,
    GeminiModel,
    GeminiTtsConfig,
    GemmaConfig,
    KnownGemini,
    KnownGeminiImage,
    KnownGeminiTts,
    KnownGemma,
    SpeechConfig,
    _to_finish_reason,
    get_model_config_schema,
    google_model_info,
    is_image_model,
    is_tts_model,
)
from genkit_google_genai._models._utils import ToolWire
from google import genai
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from google.genai import types as genai_types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError
from pytest_mock import MockerFixture

from genkit import ActionRunContext, FinishReason, GenkitError, Message, ModelResponse, Part, Role
from genkit._core._compat import StrEnum
from genkit.model import Constrained, ModelInfo, ModelRequest, OutputConfig, Supports, ToolDefinition
from genkit.plugin_api import to_json_schema

ALL_VERSIONS = sorted({
    *get_args(KnownGemini),
    *get_args(KnownGeminiTts),
    *get_args(KnownGeminiImage),
    *get_args(KnownGemma),
})
IMAGE_GENERATION_VERSIONS = ['gemini-2.5-flash']


def _kore_speech_config() -> genai.types.SpeechConfig:
    return genai.types.SpeechConfig(
        voice_config=genai.types.VoiceConfig(prebuilt_voice_config=genai.types.PrebuiltVoiceConfig(voice_name='Kore'))
    )


def _expected_gemini_api_tts_config(version: str) -> genai.types.GenerateContentConfig:
    # On the Gemini API only 3.1 TTS gets the default voice.
    if version == 'gemini-3.1-flash-tts-preview':
        return genai.types.GenerateContentConfig(response_modalities=['AUDIO'], speech_config=_kore_speech_config())
    return genai.types.GenerateContentConfig(response_modalities=['AUDIO'])


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
                    Part.from_text(request_text),
                ],
            ),
        ]
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text=response_text)]))
    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.vertexai = False
    googleai_client_mock.aio.models.generate_content.return_value = resp

    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ActionRunContext()
    response = await gemini.generate(request, ctx)

    # Determine expected config based on model type
    if is_tts_model(version):
        expected_config = _expected_gemini_api_tts_config(version)
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
    assert response.message.content[0].text == response_text


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
                    Part.from_text(request_text),
                ],
            ),
        ]
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text=response_text)]))

    resp = genai.types.GenerateContentResponse(candidates=[candidate])

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.vertexai = False
    googleai_client_mock.aio.models.generate_content_stream.__aiter__.side_effect = [resp]
    on_chunk_mock = mocker.MagicMock()
    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ActionRunContext(streaming_callback=on_chunk_mock)
    response = await gemini.generate(request, ctx)

    # Determine expected config based on model type
    if is_tts_model(version):
        expected_config = _expected_gemini_api_tts_config(version)
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
                content=[Part.from_text('hi')],
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
                content=[Part.from_text('hi')],
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
                    Part.from_text(request_text),
                ],
            ),
        ],
        config=GeminiConfig.model_validate({'response_modalities': modalities}),
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
    assert content.media is not None

    assert content.media.content_type == response_mimetype

    # Verify the data URL contains the correct base64-encoded content
    # Data URLs have format: data:<mimetype>;base64,<data>
    data_url = content.media.url
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
    version = 'gemini-2.5-flash'

    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part.from_text(request_text),
                ],
            ),
            Message(
                role=Role.SYSTEM,
                content=[
                    Part.from_text(system_instruction),
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
    assert response.message.content[0].text == response_text


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
                label='Google AI - Gemini 2.5 Flash Preview TTS',
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

    These names are registered in the Gemini catalog. The generic fallback
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


@pytest.mark.parametrize(
    ('model_name', 'expected_label'),
    [
        ('gemini-2.5-pro', 'Google AI - Gemini 2.5 Pro'),
        ('gemini-2.5-flash', 'Google AI - Gemini 2.5 Flash'),
        ('gemini-flash-lite-latest', 'Google AI - Gemini Flash Lite Latest'),
    ],
)
def test_stable_gemini_text_models_register_real_capabilities(model_name: str, expected_label: str) -> None:
    """Stable text ids resolve to their own ModelInfo, not the generic fallback.

    The fallback (DEFAULT_SUPPORTS_MODEL) leaves ``output`` unset and labels the id verbatim,
    so the label and ``output == ['text', 'json']`` together prove real metadata.
    """
    model_info = google_model_info(model_name)

    assert model_info.label == expected_label
    assert model_info.supports is not None
    assert model_info.supports.tools is True
    assert model_info.supports.tool_choice is True
    assert model_info.supports.system_role is True
    assert model_info.supports.constrained == Constrained.ALL
    assert model_info.supports.output == ['text', 'json']


@pytest.mark.parametrize(
    ('model_name', 'expected_label'),
    [
        ('gemini-2.5-flash-preview-tts', 'Google AI - Gemini 2.5 Flash Preview TTS'),
        ('gemini-2.5-pro-preview-tts', 'Google AI - Gemini 2.5 Pro Preview TTS'),
        ('gemini-3.1-flash-tts-preview', 'Google AI - Gemini 3.1 Flash TTS Preview'),
    ],
)
def test_tts_models_register_per_name_capabilities(model_name: str, expected_label: str) -> None:
    """Each TTS id carries its own label instead of sharing the generic TTS entry."""
    model_info = google_model_info(model_name)

    assert model_info.label == expected_label
    assert model_info.supports == Supports(
        multiturn=False,
        media=False,
        tools=False,
        tool_choice=False,
        system_role=False,
        constrained=Constrained.NONE,
        output=['media'],
    )
    assert get_model_config_schema(model_name) is GeminiTtsConfig


@pytest.mark.parametrize(
    'model_name',
    [
        'gemini-2.5-flash-preview-tts',
        'gemini-2.5-pro-preview-tts',
        'gemini-3.1-flash-tts-preview',
        'gemini-9.9-flash-preview-tts',
    ],
)
def test_tts_models_do_not_advertise_system_role(model_name: str) -> None:
    """TTS ignores system instructions, so no TTS entry advertises a system role."""
    supports = google_model_info(model_name).supports

    assert supports is not None
    assert supports.system_role is False


@pytest.mark.parametrize(
    ('model_name', 'expected_label'),
    [
        ('gemma-4-26b-a4b-it', 'Google AI - Gemma 4 26B A4B IT'),
        ('gemma-4-31b-it', 'Google AI - Gemma 4 31B IT'),
    ],
)
def test_gemma_4_models_register_per_name_capabilities(model_name: str, expected_label: str) -> None:
    """Each gemma-4 id carries its own label instead of sharing the generic Gemma entry."""
    model_info = google_model_info(model_name)

    assert model_info.label == expected_label
    assert model_info.supports == Supports(
        multiturn=True,
        media=True,
        tools=True,
        tool_choice=True,
        system_role=True,
        constrained=Constrained.ALL,
        output=['text', 'json'],
    )
    assert get_model_config_schema(model_name) is GemmaConfig


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


@patch('genkit_google_genai._models._gemini.GeminiModel._create_tool')
def test_gemini_model__get_tools(
    mock_create_tool: MagicMock,
    gemini_model_instance: GeminiModel,
) -> None:
    """Unit test for GeminiModel._get_tools."""
    mock_create_tool.return_value = (
        genai_types.Tool(),
        ToolWire(original_name='tool_1', wire_name='tool_1', wrapped=False),
    )

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
                    Part.from_text('test text'),
                ],
            ),
        ],
    )

    tools = gemini_model_instance._get_tools(request)

    assert len(tools) == len(request_tools)
    for tool in tools:
        assert isinstance(tool, genai_types.Tool)


@patch('genkit_google_genai._models._gemini.GeminiModel._convert_schema_property')
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

    gemini_tool, wire = gemini_model_instance._create_tool(
        tool_defined,
    )

    assert isinstance(gemini_tool, genai_types.Tool)
    assert wire.original_name == 'model_tool'
    assert wire.wire_name == 'model_tool'


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
    'genkit_google_genai._models._gemini.generate_cache_key',
    new_callable=MagicMock,
)
@patch(
    'genkit_google_genai._models._gemini.validate_context_cache_request',
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
                    Part.from_text('request text'),
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
    dumped = gemini_model_instance._normalize_config_to_dict(GeminiConfig.model_validate({'code_execution': True}))

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
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'code_execution': True}),
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

    dumped = gemma_model._normalize_config_to_dict(GemmaConfig(temperature=3.0))

    assert dumped == {'temperature': 3.0}


@pytest.mark.asyncio
async def test_gemini_model__config_extra_rides_on_extra_body(
    gemini_model_instance: GeminiModel,
) -> None:
    """`extra` rides on extra_body under its wire path so a newly supported field still reaches the API."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'temperature': 0.5, 'extra': {'generationConfig': {'fooBar': 1}}}),
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.temperature == 0.5
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body == {'generationConfig': {'fooBar': 1}}


def _json_output_request() -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        output=OutputConfig(
            format='json',
            json_schema={'type': 'object', 'properties': {'name': {'type': 'string'}}},
            constrained=True,
        ),
    )


@pytest.mark.asyncio
async def test_gemini_model__json_output_sets_constrained_config() -> None:
    """A standard Gemini model receives response_mime_type and response_schema for JSON output."""
    model = GeminiModel(version='gemini-2.5-flash', client=MagicMock(spec=genai.Client))

    cfg = await model._genkit_to_googleai_cfg(_json_output_request())

    assert cfg is not None
    assert cfg.response_mime_type == 'application/json'
    assert cfg.response_schema is not None


@pytest.mark.asyncio
async def test_gemini_model__tts_json_output_skips_constrained_config() -> None:
    """A TTS model receives neither response_mime_type nor response_schema for JSON output."""
    model = GeminiModel(version='gemini-2.5-flash-preview-tts', client=MagicMock(spec=genai.Client))

    cfg = await model._genkit_to_googleai_cfg(_json_output_request())

    assert cfg is not None
    assert cfg.response_mime_type is None
    assert cfg.response_schema is None


@pytest.mark.parametrize(
    ('version', 'expected_schema'),
    [
        ('gemini-2.5-flash-preview-tts', GeminiTtsConfig),
        ('gemini-3.1-flash-tts-preview', GeminiTtsConfig),
        ('gemini-2.0-flash-preview-image-generation', GeminiImageConfig),
        ('gemini-3-pro-image', GeminiImageConfig),
        ('gemini-3.1-flash-image', GeminiImageConfig),
        ('gemini-3.1-flash-image-preview', GeminiImageConfig),
        ('gemini-3-pro-image-preview', GeminiImageConfig),
        ('gemini-2.5-flash-image', GeminiImageConfig),
        ('gemini-2.5-flash-image-preview', GeminiImageConfig),
        ('gemma-2-27b-it', GemmaConfig),
        ('gemma-4-31b-it', GemmaConfig),
        ('gemini-2.0-flash-001', GeminiConfig),
    ],
)
def test_get_model_config_schema_routes_by_model_family(
    version: str,
    expected_schema: type[GeminiConfig],
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
            Message(role=Role.USER, content=[Part.from_text('What is the weather in Seattle?')]),
            Message(
                role=Role.MODEL,
                content=[Part.from_text('I will check.')],
            ),
            Message(
                role=Role.TOOL,
                content=[Part.from_text('Sunny, 72°F in Seattle')],
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
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
    )

    async def failing_stream() -> Any:  # noqa: ANN401
        raise APIError(code, {'error': {'message': 'provider failed'}})
        if False:
            yield  # pragma: no cover

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content_stream.return_value = failing_stream()
    gemini = GeminiModel('gemini-2.5-flash', googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(request, ctx)
    assert raised.value.status == status


@pytest.mark.asyncio
async def test_streaming_generate_classifies_mid_stream_error(mocker: MockerFixture) -> None:
    """A 503 after the first chunk is still UNAVAILABLE so retry can wait it out."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
    )
    first = genai.types.GenerateContentResponse(
        candidates=[genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='Hello')]))]
    )

    async def mid_stream_fail() -> Any:  # noqa: ANN401
        yield first
        raise APIError(503, {'error': {'message': 'overloaded'}})

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content_stream.return_value = mid_stream_fail()
    gemini = GeminiModel('gemini-2.5-flash', googleai_client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(request, ctx)
    assert raised.value.status == 'UNAVAILABLE'


@pytest.mark.asyncio
async def test_generate_classifies_503_as_unavailable(mocker: MockerFixture) -> None:
    """A provider 503 must stay retryable, not collapse to INTERNAL."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
    )
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.side_effect = APIError(503, {'error': {'message': 'overloaded'}})
    gemini = GeminiModel('gemini-2.5-flash', googleai_client_mock)

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


@pytest.fixture
def tts_model_instance() -> GeminiModel:
    """Common initialization of a TTS GeminiModel."""
    return GeminiModel(
        version='gemini-2.5-flash-preview-tts',
        client=MagicMock(spec=genai.Client),
    )


def test_speech_config_declares_sdk_fields() -> None:
    """Language code and multi-speaker voice config validate as typed fields, by name or alias."""
    config = SpeechConfig.model_validate({
        'language_code': 'en-US',
        'multiSpeakerVoiceConfig': {
            'speakerVoiceConfigs': [
                {'speaker': 'Alice', 'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}},
            ]
        },
    })

    assert config.language_code == 'en-US'
    assert config.multi_speaker_voice_config is not None
    speakers = config.multi_speaker_voice_config.speaker_voice_configs
    assert speakers is not None
    assert speakers[0].speaker == 'Alice'
    assert speakers[0].voice_config is not None
    assert speakers[0].voice_config.prebuilt_voice_config is not None
    assert speakers[0].voice_config.prebuilt_voice_config.voice_name == 'Kore'


def test_tts_config_json_schema_exposes_speech_config_fields() -> None:
    """The Dev UI schema lists every speech config field the SDK accepts."""
    schema = GeminiTtsConfig.model_json_schema(by_alias=True)
    speech = schema['$defs']['SpeechConfig']['properties']

    assert {'voiceConfig', 'languageCode', 'multiSpeakerVoiceConfig'} <= set(speech)


def test_speech_config_populates_by_field_name() -> None:
    """The speech config validates from snake_case field names, not only aliases."""
    config = SpeechConfig.model_validate({'voice_config': {'prebuilt_voice_config': {'voice_name': 'Kore'}}})

    assert config.voice_config is not None
    assert config.voice_config.prebuilt_voice_config is not None
    assert config.voice_config.prebuilt_voice_config.voice_name == 'Kore'


@pytest.mark.asyncio
async def test_gemini_model__speech_config_keeps_language_code(
    tts_model_instance: GeminiModel,
) -> None:
    """A language code on the speech config reaches the SDK config."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({
            'speechConfig': {
                'languageCode': 'en-US',
                'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}},
            }
        }),
    )

    cfg = await tts_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert isinstance(cfg.speech_config, genai_types.SpeechConfig)
    assert cfg.speech_config.language_code == 'en-US'
    assert cfg.speech_config.voice_config is not None


@pytest.mark.asyncio
async def test_gemini_model__speech_config_keeps_multi_speaker_voice_config(
    tts_model_instance: GeminiModel,
) -> None:
    """A multi-speaker voice config on the speech config reaches the SDK config."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({
            'speechConfig': {
                'multiSpeakerVoiceConfig': {
                    'speakerVoiceConfigs': [
                        {'speaker': 'Alice', 'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}},
                    ]
                }
            }
        }),
    )

    cfg = await tts_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert isinstance(cfg.speech_config, genai_types.SpeechConfig)
    assert cfg.speech_config.multi_speaker_voice_config is not None
    speakers = cfg.speech_config.multi_speaker_voice_config.speaker_voice_configs
    assert speakers is not None
    assert [s.speaker for s in speakers] == ['Alice']


def test_gemini_tts_config_with_unknown_speech_config_key_raises_validation_error() -> None:
    """`speechConfig={'languageCodes': ...}` fails at construction instead of being silently dropped."""
    with pytest.raises(ValidationError, match='languageCodes'):
        GeminiTtsConfig.model_validate({'speechConfig': {'languageCodes': 'en-US'}})


@pytest.mark.asyncio
async def test_generate_keeps_caller_response_modalities_on_tts_model(mocker: MockerFixture) -> None:
    """A TTS model keeps the response modalities the caller asked for."""
    version = 'gemini-2.5-flash-preview-tts'
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({'responseModalities': ['AUDIO', 'TEXT']}),
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='ok')]))
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.return_value = genai.types.GenerateContentResponse(candidates=[candidate])

    await GeminiModel(version, client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert sent_config.response_modalities == ['AUDIO', 'TEXT']


@pytest.mark.asyncio
async def test_generate_keeps_caller_response_modalities_on_image_model(mocker: MockerFixture) -> None:
    """An image model keeps the response modalities the caller asked for."""
    version = 'gemini-2.5-flash-image'
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiImageConfig.model_validate({'responseModalities': ['IMAGE']}),
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='ok')]))
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.return_value = genai.types.GenerateContentResponse(candidates=[candidate])

    await GeminiModel(version, client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert sent_config.response_modalities == ['IMAGE']


def _tts_client_mock(mocker: MockerFixture, *, vertexai: bool) -> AsyncMock:
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='ok')]))
    client_mock = mocker.AsyncMock()
    client_mock.vertexai = vertexai
    client_mock.aio.models.generate_content.return_value = genai.types.GenerateContentResponse(candidates=[candidate])
    return client_mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('version', 'vertexai'),
    [
        ('gemini-3.1-flash-tts-preview', False),
        ('gemini-3.1-flash-tts-preview', True),
        ('gemini-2.5-flash-preview-tts', True),
    ],
)
async def test_generate_defaults_the_tts_voice_where_one_is_needed(
    mocker: MockerFixture, version: str, vertexai: bool
) -> None:
    """A TTS request without a speech config gets the default voice on Vertex AI and for 3.1 TTS."""
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])])
    client_mock = _tts_client_mock(mocker, vertexai=vertexai)

    await GeminiModel(version, client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert isinstance(sent_config.speech_config, genai_types.SpeechConfig)
    assert sent_config.speech_config.voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Kore'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'version', ['gemini-2.5-flash-preview-tts', 'gemini-2.5-pro-preview-tts', 'gemini-3.8-flash-tts']
)
async def test_generate_sends_no_voice_to_gemini_api_tts_that_picks_its_own(
    mocker: MockerFixture, version: str
) -> None:
    """On the Gemini API, a TTS model that accepts a voiceless request is sent no speech config."""
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])])
    client_mock = _tts_client_mock(mocker, vertexai=False)

    await GeminiModel(version, client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert sent_config.response_modalities == ['AUDIO']
    assert sent_config.speech_config is None


@pytest.mark.asyncio
async def test_generate_defaults_the_tts_voice_next_to_a_language_code(mocker: MockerFixture) -> None:
    """A speech config that sets only a language code still gets the default voice."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({'speechConfig': {'languageCode': 'en-US'}}),
    )
    client_mock = _tts_client_mock(mocker, vertexai=False)

    await GeminiModel('gemini-3.1-flash-tts-preview', client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert isinstance(sent_config.speech_config, genai_types.SpeechConfig)
    assert sent_config.speech_config.language_code == 'en-US'
    assert sent_config.speech_config.voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Kore'


@pytest.mark.asyncio
async def test_generate_keeps_the_caller_tts_voice(mocker: MockerFixture) -> None:
    """A voice named by the caller is sent unchanged where the default would otherwise apply."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({
            'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Puck'}}}
        }),
    )
    client_mock = _tts_client_mock(mocker, vertexai=True)

    await GeminiModel('gemini-2.5-flash-preview-tts', client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert isinstance(sent_config.speech_config, genai_types.SpeechConfig)
    assert sent_config.speech_config.voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config is not None
    assert sent_config.speech_config.voice_config.prebuilt_voice_config.voice_name == 'Puck'


@pytest.mark.asyncio
async def test_generate_adds_no_voice_to_a_multi_speaker_config(mocker: MockerFixture) -> None:
    """A multi-speaker voice config is sent without a single-voice config beside it."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiTtsConfig.model_validate({
            'speechConfig': {
                'multiSpeakerVoiceConfig': {
                    'speakerVoiceConfigs': [
                        {'speaker': 'Alice', 'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}},
                    ]
                }
            }
        }),
    )
    client_mock = _tts_client_mock(mocker, vertexai=True)

    await GeminiModel('gemini-2.5-flash-preview-tts', client_mock).generate(request, ActionRunContext())

    sent_config = client_mock.aio.models.generate_content.call_args.kwargs['config']
    assert isinstance(sent_config.speech_config, genai_types.SpeechConfig)
    assert sent_config.speech_config.voice_config is None
    assert sent_config.speech_config.multi_speaker_voice_config is not None


# Each strict nested Gemini setting and the google.genai type it is sent as.
_NESTED_SDK_MIRRORS: list[tuple[type[BaseModel], type[BaseModel]]] = [
    (_gemini.SafetySettingsSchema, genai_types.SafetySetting),
    (_gemini.PrebuiltVoiceConfig, genai_types.PrebuiltVoiceConfig),
    (_gemini.FunctionCallingConfig, genai_types.FunctionCallingConfig),
    (_gemini.ThinkingConfig, genai_types.ThinkingConfig),
    (_gemini.FileSearchConfig, genai_types.FileSearch),
    (_gemini.ImageConfig, genai_types.ImageConfig),
    (_gemini.VoiceConfig, genai_types.VoiceConfig),
    (_gemini.SpeakerVoiceConfig, genai_types.SpeakerVoiceConfig),
    (_gemini.MultiSpeakerVoiceConfig, genai_types.MultiSpeakerVoiceConfig),
    (_gemini.SpeechConfig, genai_types.SpeechConfig),
    (_gemini.ImageOutputOptions, genai_types.ImageConfigImageOutputOptions),
    (_gemini.ReplicatedVoiceConfig, genai_types.ReplicatedVoiceConfig),
    (_gemini.VoiceConsentSignature, genai_types.VoiceConsentSignature),
    (_gemini.CodeExecutionConfig, genai_types.ToolCodeExecution),
    (_gemini.UrlContextConfig, genai_types.UrlContext),
    (_gemini.GoogleSearchConfig, genai_types.GoogleSearch),
    (_gemini.SearchTypes, genai_types.SearchTypes),
    (_gemini.WebSearchConfig, genai_types.WebSearch),
    (_gemini.ImageSearchConfig, genai_types.ImageSearch),
    (_gemini.TimeRangeFilter, genai_types.Interval),
]


@pytest.mark.parametrize(('ours', 'sdk'), _NESTED_SDK_MIRRORS, ids=lambda c: c.__name__)
def test_nested_gemini_setting_declares_exactly_the_sdk_fields(ours: type[BaseModel], sdk: type[BaseModel]) -> None:
    """A strict nested setting accepts every key its google-genai type accepts, and no other."""
    assert set(ours.model_fields) == set(sdk.model_fields)


@pytest.mark.parametrize(
    ('config_class', 'field', 'nested'),
    [
        (GeminiConfig, 'safetySettings', _gemini.SafetySettingsSchema),
        (GeminiConfig, 'functionCallingConfig', _gemini.FunctionCallingConfig),
        (GeminiConfig, 'thinkingConfig', _gemini.ThinkingConfig),
        (GeminiConfig, 'fileSearch', _gemini.FileSearchConfig),
        (GeminiImageConfig, 'imageConfig', _gemini.ImageConfig),
    ],
)
def test_gemini_config_form_lists_every_nested_field(
    config_class: type[BaseModel], field: str, nested: type[BaseModel]
) -> None:
    """The hand-written Dev UI schema for a nested setting lists every field the class accepts."""
    schema = config_class.model_json_schema(by_alias=True)['properties'][field]
    properties = schema.get('items', schema)['properties']

    assert set(properties) == {f.alias or name for name, f in nested.model_fields.items()}


@pytest.mark.asyncio
async def test_gemini_model__tool_toggle_false_attaches_no_tool(gemini_model_instance: GeminiModel) -> None:
    """`code_execution`, `google_search`, and `url_context` set to False add no tool."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'code_execution': False, 'google_search': False, 'url_context': False}),
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is None or not cfg.tools


@pytest.mark.asyncio
async def test_gemini_model__tool_toggle_empty_options_attaches_tool(gemini_model_instance: GeminiModel) -> None:
    """An empty options dict attaches the tool, same as True."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        config=GeminiConfig.model_validate({'code_execution': {}, 'google_search': {}, 'url_context': {}}),
    )

    cfg = await gemini_model_instance._genkit_to_googleai_cfg(request)

    assert cfg is not None
    tools = cast(list[genai_types.Tool], cfg.tools)
    assert [t.code_execution is not None for t in tools] == [True, False, False]
    assert [t.google_search is not None for t in tools] == [False, True, False]
    assert [t.url_context is not None for t in tools] == [False, False, True]


# ---------------------------------------------------------------------------
# Error classification: credential failures, cache calls, unknown exceptions
# ---------------------------------------------------------------------------


def _hi_request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Is the salmon tartine nut-free?')])])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'auth_error',
    [
        DefaultCredentialsError('Your default credentials were not found.'),
        RefreshError('invalid_grant: Token has been expired or revoked.'),
    ],
)
async def test_generate_credential_failure_is_unauthenticated(mocker: MockerFixture, auth_error: Exception) -> None:
    """The SDK resolves ADC on the request; a missing or revoked credential is UNAUTHENTICATED, not INTERNAL."""
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = auth_error
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ActionRunContext())

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is auth_error


@pytest.mark.asyncio
async def test_streaming_generate_credential_failure_is_unauthenticated(mocker: MockerFixture) -> None:
    """Streaming classifies a revoked token the same way non-streaming does."""
    revoked = RefreshError('invalid_grant: Token has been expired or revoked.')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content_stream.side_effect = revoked
    gemini = GeminiModel('gemini-2.5-flash', client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ctx)

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is revoked


@pytest.mark.asyncio
async def test_generate_metadata_server_refresh_error_stays_raw(mocker: MockerFixture) -> None:
    """On Cloud Run or GKE a metadata-server blip is RefreshError from TransportError, retryable=False; still raw."""
    try:
        try:
            raise TransportError('metadata server unreachable')
        except TransportError as e:
            raise RefreshError(e) from e
    except RefreshError as error:
        blip = error
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = blip
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(RefreshError) as raised:
        await gemini.generate(_hi_request(), ActionRunContext())

    assert raised.value is blip


@pytest.mark.asyncio
async def test_generate_retryable_refresh_error_stays_raw(mocker: MockerFixture) -> None:
    """google.auth marked the refresh retryable (token endpoint 503), so retry must still see it."""
    flaky = RefreshError('token endpoint returned 503', retryable=True)
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = flaky
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(RefreshError) as raised:
        await gemini.generate(_hi_request(), ActionRunContext())

    assert raised.value is flaky


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
async def test_generate_unknown_exception_stays_raw(mocker: MockerFixture, streaming: bool) -> None:
    """A dropped connection has no known status; it reaches the caller unchanged, not as INTERNAL."""
    dropped = ConnectionResetError('Connection reset by peer')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = dropped
    client_mock.aio.models.generate_content_stream.side_effect = dropped
    gemini = GeminiModel('gemini-2.5-flash', client_mock)
    ctx = ActionRunContext(streaming_callback=mocker.MagicMock()) if streaming else ActionRunContext()

    with pytest.raises(ConnectionResetError) as raised:
        await gemini.generate(_hi_request(), ctx)

    assert raised.value is dropped


@pytest.mark.asyncio
@patch('genkit_google_genai._models._gemini.validate_context_cache_request', new_callable=MagicMock)
@pytest.mark.parametrize(
    ('failing_call', 'code', 'status'),
    [
        ('list', 429, 'RESOURCE_EXHAUSTED'),
        ('create', 400, 'INVALID_ARGUMENT'),
        ('create', 503, 'UNAVAILABLE'),
    ],
)
async def test_retrieve_cached_content_classifies_api_error(
    _validate: MagicMock, failing_call: str, code: int, status: str
) -> None:
    """Cache calls run before generate, so their provider errors need a status too."""
    pages = AsyncMock()
    pages.__aiter__.return_value = []
    client_mock = MagicMock()
    client_mock.aio.caches.list = AsyncMock(return_value=pages)
    client_mock.aio.caches.create = AsyncMock(return_value=genai_types.CachedContent())
    api_error = APIError(code, {'error': {'message': 'cache call failed'}})
    getattr(client_mock.aio.caches, failing_call).side_effect = api_error
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini._retrieve_cached_content(
            request=_hi_request(), model_name='gemini-2.5-flash', cache_config={}, contents=[]
        )

    assert raised.value.status == status
    assert raised.value.cause is api_error


@pytest.mark.asyncio
@patch('genkit_google_genai._models._gemini.validate_context_cache_request', new_callable=MagicMock)
async def test_retrieve_cached_content_credential_failure_is_unauthenticated(_validate: MagicMock) -> None:
    """Missing ADC on the cache lookup is UNAUTHENTICATED."""
    client_mock = MagicMock()
    client_mock.aio.caches.list = AsyncMock(side_effect=DefaultCredentialsError('no ADC'))
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini._retrieve_cached_content(
            request=_hi_request(), model_name='gemini-2.5-flash', cache_config={}, contents=[]
        )

    assert raised.value.status == 'UNAUTHENTICATED'


@pytest.mark.asyncio
async def test_request_client_credential_failure_is_unauthenticated() -> None:
    """A Vertex location override with no ADC configured is a credential problem, not a bad argument."""
    client = MagicMock()
    client.vertexai = True
    gemini = GeminiModel(
        'gemini-2.5-flash',
        client,
        client_kwargs={'vertexai': True, 'project': 'menu-prod', 'location': 'us-central1'},
    )
    no_adc = DefaultCredentialsError('Your default credentials were not found.')

    with patch('genkit_google_genai._models._gemini.genai.Client', side_effect=no_adc):
        with pytest.raises(GenkitError) as raised:
            await gemini._resolve_request_client(
                ModelRequest(messages=_hi_request().messages, config={'location': 'europe-west4'})
            )

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is no_adc


@pytest.mark.asyncio
async def test_request_client_bad_override_is_invalid_argument() -> None:
    """The SDK rejecting an override combination is the caller's input."""
    client = MagicMock()
    client.vertexai = False
    gemini = GeminiModel('gemini-2.5-flash', client, client_kwargs={'vertexai': False, 'api_key': 'k'})
    rejected = ValueError('Project/location and API key are mutually exclusive')

    with patch('genkit_google_genai._models._gemini.genai.Client', side_effect=rejected):
        with pytest.raises(GenkitError, match='mutually exclusive') as raised:
            await gemini._resolve_request_client(
                ModelRequest(messages=_hi_request().messages, config={'api_version': 'v1alpha'})
            )

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert raised.value.cause is rejected


@pytest.mark.asyncio
async def test_request_client_unknown_failure_stays_raw() -> None:
    """Any other client construction failure keeps its own type."""
    client = MagicMock()
    client.vertexai = False
    gemini = GeminiModel('gemini-2.5-flash', client, client_kwargs={'vertexai': False, 'api_key': 'k'})
    boom = RuntimeError('SDK bug')

    with patch('genkit_google_genai._models._gemini.genai.Client', side_effect=boom):
        with pytest.raises(RuntimeError) as raised:
            await gemini._resolve_request_client(
                ModelRequest(messages=_hi_request().messages, config={'api_version': 'v1alpha'})
            )

    assert raised.value is boom
