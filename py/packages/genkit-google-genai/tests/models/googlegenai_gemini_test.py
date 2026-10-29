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
import math
from datetime import date, datetime
from typing import Annotated, Any, Literal, cast, get_args
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import httpx
import pytest
from genkit_google_genai._models import _gemini
from genkit_google_genai._models._gemini import (
    DEFAULT_SUPPORTS_MODEL,
    GEMINI_CATALOG_IDS,
    GeminiConfig,
    GeminiImageConfig,
    GeminiModel,
    GeminiTtsConfig,
    GemmaConfig,
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
from pydantic import BaseModel, Field, HttpUrl, ValidationError
from pytest_mock import MockerFixture

from genkit import FinishReason, GenkitError, Message, ModelResponse, Part, Role
from genkit._core._compat import StrEnum
from genkit.model import Constrained, ModelContext, ModelInfo, ModelRequest, OutputConfig, Supports, ToolDefinition
from genkit.plugin_api import to_json_schema

ALL_VERSIONS = sorted({*GEMINI_CATALOG_IDS, *get_args(KnownGemma)})
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

    ctx = ModelContext()
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

    async def stream() -> Any:  # noqa: ANN401
        yield resp

    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.vertexai = False
    googleai_client_mock.aio.models.generate_content_stream.return_value = stream()
    on_chunk_mock = mocker.MagicMock()
    gemini = GeminiModel(version, googleai_client_mock)

    ctx = ModelContext(streaming_callback=on_chunk_mock)
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
    assert response.message.content[0].text == response_text
    on_chunk_mock.assert_called_once()
    assert on_chunk_mock.call_args.args[0].content[0].text == response_text


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
    ctx = ModelContext(streaming_callback=on_chunk_mock)

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
    ctx = ModelContext(streaming_callback=on_chunk_mock)

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

    ctx = ModelContext()
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
        baz: list[str] | None = Field(None, description='baz field')

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
            'baz': genai_types.Schema(
                type=genai_types.Type.ARRAY,
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
    ctx = ModelContext()

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
        ('gemini-3.8-flash-tts', 'Google AI - Gemini 3.8 Flash TTS'),
        ('gemini-3.8-flash-lite-tts', 'Google AI - Gemini 3.8 Flash Lite TTS'),
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
        'gemini-3.8-flash-tts',
        'gemini-3.8-flash-lite-tts',
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
        # Test Case 13: $ref to the document root
        (
            {'type': 'OBJECT', 'properties': {'user': {'$ref': '#'}}},
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
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema, defs)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'cannot be resolved' in exc_info.value.original_message


class _LinkedList(BaseModel):
    value: int
    next: '_LinkedList | None' = None


class _RequiredLinkedList(BaseModel):
    value: int
    next: '_RequiredLinkedList | None'


class _Tree(BaseModel):
    children: list['_Tree'] = []


class _RequiredTree(BaseModel):
    children: list['_RequiredTree']


class _OptionalTree(BaseModel):
    children: list['_OptionalTree'] | None = None


class _Grid(BaseModel):
    rows: list[list['_Grid']] = []


class _Mixed(BaseModel):
    x: list['_Mixed'] | int


class _OptionalMixed(BaseModel):
    x: list['_OptionalMixed'] | int | None = None


class _NodeOrInt(BaseModel):
    next: '_NodeOrInt | int'


def test_gemini_model__create_tool_leaves_out_recursive_param(gemini_model_instance: GeminiModel) -> None:
    """A property that refers back to a model that contains it is left out."""
    tool = ToolDefinition(name='t', description='d', input_schema=to_json_schema(_LinkedList))

    gemini_tool, _ = gemini_model_instance._create_tool(tool)
    declarations = gemini_tool.function_declarations

    assert declarations is not None
    assert declarations[0].parameters == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=['value'],
        properties={'value': genai_types.Schema(type=genai_types.Type.INTEGER)},
    )


@pytest.mark.parametrize('model', [_Tree, _OptionalTree, _Grid])
def test_gemini_model__create_tool_leaves_out_recursive_items(
    model: type[BaseModel],
    gemini_model_instance: GeminiModel,
) -> None:
    """An array whose items refer back to a model that contains it is left out, as is an optional or nested one."""
    tool = ToolDefinition(name='t', description='d', input_schema=to_json_schema(model))

    gemini_tool, _ = gemini_model_instance._create_tool(tool)
    declarations = gemini_tool.function_declarations

    assert declarations is not None
    assert declarations[0].parameters == genai_types.Schema(type=genai_types.Type.OBJECT, properties={})


def test_gemini_model__convert_schema_property_strict_leaves_out_recursive_items(
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode an optional array whose items refer back to a model that contains it is left out."""
    schema = gemini_model_instance._convert_schema_property(to_json_schema(_Tree), strict=True)

    assert schema == genai_types.Schema(type=genai_types.Type.OBJECT, properties={})


def test_gemini_model__convert_schema_property_strict_rejects_required_recursive_items(
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode a required array whose items refer back to a model that contains it raises naming it."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(to_json_schema(_RequiredTree), strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'children' in exc_info.value.original_message


def test_gemini_model__convert_schema_property_strict_leaves_out_recursive_property(
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode an optional recursive property is left out rather than raising."""
    schema = gemini_model_instance._convert_schema_property(to_json_schema(_LinkedList), strict=True)

    assert schema is not None and list(schema.properties or {}) == ['value']


def test_gemini_model__convert_schema_property_strict_rejects_required_recursive_property(
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode a required recursive property cannot be left out, so it raises naming the property."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(to_json_schema(_RequiredLinkedList), strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'next' in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema, needle',
    [
        (to_json_schema(_Mixed), 'anyOf branch'),
        (to_json_schema(_OptionalMixed), 'anyOf branch'),
        (to_json_schema(_NodeOrInt), 'anyOf branch'),
        ({'oneOf': [{'type': 'integer'}, {'type': 'array'}]}, 'oneOf branch'),
        ({'type': ['string', 'integer', 'null']}, "type ['string', 'integer', 'null']"),
    ],
)
def test_gemini_model__convert_schema_property_strict_rejects_union_losing_a_branch(
    input_schema: dict[str, object],
    needle: str,
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode a union with a branch that cannot be converted raises instead of narrowing to the others."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema, strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert needle in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema, expected',
    [
        (
            to_json_schema(_Mixed),
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=['x'],
                properties={'x': genai_types.Schema(type=genai_types.Type.INTEGER)},
            ),
        ),
        ({'type': ['string', 'integer', 'null']}, genai_types.Schema(type=genai_types.Type.STRING, nullable=True)),
    ],
)
def test_gemini_model__convert_schema_property_leaves_out_union_branch(
    input_schema: dict[str, object],
    expected: genai_types.Schema,
    gemini_model_instance: GeminiModel,
) -> None:
    """Without strict, a union branch that cannot be converted is left out and the others are kept."""
    assert gemini_model_instance._convert_schema_property(input_schema) == expected


@pytest.mark.parametrize('strict', [False, True])
def test_gemini_model__convert_schema_property_null_type_list_branch_is_nullable(
    strict: bool,
    gemini_model_instance: GeminiModel,
) -> None:
    """A branch typed ``['null']`` counts as a null branch and makes the union nullable."""
    schema: dict[str, object] = {'anyOf': [{'type': 'integer'}, {'type': ['null']}]}

    assert gemini_model_instance._convert_schema_property(schema, strict=strict) == genai_types.Schema(
        type=genai_types.Type.INTEGER, nullable=True
    )


def test_gemini_model__convert_schema_property_expands_repeated_ref(gemini_model_instance: GeminiModel) -> None:
    """A definition used by two sibling properties is expanded for both."""

    class Point(BaseModel):
        x: int

    class Segment(BaseModel):
        start: Point
        end: Point

    schema = gemini_model_instance._convert_schema_property(to_json_schema(Segment))

    assert schema is not None and schema.properties is not None
    point = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=['x'],
        properties={'x': genai_types.Schema(type=genai_types.Type.INTEGER)},
    )
    assert schema.properties == {'start': point, 'end': point}


@pytest.mark.parametrize('strict', [False, True])
def test_gemini_model__convert_schema_property_keeps_ref_sibling_keywords(
    strict: bool, gemini_model_instance: GeminiModel
) -> None:
    """Keywords next to a $ref apply to the schema it resolves to."""

    class Color(StrEnum):
        RED = 'red'
        BLUE = 'blue'

    class Paint(BaseModel):
        color: Color = Field(Color.RED, description='Paint color')

    schema = gemini_model_instance._convert_schema_property(to_json_schema(Paint), strict=strict)

    assert schema is not None and schema.properties is not None
    assert schema.properties['color'] == genai_types.Schema(
        type=genai_types.Type.STRING, enum=['red', 'blue'], default='red', description='Paint color'
    )


def test_gemini_model__convert_schema_property_keeps_keywords(gemini_model_instance: GeminiModel) -> None:
    """anyOf, type lists, format, default, bounds and propertyOrdering survive conversion.

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
            'ratio': {
                'anyOf': [{'type': 'number', 'minimum': 0.5, 'maximum': 1.5}, {'type': 'null'}],
                'title': 'Ratio',
            },
            'tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 3},
            'anything': {'title': 'Anything'},
        },
    }

    assert gemini_model_instance._convert_schema_property(schema) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        property_ordering=['count', 'label', 'kind', 'ratio', 'tags'],
        required=['count'],
        properties={
            'count': genai_types.Schema(type=genai_types.Type.INTEGER, default=1, minimum=0.0, maximum=10.0),
            'label': genai_types.Schema(type=genai_types.Type.STRING, format='date-time', nullable=True),
            'pick': genai_types.Schema(type=genai_types.Type.STRING),
            'kind': genai_types.Schema(
                any_of=[
                    genai_types.Schema(type=genai_types.Type.STRING),
                    genai_types.Schema(type=genai_types.Type.INTEGER),
                ],
                description='string or int',
            ),
            'ratio': genai_types.Schema(type=genai_types.Type.NUMBER, nullable=True, minimum=0.5, maximum=1.5),
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
        ({'type': 'number', 'default': math.inf}, 'default'),
        ({'type': 'number', 'maximum': math.nan}, 'maximum'),
        ({'type': 'number', 'minimum': 10**400}, 'minimum'),
        ({'type': 'array', 'items': {'type': 'string'}, 'minItems': math.inf}, 'minItems'),
    ],
)
def test_gemini_model__convert_schema_property_rejects_bad_keyword_value(
    input_schema: dict[str, object],
    keyword: str,
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode a keyword value that cannot be coerced raises INVALID_ARGUMENT naming the keyword."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema, strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert keyword in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema, expected',
    [
        ({'type': 'integer', 'minimum': 'zero'}, genai_types.Schema(type=genai_types.Type.INTEGER)),
        ({'type': 'object', 'propertyOrdering': 'ab'}, genai_types.Schema(type=genai_types.Type.OBJECT, properties={})),
        ({'type': 'number', 'default': math.inf}, genai_types.Schema(type=genai_types.Type.NUMBER)),
        ({'type': 'number', 'maximum': math.nan}, genai_types.Schema(type=genai_types.Type.NUMBER)),
        ({'type': 'number', 'minimum': 10**400}, genai_types.Schema(type=genai_types.Type.NUMBER)),
        (
            {'type': 'array', 'items': {'type': 'string'}, 'minItems': math.inf},
            genai_types.Schema(type=genai_types.Type.ARRAY, items=genai_types.Schema(type=genai_types.Type.STRING)),
        ),
    ],
)
def test_gemini_model__convert_schema_property_leaves_out_bad_keyword_value(
    input_schema: dict[str, object],
    expected: genai_types.Schema,
    gemini_model_instance: GeminiModel,
) -> None:
    """Without strict, a keyword value that cannot be coerced is left out."""
    assert gemini_model_instance._convert_schema_property(input_schema) == expected


def test_gemini_model__convert_schema_property_non_object_property(gemini_model_instance: GeminiModel) -> None:
    """A property whose schema is not an object is left out, and raises in strict mode when it is required."""
    schema: dict[str, object] = {
        'type': 'object',
        'properties': {'x': 'string', 'y': {'type': 'string'}},
        'required': ['x', 'y'],
    }

    assert gemini_model_instance._convert_schema_property(schema) == genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=['y'],
        properties={'y': genai_types.Schema(type=genai_types.Type.STRING)},
    )
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(schema, strict=True)
    assert 'required property x' in exc_info.value.original_message


@pytest.mark.parametrize('raw_type', ['bogus', None, 123])
def test_gemini_model__convert_schema_property_rejects_unsupported_type(
    raw_type: object,
    gemini_model_instance: GeminiModel,
) -> None:
    """A type the Gemini Schema cannot express raises INVALID_ARGUMENT."""
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
        required=['pet'],
        properties={
            'pet': genai_types.Schema(
                any_of=[
                    genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        required=['kind', 'meows'],
                        properties={
                            'kind': genai_types.Schema(type=genai_types.Type.STRING, enum=['cat']),
                            'meows': genai_types.Schema(type=genai_types.Type.INTEGER),
                        },
                    ),
                    genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        required=['kind', 'barks'],
                        properties={
                            'kind': genai_types.Schema(type=genai_types.Type.STRING, enum=['dog']),
                            'barks': genai_types.Schema(type=genai_types.Type.BOOLEAN),
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


def test_gemini_model__convert_schema_property_keeps_only_string_enums(gemini_model_instance: GeminiModel) -> None:
    """Non-string or mixed members leave a bare type; a typeless string enum with null is a nullable STRING enum."""

    class Pick(BaseModel):
        tag: Literal['a', None]

    convert = gemini_model_instance._convert_schema_property

    assert convert({'type': 'integer', 'enum': [1, 2]}) == genai_types.Schema(type=genai_types.Type.INTEGER)
    assert convert({'type': 'string', 'enum': ['a', 1]}) == genai_types.Schema(type=genai_types.Type.STRING)
    pick = convert(to_json_schema(Pick), strict=True)
    assert pick is not None and pick.properties is not None
    assert pick.properties['tag'] == genai_types.Schema(type=genai_types.Type.STRING, nullable=True, enum=['a'])


def test_gemini_model__convert_schema_property_keeps_pydantic_string_formats(
    gemini_model_instance: GeminiModel,
) -> None:
    """The format pydantic emits for datetime, date, UUID and HttpUrl fields is carried through."""

    class Stamps(BaseModel):
        at: datetime
        day: date
        ident: UUID
        link: HttpUrl

    schema = gemini_model_instance._convert_schema_property(to_json_schema(Stamps))

    assert schema is not None and schema.properties is not None
    assert {name: prop.format for name, prop in schema.properties.items()} == {
        'at': 'date-time',
        'day': 'date',
        'ident': 'uuid',
        'link': 'uri',
    }
    assert all(prop.type == genai_types.Type.STRING for prop in schema.properties.values())


def test_gemini_model__convert_schema_property_rejects_required_untyped_property(
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode a required property with no type cannot be left out, so it raises naming the property."""

    class Payload(BaseModel):
        data: Any
        note: Any = None

    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(to_json_schema(Payload), strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'data' in exc_info.value.original_message


class _RequiredAny(BaseModel):
    value: Any
    count: int


class _RequiredValues(BaseModel):
    count: int
    values: list[Any]


@pytest.mark.parametrize(
    'input_schema, expected',
    [
        (
            to_json_schema(_RequiredAny),
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=['count'],
                properties={'count': genai_types.Schema(type=genai_types.Type.INTEGER)},
            ),
        ),
        (
            to_json_schema(_RequiredValues),
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=['count'],
                properties={'count': genai_types.Schema(type=genai_types.Type.INTEGER)},
            ),
        ),
        (
            to_json_schema(_RequiredTree),
            genai_types.Schema(type=genai_types.Type.OBJECT, properties={}),
        ),
        (
            to_json_schema(_RequiredLinkedList),
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=['value'],
                properties={'value': genai_types.Schema(type=genai_types.Type.INTEGER)},
            ),
        ),
    ],
)
def test_gemini_model__create_tool_drops_left_out_param_from_required(
    input_schema: dict[str, object],
    expected: genai_types.Schema,
    gemini_model_instance: GeminiModel,
) -> None:
    """A required tool parameter that is left out is dropped from the declaration's required list."""
    tool = ToolDefinition(name='t', description='d', input_schema=input_schema)

    gemini_tool, _ = gemini_model_instance._create_tool(tool)
    declarations = gemini_tool.function_declarations

    assert declarations is not None
    assert declarations[0].parameters == expected


@pytest.mark.parametrize(
    'input_schema, expected, wrapped',
    [
        (to_json_schema(list[Any]), genai_types.Schema(type=genai_types.Type.OBJECT, properties={}), False),
        (
            to_json_schema(int),
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={'input': genai_types.Schema(type=genai_types.Type.INTEGER)},
                required=['input'],
            ),
            True,
        ),
    ],
)
def test_gemini_model__create_tool_wraps_only_a_converted_input(
    input_schema: dict[str, object],
    expected: genai_types.Schema,
    wrapped: bool,
    gemini_model_instance: GeminiModel,
) -> None:
    """A non-object input is wrapped after conversion; one that converts to None makes a tool with no arguments."""
    tool = ToolDefinition(name='t', description='d', input_schema=input_schema)

    gemini_tool, wire = gemini_model_instance._create_tool(tool)
    declarations = gemini_tool.function_declarations

    assert declarations is not None
    assert declarations[0].parameters == expected
    assert wire.wrapped is wrapped


@pytest.mark.parametrize(
    'input_schema, expected',
    [
        (
            {
                'type': 'object',
                'properties': {'id': {'type': 'string'}, 'name': {'type': 'string'}},
                'anyOf': [{'required': ['id']}, {'required': ['name']}],
            },
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    'id': genai_types.Schema(type=genai_types.Type.STRING),
                    'name': genai_types.Schema(type=genai_types.Type.STRING),
                },
            ),
        ),
        (
            {
                'type': 'object',
                'properties': {
                    'mode': {
                        'type': 'string',
                        'oneOf': [{'const': 'fast', 'title': 'Fast'}, {'const': 'slow', 'title': 'Slow'}],
                    },
                },
                'required': ['mode'],
            },
            genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=['mode'],
                properties={'mode': genai_types.Schema(type=genai_types.Type.STRING)},
            ),
        ),
    ],
    ids=['anyOf-required', 'titled-oneOf'],
)
def test_gemini_model__create_tool_keeps_type_next_to_union(
    input_schema: dict[str, object],
    expected: genai_types.Schema,
    gemini_model_instance: GeminiModel,
) -> None:
    """A node with its own type converts by that type and leaves out a sibling anyOf or oneOf, strict or not."""
    tool = ToolDefinition(name='t', description='d', input_schema=input_schema)

    gemini_tool, _ = gemini_model_instance._create_tool(tool)
    declarations = gemini_tool.function_declarations

    assert declarations is not None
    assert declarations[0].parameters == expected
    assert gemini_model_instance._convert_schema_property(input_schema, strict=True) == expected


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
    """In strict mode a node that constrains without a type raises instead of dropping out of its parent."""
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property(input_schema, strict=True)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'no type' in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema, expected',
    [
        ({'allOf': [{'type': 'string'}]}, None),
        (
            {'type': 'object', 'properties': {'x': {'allOf': [{'type': 'string'}]}}},
            genai_types.Schema(type=genai_types.Type.OBJECT, properties={}),
        ),
        ({'type': 'array', 'items': {'not': {'type': 'null'}}}, None),
    ],
)
def test_gemini_model__convert_schema_property_leaves_out_untyped_constraint(
    input_schema: dict[str, object],
    expected: genai_types.Schema | None,
    gemini_model_instance: GeminiModel,
) -> None:
    """Without strict, a node that constrains without a type is left out of its parent."""
    assert gemini_model_instance._convert_schema_property(input_schema) == expected


@pytest.mark.parametrize(
    'annotations',
    [
        {'title': 'Note', 'default': None},
        {'description': 'd', 'examples': [1], 'deprecated': True},
        {'readOnly': True, 'writeOnly': False, '$comment': 'c'},
        {'$schema': 'https://json-schema.org/draft/2020-12/schema', '$id': 'note', '$defs': {}},
    ],
)
def test_gemini_model__convert_schema_property_strict_leaves_out_annotation_only_property(
    annotations: dict[str, object],
    gemini_model_instance: GeminiModel,
) -> None:
    """In strict mode an optional property with only annotations and no type is left out rather than raising."""
    schema: dict[str, object] = {'type': 'object', 'properties': {'note': annotations, 'text': {'type': 'string'}}}

    assert gemini_model_instance._convert_schema_property(schema, strict=True) == genai_types.Schema(
        type=genai_types.Type.OBJECT, properties={'text': genai_types.Schema(type=genai_types.Type.STRING)}
    )


@pytest.mark.parametrize('additional', [True, {}, {'type': 'integer'}])
def test_gemini_model__convert_schema_property_strict_map(
    additional: object, gemini_model_instance: GeminiModel
) -> None:
    """In strict mode a map is left out when optional and raises naming it when required."""
    schema: dict[str, object] = {
        'type': 'object',
        'properties': {
            'scores': {'type': 'object', 'additionalProperties': additional},
            'name': {'type': 'string'},
        },
    }

    assert gemini_model_instance._convert_schema_property(schema, strict=True) == genai_types.Schema(
        type=genai_types.Type.OBJECT, properties={'name': genai_types.Schema(type=genai_types.Type.STRING)}
    )
    with pytest.raises(GenkitError) as exc_info:
        gemini_model_instance._convert_schema_property({**schema, 'required': ['scores']}, strict=True)
    assert 'required property scores' in exc_info.value.original_message


@pytest.mark.parametrize(
    'input_schema, strict',
    [
        ({'type': 'object', 'additionalProperties': {'type': 'integer'}}, False),
        ({'type': 'object', 'additionalProperties': False}, True),
    ],
    ids=['map-without-strict', 'closed-object-strict'],
)
def test_gemini_model__convert_schema_property_keeps_object_without_properties(
    input_schema: dict[str, object], strict: bool, gemini_model_instance: GeminiModel
) -> None:
    """A map without strict, and an object closed by additionalProperties false, stay an empty OBJECT."""
    assert gemini_model_instance._convert_schema_property(input_schema, strict=strict) == genai_types.Schema(
        type=genai_types.Type.OBJECT, properties={}
    )


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
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
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
@pytest.mark.parametrize('legacy_response_schema', [False, True])
async def test_gemini_model__json_unconstrained_sets_mime_type_only(legacy_response_schema: bool) -> None:
    """Unconstrained JSON output turns on JSON mode without a schema."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=legacy_response_schema)
    request = _output_request(format='json', json_schema=_STRUCTURED_SCHEMA, constrained=False)

    cfg = await model._genkit_to_googleai_cfg(request)

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
@pytest.mark.parametrize(
    'json_schema, needle',
    [
        ({'description': 'anything'}, 'response_schema'),
        ({'type': 'object', 'required': ['data'], 'properties': {'data': {'title': 'Data'}}}, 'data'),
        ({'type': 'object', 'properties': {'x': {'allOf': [{'type': 'string'}]}}}, 'no type'),
        ({'type': 'array'}, 'cannot be converted'),
        (to_json_schema(list[Any]), 'cannot be converted'),
        (to_json_schema(tuple[int, str]), 'cannot be converted'),
        (to_json_schema(_Mixed), 'anyOf branch'),
        (to_json_schema(dict[str, int]), 'cannot be converted'),
    ],
)
async def test_gemini_model__legacy_unconvertible_output_schema_raises(
    json_schema: dict[str, Any], needle: str
) -> None:
    """A schema that response_schema cannot hold raises INVALID_ARGUMENT."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=True)
    request = _output_request(format='json', json_schema=json_schema, constrained=True)

    with pytest.raises(GenkitError) as exc_info:
        await model._genkit_to_googleai_cfg(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert needle in exc_info.value.original_message


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_response_schema', [False, True])
@pytest.mark.parametrize('schema_type', ['string', 'enum'])
@pytest.mark.parametrize(
    'enum_output',
    [{'format': 'enum'}, {'content_type': 'text/enum'}, {'format': 'enum', 'content_type': 'text/enum'}],
    ids=['format', 'content_type', 'both'],
)
async def test_gemini_model__enum_output_sets_text_x_enum(
    enum_output: dict[str, str], schema_type: str, legacy_response_schema: bool
) -> None:
    """Constrained enum output sends text/x.enum with a string schema listing the values."""
    model = GeminiModel('version', MagicMock(spec=genai.Client), legacy_response_schema=legacy_response_schema)
    request = _output_request(json_schema={'type': schema_type, 'enum': ['a', 'b']}, constrained=True, **enum_output)

    cfg = await model._genkit_to_googleai_cfg(request)

    assert cfg is not None
    assert cfg.response_mime_type == 'text/x.enum'
    assert cfg.response_schema == genai_types.Schema(type=genai_types.Type.STRING, enum=['a', 'b'])
    assert cfg.response_json_schema is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'enum_output',
    [
        {'format': 'enum'},
        {'content_type': 'text/enum'},
        {'format': 'enum', 'json_schema': {'type': 'string', 'enum': ['a', 'b']}, 'constrained': False},
    ],
    ids=['no-schema', 'content-type-only', 'unconstrained'],
)
async def test_gemini_model__enum_output_without_constrained_schema_sends_neither(
    enum_output: dict[str, Any], gemini_model_instance: GeminiModel
) -> None:
    """Enum output without a constrained schema sends neither text/x.enum nor a schema."""
    cfg = await gemini_model_instance._genkit_to_googleai_cfg(_output_request(**enum_output))

    assert cfg is not None
    assert cfg.response_mime_type is None
    assert cfg.response_schema is None
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
@pytest.mark.parametrize('legacy_response_schema', [False, True])
async def test_generate_json_output_pins_request_body(mocker: MockerFixture, legacy_response_schema: bool) -> None:
    """The client receives JSON mode plus the schema in one response field, and nothing else."""
    version = 'gemini-2.5-flash'
    request = _output_request(
        format='json', json_schema=_STRUCTURED_SCHEMA, constrained=True, content_type='application/json'
    )
    candidate = genai.types.Candidate(content=genai.types.Content(parts=[genai.types.Part(text='{"n": 1}')]))
    googleai_client_mock = mocker.AsyncMock()
    googleai_client_mock.aio.models.generate_content.return_value = genai.types.GenerateContentResponse(
        candidates=[candidate]
    )
    gemini = GeminiModel(version, googleai_client_mock, legacy_response_schema=legacy_response_schema)

    response = await gemini.generate(request, ModelContext())

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
    assert response.message.content[0].text == '{"n": 1}'


@pytest.mark.parametrize(
    ('version', 'expected_schema'),
    [
        ('gemini-2.5-flash-preview-tts', GeminiTtsConfig),
        ('gemini-3.1-flash-tts-preview', GeminiTtsConfig),
        ('gemini-3.8-flash-tts', GeminiTtsConfig),
        ('gemini-3.8-flash-lite-tts', GeminiTtsConfig),
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
    ctx = ModelContext(streaming_callback=mocker.MagicMock())

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
    ctx = ModelContext(streaming_callback=mocker.MagicMock())

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
        await gemini.generate(request, ModelContext())
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

    await GeminiModel(version, client_mock).generate(request, ModelContext())

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

    await GeminiModel(version, client_mock).generate(request, ModelContext())

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

    await GeminiModel(version, client_mock).generate(request, ModelContext())

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

    await GeminiModel(version, client_mock).generate(request, ModelContext())

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

    await GeminiModel('gemini-3.1-flash-tts-preview', client_mock).generate(request, ModelContext())

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

    await GeminiModel('gemini-2.5-flash-preview-tts', client_mock).generate(request, ModelContext())

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

    await GeminiModel('gemini-2.5-flash-preview-tts', client_mock).generate(request, ModelContext())

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
        await gemini.generate(_hi_request(), ModelContext())

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is auth_error


@pytest.mark.asyncio
async def test_streaming_generate_credential_failure_is_unauthenticated(mocker: MockerFixture) -> None:
    """Streaming classifies a revoked token the same way non-streaming does."""
    revoked = RefreshError('invalid_grant: Token has been expired or revoked.')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content_stream.side_effect = revoked
    gemini = GeminiModel('gemini-2.5-flash', client_mock)
    ctx = ModelContext(streaming_callback=mocker.MagicMock())

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ctx)

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is revoked


@pytest.mark.asyncio
async def test_generate_metadata_server_refresh_error_is_unavailable(mocker: MockerFixture) -> None:
    """On Cloud Run or GKE a metadata-server blip is RefreshError from TransportError, retryable=False: UNAVAILABLE."""
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

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext())

    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.cause is blip


@pytest.mark.asyncio
async def test_generate_retryable_refresh_error_is_unavailable(mocker: MockerFixture) -> None:
    """google.auth marked the refresh retryable (token endpoint 503): UNAVAILABLE, so Retry and Fallback act on it."""
    flaky = RefreshError('token endpoint returned 503', retryable=True)
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = flaky
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext())

    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.cause is flaky


@pytest.mark.asyncio
async def test_generate_bare_auth_transport_error_is_unavailable(mocker: MockerFixture) -> None:
    """A google.auth TransportError that reaches the call is UNAVAILABLE, as in Vertex AI Model Garden."""
    blip = TransportError('metadata server unreachable')
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = blip
    gemini = GeminiModel('gemini-2.5-flash', client_mock)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext())

    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.cause is blip


async def _generate_failure(mocker: MockerFixture, failure: BaseException, *, streaming: bool = False) -> GenkitError:
    client_mock = mocker.AsyncMock()
    client_mock.aio.models.generate_content.side_effect = failure
    client_mock.aio.models.generate_content_stream.side_effect = failure
    gemini = GeminiModel('gemini-2.5-flash', client_mock)
    ctx = ModelContext(streaming_callback=mocker.MagicMock()) if streaming else ModelContext()
    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ctx)
    return raised.value


def _quota_error(*, retry_delay: object = None, headers: dict[str, str] | None = None) -> APIError:
    details = [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': retry_delay}] if retry_delay else []
    body = {'error': {'code': 429, 'message': 'Quota exceeded', 'status': 'RESOURCE_EXHAUSTED', 'details': details}}
    response = httpx.Response(429, json=body, headers=headers or {})
    return APIError(429, body, response)


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize(
    'failure',
    [
        httpx.ConnectError('[Errno 61] Connection refused'),
        httpx.RemoteProtocolError('Server disconnected without sending a response.'),
        ConnectionResetError('Connection reset by peer'),
    ],
)
async def test_connection_refused_is_unavailable(mocker: MockerFixture, failure: Exception, streaming: bool) -> None:
    """A refused or dropped connection to Gemini is UNAVAILABLE so retry and fallback try again."""
    error = await _generate_failure(mocker, failure, streaming=streaming)

    assert error.status == 'UNAVAILABLE'
    assert error.cause is failure


@pytest.mark.asyncio
@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('failure', [httpx.ReadTimeout('timed out'), httpx.ConnectTimeout('timed out'), TimeoutError()])
async def test_timeout_is_deadline_exceeded(mocker: MockerFixture, failure: Exception, streaming: bool) -> None:
    """A Gemini call that times out is DEADLINE_EXCEEDED, not a raw transport error."""
    error = await _generate_failure(mocker, failure, streaming=streaming)

    assert error.status == 'DEADLINE_EXCEEDED'
    assert error.cause is failure


@pytest.mark.asyncio
@pytest.mark.parametrize('code', [413, 418])
async def test_unmapped_4xx_is_unknown_genkit_error(mocker: MockerFixture, code: int) -> None:
    """A 4xx with no matching status is an UNKNOWN GenkitError that keeps the SDK error as its cause."""
    api_error = APIError(code, {'error': {'message': 'Request payload too large'}})

    error = await _generate_failure(mocker, api_error)

    assert error.status == 'UNKNOWN'
    assert error.cause is api_error
    assert error.original_message == 'Request payload too large'


@pytest.mark.asyncio
async def test_429_retry_after_header_sets_retry_delay(mocker: MockerFixture) -> None:
    """A Gemini 429 with a Retry-After header tells retry how long to wait."""
    error = await _generate_failure(mocker, _quota_error(headers={'Retry-After': '7'}))

    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.response_metadata == {'retry_after_ms': 7000.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('retry_delay', 'headers', 'expected'),
    [
        ('30s', {}, {'retry_after_ms': 30000.0}),
        ('1.5s', {}, {'retry_after_ms': 1500.0}),
        ('30s', {'Retry-After': '7'}, {'retry_after_ms': 30000.0}),
        ('soon', {}, None),
        (30, {}, None),
    ],
)
async def test_429_retry_info_sets_retry_delay(
    mocker: MockerFixture, retry_delay: object, headers: dict[str, str], expected: dict[str, float] | None
) -> None:
    """A Gemini 429 whose body says RetryInfo "30s" waits 30000 ms, over any header; a malformed delay is ignored."""
    error = await _generate_failure(mocker, _quota_error(retry_delay=retry_delay, headers=headers))

    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.response_metadata == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('code', 'body', 'status'),
    [
        (
            400,
            {'error': {'code': 400, 'message': 'User location is not supported', 'status': 'FAILED_PRECONDITION'}},
            'FAILED_PRECONDITION',
        ),
        (409, {'error': {'code': 409, 'message': 'exists', 'status': 'ALREADY_EXISTS'}}, 'ALREADY_EXISTS'),
        (500, {'error': {'code': 500, 'message': 'lost', 'status': 'DATA_LOSS'}}, 'DATA_LOSS'),
        (500, {'error': {'code': 500, 'message': 'unknown', 'status': 'UNKNOWN'}}, 'INTERNAL'),
        (402, {'error': {'code': 402, 'message': 'Prepay credit depleted', 'status': 'RESOURCE_EXHAUSTED'}}, 'UNKNOWN'),
        (502, {'message': '<html>Bad Gateway</html>', 'status': 'Bad Gateway'}, 'INTERNAL'),
    ],
)
async def test_error_body_status_picks_among_statuses_sharing_the_code(
    mocker: MockerFixture, code: int, body: dict[str, Any], status: str
) -> None:
    """The body's status narrows a shared HTTP code and never overrides a different code."""
    error = await _generate_failure(mocker, APIError(code, body))

    assert error.status == status


@pytest.mark.asyncio
async def test_rejected_api_key_is_unauthenticated(mocker: MockerFixture) -> None:
    """A 400 INVALID_ARGUMENT whose ErrorInfo reason is API_KEY_INVALID is UNAUTHENTICATED."""
    message = 'API key not valid. Please pass a valid API key.'
    localized = {'@type': 'type.googleapis.com/google.rpc.LocalizedMessage', 'locale': 'en-US', 'message': message}
    error_info = {
        '@type': 'type.googleapis.com/google.rpc.ErrorInfo',
        'reason': 'API_KEY_INVALID',
        'domain': 'googleapis.com',
        'metadata': {'service': 'generativelanguage.googleapis.com'},
    }
    details = [localized, error_info]
    body = {'error': {'code': 400, 'message': message, 'status': 'INVALID_ARGUMENT', 'details': details}}
    rejected = APIError(400, body)

    error = await _generate_failure(mocker, rejected)

    assert error.status == 'UNAUTHENTICATED'
    assert error.original_message == message
    assert error.cause is rejected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('code', 'status', 'reason'),
    [
        (403, 'PERMISSION_DENIED', 'API_KEY_SERVICE_BLOCKED'),
        (403, 'PERMISSION_DENIED', 'SERVICE_DISABLED'),
        (429, 'RESOURCE_EXHAUSTED', 'RATE_LIMIT_EXCEEDED'),
    ],
)
async def test_other_error_info_reasons_keep_the_body_status(
    mocker: MockerFixture, code: int, status: str, reason: str
) -> None:
    """An ErrorInfo reason other than API_KEY_INVALID leaves the body's status as it is."""
    error_info = {'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': reason}
    body = {'error': {'code': code, 'message': 'request refused', 'status': status, 'details': [error_info]}}

    error = await _generate_failure(mocker, APIError(code, body))

    assert error.status == status


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


# ---------------------------------------------------------------------------
# Blocked prompts and replies with no candidates
# ---------------------------------------------------------------------------


def _text_chunk(
    text: str, finish_reason: genai_types.FinishReason | None = None
) -> genai_types.GenerateContentResponse:
    candidate = genai_types.Candidate(
        content=genai_types.Content(parts=[genai_types.Part(text=text)]),
        finish_reason=finish_reason,
    )
    return genai_types.GenerateContentResponse(candidates=[candidate])


@pytest.mark.asyncio
async def test_generate_blocked_prompt(mocker: MockerFixture) -> None:
    """A prompt the service refused returns BLOCKED with the reason, the feedback, and usage."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(
        block_reason=genai_types.BlockedReason.SAFETY,
        block_reason_message='unsafe prompt',
    )
    usage = genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=7, total_token_count=7)
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[], prompt_feedback=feedback, usage_metadata=usage
    )
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext())

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'unsafe prompt'
    assert response.message is not None
    assert response.message.role == Role.MODEL
    assert [p.text for p in response.message.content] == ['']
    assert response.candidates == []
    assert response.custom == {'promptFeedback': {'blockReason': 'SAFETY', 'blockReasonMessage': 'unsafe prompt'}}
    assert response.usage is not None
    assert response.usage.input_tokens == 7


@pytest.mark.asyncio
async def test_generate_blocked_prompt_default_message(mocker: MockerFixture) -> None:
    """Without a service message the finish message names the block reason."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(
        block_reason=genai_types.BlockedReason.PROHIBITED_CONTENT
    )
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(prompt_feedback=feedback)
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext())

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'prompt blocked: PROHIBITED_CONTENT'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'feedback',
    [
        None,
        genai_types.GenerateContentResponsePromptFeedback(
            block_reason=genai_types.BlockedReason.BLOCKED_REASON_UNSPECIFIED
        ),
    ],
)
async def test_generate_no_candidates_without_block_is_internal(
    mocker: MockerFixture, feedback: genai_types.GenerateContentResponsePromptFeedback | None
) -> None:
    """No candidates and no block reason, BLOCKED_REASON_UNSPECIFIED included, raise INTERNAL."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(prompt_feedback=feedback)
    gemini = GeminiModel('gemini-2.5-flash', client)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext())
    assert raised.value.status == 'INTERNAL'
    assert raised.value.original_message == 'Model returned no candidates.'
    assert raised.value.cause is None


@pytest.mark.asyncio
async def test_generate_candidate_without_content(mocker: MockerFixture) -> None:
    """A candidate that stopped before producing content is a normal response with its finish reason."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(finish_reason=genai_types.FinishReason.MAX_TOKENS)]
    )
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext())

    assert response.finish_reason == FinishReason.LENGTH
    assert response.message is not None
    assert response.message.content[0].text == ''


@pytest.mark.asyncio
async def test_generate_with_candidates_ignores_block_reason(mocker: MockerFixture) -> None:
    """Candidates win over a block reason in the prompt feedback."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(block_reason=genai_types.BlockedReason.SAFETY)
    answer = _text_chunk('Hello', genai_types.FinishReason.STOP)
    answer.prompt_feedback = feedback
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = answer
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext())

    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    assert response.message.content[0].text == 'Hello'
    assert response.custom == {'promptFeedback': {'blockReason': 'SAFETY'}}


@pytest.mark.asyncio
async def test_generate_without_prompt_feedback_leaves_custom_empty(mocker: MockerFixture) -> None:
    """A response with no prompt feedback keeps the default empty custom payload."""
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = _text_chunk('Hello', genai_types.FinishReason.STOP)
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext())

    assert response.custom == {}


@pytest.mark.asyncio
async def test_streaming_generate_blocked_prompt(mocker: MockerFixture) -> None:
    """A streamed blocked prompt returns BLOCKED, keeps an earlier chunk's block reason, and streams nothing."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(block_reason=genai_types.BlockedReason.SAFETY)
    usage = genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=3, total_token_count=3)

    async def feedback_only() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(prompt_feedback=feedback)
        yield genai_types.GenerateContentResponse(usage_metadata=usage)

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = feedback_only()
    gemini = GeminiModel('gemini-2.5-flash', client)
    on_chunk = mocker.MagicMock()

    response = await gemini.generate(_hi_request(), ModelContext(streaming_callback=on_chunk))

    assert response.finish_reason == FinishReason.BLOCKED
    assert response.finish_message == 'prompt blocked: SAFETY'
    assert response.message is not None
    assert [p.text for p in response.message.content] == ['']
    assert response.custom == {'promptFeedback': {'blockReason': 'SAFETY'}}
    assert response.usage is not None
    assert response.usage.input_tokens == 3
    on_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_generate_no_candidates_without_block_is_internal(mocker: MockerFixture) -> None:
    """Chunks that carry no candidate and no block reason raise INTERNAL."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(
        safety_ratings=[
            genai_types.SafetyRating(
                category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                probability=genai_types.HarmProbability.NEGLIGIBLE,
            )
        ]
    )
    usage = genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=3, total_token_count=3)

    async def no_candidates() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(prompt_feedback=feedback)
        yield genai_types.GenerateContentResponse(usage_metadata=usage)

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = no_candidates()
    gemini = GeminiModel('gemini-2.5-flash', client)
    on_chunk = mocker.MagicMock()

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext(streaming_callback=on_chunk))
    assert raised.value.status == 'INTERNAL'
    assert raised.value.original_message == 'Model returned no candidates.'
    on_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_generate_empty_stream_is_unavailable(mocker: MockerFixture) -> None:
    """A stream that ends without a single chunk raises UNAVAILABLE."""

    async def no_chunks() -> Any:  # noqa: ANN401
        for chunk in ():
            yield chunk

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = no_chunks()
    gemini = GeminiModel('gemini-2.5-flash', client)

    with pytest.raises(GenkitError) as raised:
        await gemini.generate(_hi_request(), ModelContext(streaming_callback=mocker.MagicMock()))
    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.original_message == 'Model stream returned no responses.'


@pytest.mark.asyncio
async def test_streaming_generate_candidate_without_content(mocker: MockerFixture) -> None:
    """A streamed candidate that stopped before producing content is a normal response with its finish reason."""

    async def contentless() -> Any:  # noqa: ANN401
        yield genai_types.GenerateContentResponse(
            candidates=[genai_types.Candidate(finish_reason=genai_types.FinishReason.MAX_TOKENS)]
        )

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = contentless()
    gemini = GeminiModel('gemini-2.5-flash', client)
    on_chunk = mocker.MagicMock()

    response = await gemini.generate(_hi_request(), ModelContext(streaming_callback=on_chunk))

    assert response.finish_reason == FinishReason.LENGTH
    on_chunk.assert_not_called()


@pytest.mark.asyncio
async def test_streaming_generate_block_reason_after_candidates_is_not_blocked(mocker: MockerFixture) -> None:
    """A block reason arriving after streamed candidates does not turn the answer into BLOCKED."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(block_reason=genai_types.BlockedReason.SAFETY)

    async def answer_then_feedback() -> Any:  # noqa: ANN401
        yield _text_chunk('Hello', genai_types.FinishReason.STOP)
        yield genai_types.GenerateContentResponse(prompt_feedback=feedback)

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = answer_then_feedback()
    gemini = GeminiModel('gemini-2.5-flash', client)
    on_chunk = mocker.MagicMock()

    response = await gemini.generate(_hi_request(), ModelContext(streaming_callback=on_chunk))

    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    assert response.message.content[0].text == 'Hello'
    assert response.custom == {'promptFeedback': {'blockReason': 'SAFETY'}}
    assert on_chunk.call_count == 1


@pytest.mark.asyncio
async def test_streaming_generate_exposes_prompt_feedback_without_block(mocker: MockerFixture) -> None:
    """Prompt feedback that only carries safety ratings leaves the answer intact and is exposed in custom."""
    feedback = genai_types.GenerateContentResponsePromptFeedback(
        safety_ratings=[
            genai_types.SafetyRating(
                category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                probability=genai_types.HarmProbability.NEGLIGIBLE,
            )
        ]
    )
    answer = _text_chunk('Hello', genai_types.FinishReason.STOP)
    answer.prompt_feedback = feedback

    async def single_chunk() -> Any:  # noqa: ANN401
        yield answer

    client = mocker.AsyncMock()
    client.aio.models.generate_content_stream.return_value = single_chunk()
    gemini = GeminiModel('gemini-2.5-flash', client)

    response = await gemini.generate(_hi_request(), ModelContext(streaming_callback=mocker.MagicMock()))

    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    assert response.message.content[0].text == 'Hello'
    assert response.custom == {
        'promptFeedback': {'safetyRatings': [{'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE'}]}
    }
