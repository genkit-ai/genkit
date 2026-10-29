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

"""What a Gemini, Veo or Virtual Try-On config key does: rejected, sent, or merged via extra.

Each generate test captures the HTTP body the google-genai SDK would send,
so the pins hold on the wire, not on an intermediate config object.
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GoogleAI, VeoConfig, VertexAI, VirtualTryOnConfig
from google.auth.credentials import AnonymousCredentials
from google.genai import _api_client
from pydantic import BaseModel

from genkit import Genkit, GenkitError, Message, Part, Role

_GEMINI = 'googleai/gemini-2.5-flash'
_TRY_ON = 'vertexai/virtual-try-on-001'
_TRY_ON_MESSAGES = [
    Message(
        role=Role.USER,
        content=[
            Part.from_media('gs://in/person.png', content_type='image/png', metadata={'type': 'personImage'}),
            Part.from_media('gs://in/shirt.png', content_type='image/png', metadata={'type': 'productImage'}),
        ],
    )
]


class _Sent:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    @property
    def body(self) -> dict[str, Any]:
        assert self.bodies, 'no request reached the API'
        return self.bodies[-1]


@pytest.fixture
def sent() -> Iterator[_Sent]:
    """Capture every request body the SDK sends, and answer with a canned success."""
    captured = _Sent()

    async def fake_request(
        _self: object, http_request: _api_client.HttpRequest, http_options: object = None, stream: bool = False
    ) -> _api_client.HttpResponse:
        if http_request.method.lower() == 'get':
            return _api_client.HttpResponse(headers={}, response_stream=['{}'])
        assert isinstance(http_request.data, dict)
        captured.bodies.append(http_request.data)
        if ':predictLongRunning' in http_request.url:
            body: dict[str, Any] = {'name': 'operations/1', 'done': False}
        elif ':predict' in http_request.url:
            body = {'predictions': [{'bytesBase64Encoded': 'aGk=', 'mimeType': 'image/png'}]}
        else:
            body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}
        return _api_client.HttpResponse(headers={}, response_stream=[json.dumps(body)])

    with patch.object(_api_client.BaseApiClient, '_async_request', fake_request):
        yield captured


def _googleai() -> Genkit:
    return Genkit(plugins=[GoogleAI(api_key='test-key')])


def _vertexai() -> Genkit:
    return Genkit(plugins=[VertexAI(project='p', location='us-central1', credentials=AnonymousCredentials())])


# -- unknown keys -------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_gemini_unknown_safety_setting_key_raises(sent: _Sent) -> None:
    """A typo inside a safety setting raises INVALID_ARGUMENT naming the path, and no request is sent."""
    config = {
        'safety_settings': [
            {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'methd': 'SEVERITY'}
        ]
    }
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "'safety_settings.0.methd'" in str(raised.value)
    assert sent.bodies == []


@pytest.mark.parametrize(
    ('model', 'config', 'message'),
    [
        (
            _GEMINI,
            {'thinking_config': {'thinking_budget': 0}},
            'thinking_config was flattened; use include_thoughts / thinking_budget / thinking_level',
        ),
        (
            _GEMINI,
            {'thinkingConfig': {'thinkingLevel': 'MEDIUM'}},
            'thinkingConfig was flattened; use includeThoughts / thinkingBudget / thinkingLevel',
        ),
        (
            'googleai/gemini-2.5-flash-preview-tts',
            {'speechConfig': {'voiceConfig': {'prebuiltVoiceConfig': {'voiceName': 'Kore'}}}},
            'speechConfig was flattened; use voice / speakers / languageCode',
        ),
        (
            'googleai/gemini-2.5-flash-image',
            {'image_config': {'aspect_ratio': '16:9'}},
            'image_config was flattened; use aspect_ratio / image_size',
        ),
        (
            _GEMINI,
            {'file_search': {'file_search_store_names': ['fileSearchStores/menu']}},
            'file_search was flattened; use file_search_store_names / file_search_metadata_filter / file_search_top_k',
        ),
        (
            _GEMINI,
            {'google_search': {'exclude_domains': ['example.com']}},
            'google_search takes True or False; set options with google_search_types / ',
        ),
        (_GEMINI, {'url_context': {}}, 'url_context takes True or False'),
        (_GEMINI, {'code_execution': {}}, 'code_execution takes True or False'),
        (_GEMINI, {'function_calling_config': {'mode': 'ANY'}}, 'function_calling_config is not a config field'),
        (_GEMINI, {'google_search_retrieval': True}, 'google_search_retrieval was flattened; use google_search'),
        (_GEMINI, {'googleSearchRetrieval': {}}, 'googleSearchRetrieval was flattened; use googleSearch'),
        (
            _GEMINI,
            {'speechConfig': {'voiceConfig': {}}},
            'speechConfig was flattened into voice / speakers / languageCode, which only GeminiTtsConfig has; '
            'build the model ref with gemini_tts()',
        ),
        (
            _GEMINI,
            {'image_config': {'aspect_ratio': '16:9'}},
            'which only GeminiImageConfig has; build the model ref with gemini_image()',
        ),
    ],
    ids=[
        'thinking_config',
        'thinkingConfig',
        'speechConfig',
        'image_config',
        'file_search',
        'google_search-dict',
        'url_context-dict',
        'code_execution-dict',
        'function_calling_config',
        'google_search_retrieval',
        'googleSearchRetrieval',
        'speechConfig-on-gemini',
        'image_config-on-gemini',
    ],
)
@pytest.mark.asyncio
async def test_generate_gemini_nested_key_names_flat_fields(
    sent: _Sent, model: str, config: dict[str, Any], message: str
) -> None:
    """A nested key the flat fields replaced raises INVALID_ARGUMENT naming them, in the caller's casing."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate(model=model, prompt='hi', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert message in str(raised.value)
    assert sent.bodies == []


@pytest.mark.parametrize(
    ('model', 'config', 'wire_path', 'wire_value'),
    [
        (
            'vertexai/gemini-2.5-flash',
            {
                'safety_settings': [
                    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'}
                ]
            },
            ('safetySettings', 0, 'method'),
            'SEVERITY',
        ),
        (
            'vertexai/gemini-2.5-flash-image',
            {'output_mime_type': 'image/jpeg'},
            ('generationConfig', 'imageConfig', 'imageOutputOptions', 'mimeType'),
            'image/jpeg',
        ),
    ],
    ids=['safety_settings.method', 'output_mime_type'],
)
@pytest.mark.asyncio
async def test_generate_gemini_sdk_field_in_nested_setting_reaches_request(
    sent: _Sent, model: str, config: dict[str, Any], wire_path: tuple[str | int, ...], wire_value: str
) -> None:
    """A Vertex-only setting lands on the wire on Vertex AI.

    The Gemini API rejects both; see test_generate_gemini_vertex_only_setting_raises_on_gemini_api.
    """
    await _vertexai().generate(model=model, prompt='hi', config=config)

    value: Any = sent.body
    for part in wire_path:
        value = value[part]
    assert value == wire_value


@pytest.mark.asyncio
async def test_generate_operation_veo_unknown_config_key_raises_and_sends_nothing(sent: _Sent) -> None:
    """`config={'durationSecs': 5}` raises INVALID_ARGUMENT naming the key instead of riding into `parameters`."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate_operation(
            model='googleai/veo-3.1-generate-preview', prompt='a cat', config={'durationSecs': 5}
        )

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "'durationSecs'" in str(raised.value)
    assert sent.bodies == []


# -- api key ------------------------------------------------------------------


@pytest.mark.parametrize(
    'config',
    [{'api_key': 'sk-tenant'}, {'apiKey': 'sk-tenant'}, {'extra': {'api_key': 'sk-tenant'}}],
    ids=['api_key', 'apiKey', 'extra'],
)
@pytest.mark.asyncio
async def test_generate_operation_veo_api_key_in_config_points_to_context_secrets(
    sent: _Sent, config: dict[str, Any]
) -> None:
    """A key in Veo config or `extra` raises INVALID_ARGUMENT pointing at context.secrets, and isn't sent."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate_operation(model='googleai/veo-3.1-generate-preview', prompt='a cat', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "context={'secrets': {'api_key': ...}}" in str(raised.value)
    assert sent.bodies == []


# -- extra --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_operation_veo_extra_merges_into_request_parameters(sent: _Sent) -> None:
    """`numberOfVideos=2` plus `extra={'parameters': {'fooBar': 1}}` sends both under `parameters`."""
    await _googleai().generate_operation(
        model='googleai/veo-3.1-generate-preview',
        prompt='a cat',
        config={'numberOfVideos': 2, 'extra': {'parameters': {'fooBar': 1}}},
    )

    assert sent.body['parameters'] == {'sampleCount': 2, 'fooBar': 1}


@pytest.mark.asyncio
async def test_generate_operation_veo_generate_audio_reaches_request(sent: _Sent) -> None:
    """`generateAudio=True` is a declared field and is sent under `parameters`."""
    await _vertexai().generate_operation(
        model='vertexai/veo-3.1-generate-001', prompt='a cat', config={'generateAudio': True}
    )

    assert sent.body['parameters'] == {'generateAudio': True}


# -- virtual try-on -----------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_virtual_try_on_unknown_config_key_raises_and_sends_nothing(sent: _Sent) -> None:
    """`config={'addWatermark': True}` raises INVALID_ARGUMENT naming the key instead of riding into `parameters`."""
    with pytest.raises(GenkitError) as raised:
        await _vertexai().generate(model=_TRY_ON, messages=_TRY_ON_MESSAGES, config={'addWatermark': True})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "'addWatermark'" in str(raised.value)
    assert sent.bodies == []


@pytest.mark.parametrize(
    'config',
    [{'api_key': 'sk-tenant'}, {'apiKey': 'sk-tenant'}, {'extra': {'api_key': 'sk-tenant'}}],
    ids=['api_key', 'apiKey', 'extra'],
)
@pytest.mark.asyncio
async def test_generate_virtual_try_on_api_key_in_config_points_to_context_secrets(
    sent: _Sent, config: dict[str, Any]
) -> None:
    """A key in try-on config or `extra` raises INVALID_ARGUMENT pointing at context.secrets, and isn't sent."""
    with pytest.raises(GenkitError) as raised:
        await _vertexai().generate(model=_TRY_ON, messages=_TRY_ON_MESSAGES, config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "context={'secrets': {'api_key': ...}}" in str(raised.value)
    assert sent.bodies == []


@pytest.mark.asyncio
async def test_generate_virtual_try_on_extra_merges_into_request_parameters(sent: _Sent) -> None:
    """`sampleCount=2` plus `extra={'parameters': {'addWatermark': True}}` sends both under `parameters`."""
    await _vertexai().generate(
        model=_TRY_ON,
        messages=_TRY_ON_MESSAGES,
        config={'sampleCount': 2, 'extra': {'parameters': {'addWatermark': True}}},
    )

    assert sent.body['parameters'] == {'sampleCount': 2, 'addWatermark': True}


# -- google_search ------------------------------------------------------------


@pytest.mark.parametrize(
    ('backend', 'model', 'config'),
    [
        ('googleai', _GEMINI, {'google_search': True}),
        ('googleai', _GEMINI, {'googleSearch': True}),
        ('vertexai', 'vertexai/gemini-2.5-flash', {'google_search': True}),
    ],
    ids=['googleai-snake', 'googleai-camel', 'vertexai'],
)
@pytest.mark.asyncio
async def test_generate_gemini_google_search_true_attaches_search_tool(
    sent: _Sent, backend: str, model: str, config: dict[str, Any]
) -> None:
    """`google_search=True` (either spelling, either backend) sends `tools=[{'googleSearch': {}}]`."""
    ai = _googleai() if backend == 'googleai' else _vertexai()

    await ai.generate(model=model, prompt='hi', config=config)

    assert sent.body['tools'] == [{'googleSearch': {}}]


@pytest.mark.asyncio
async def test_generate_gemini_google_search_option_attaches_tool_with_options(sent: _Sent) -> None:
    """`google_search_exclude_domains` alone turns search on and passes the option into the tool."""
    await _vertexai().generate(
        model='vertexai/gemini-2.5-flash', prompt='hi', config={'google_search_exclude_domains': ['example.com']}
    )

    # google-genai dumps nested tool options snake_case; Proto JSON accepts either.
    [tool] = sent.body['tools']
    assert list(tool['googleSearch'].values()) == [['example.com']]


@pytest.mark.asyncio
async def test_generate_gemini_google_search_false_attaches_no_tool(sent: _Sent) -> None:
    """`{'google_search': False}` sends no tools."""
    await _googleai().generate(model=_GEMINI, prompt='hi', config={'google_search': False})

    assert 'tools' not in sent.body


@pytest.mark.asyncio
async def test_generate_gemini_google_search_with_function_tools_sends_both(sent: _Sent) -> None:
    """Search plus a user tool sends both tools."""
    ai = _googleai()

    @ai.tool()
    async def lookup(city: str) -> str:
        """Look up a city."""
        return city

    await ai.generate(model=_GEMINI, prompt='hi', tools=[lookup], config={'google_search': True})

    tools = sent.body['tools']
    assert {'googleSearch': {}} in tools
    assert any(t.get('functionDeclarations', [{}])[0].get('name') == 'lookup' for t in tools)


# -- Dev UI -------------------------------------------------------------------


@pytest.mark.parametrize('config_class', [GeminiConfig, GeminiImageConfig, VeoConfig, VirtualTryOnConfig])
def test_config_form_advertises_no_additional_properties(config_class: type[BaseModel]) -> None:
    """The Dev UI config form says `additionalProperties: false`, so it doesn't offer free-form keys."""
    assert config_class.model_json_schema(by_alias=True)['additionalProperties'] is False


def test_gemini_config_form_safety_settings_items_are_closed() -> None:
    """Each safety setting in the form takes category, threshold and method only."""
    properties = GeminiConfig.model_json_schema(by_alias=True)['properties']

    assert properties['safetySettings']['type'] == 'array'
    assert properties['safetySettings']['items']['additionalProperties'] is False


def test_gemini_config_form_literal_lists_are_top_level_arrays() -> None:
    """List-of-Literal fields publish `{type: array, items: {enum}}`, which the Dev UI renders as a multi-select."""
    properties = GeminiConfig.model_json_schema(by_alias=True)['properties']

    assert properties['googleSearchTypes']['items']['enum'] == ['web', 'image']
    assert properties['responseModalities']['items']['enum'] == ['TEXT', 'IMAGE', 'AUDIO']


def test_gemini_config_form_shows_google_search_not_google_search_retrieval() -> None:
    """The Dev UI form shows `googleSearch` and not the old name."""
    properties = GeminiConfig.model_json_schema(by_alias=True)['properties']

    assert 'googleSearch' in properties
    assert 'googleSearchRetrieval' not in properties
