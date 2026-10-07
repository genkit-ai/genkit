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

"""What a Gemini, Veo or Lyria config key does: rejected, sent, or merged via extra.

Each generate test captures the HTTP body the google-genai SDK would send,
so the pins hold on the wire, not on an intermediate config object.
"""

import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GoogleAI, VeoConfig, VertexAI
from genkit_google_genai._models._lyria import LyriaConfig
from google.auth.credentials import AnonymousCredentials
from google.genai import _api_client
from pydantic import BaseModel, ValidationError

from genkit import Genkit, GenkitError

_GEMINI = 'googleai/gemini-2.5-flash'


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


@pytest.mark.parametrize(
    ('config', 'key'),
    [
        ({'thinking_config': {'thinkng_budget': 0}}, 'thinking_config.thinkng_budget'),
        (
            {
                'safety_settings': [
                    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'}
                ]
            },
            'safety_settings.0.method',
        ),
    ],
    ids=['thinking_config', 'safety_settings'],
)
@pytest.mark.asyncio
async def test_generate_gemini_unknown_nested_config_key_raises(sent: _Sent, config: dict[str, Any], key: str) -> None:
    """A typo one level down raises INVALID_ARGUMENT naming the nested path, and no request is sent."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert repr(key) in str(raised.value)
    assert sent.bodies == []


@pytest.mark.asyncio
async def test_generate_operation_veo_unknown_config_key_raises_and_sends_nothing(sent: _Sent) -> None:
    """`config={'durationSecs': 5}` raises INVALID_ARGUMENT naming the key instead of riding into `parameters`."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate_operation(
            model='googleai/veo-3.0-generate-001', prompt='a cat', config={'durationSecs': 5}
        )

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "'durationSecs'" in str(raised.value)
    assert sent.bodies == []


def test_lyria_config_with_unknown_key_raises_validation_error() -> None:
    """`LyriaConfig` used to drop unknown keys silently; now it fails naming the key."""
    with pytest.raises(ValidationError, match='sampleCnt'):
        LyriaConfig.model_validate({'sampleCnt': 2})


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
        await _googleai().generate_operation(model='googleai/veo-3.0-generate-001', prompt='a cat', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "context={'secrets': {'api_key': ...}}" in str(raised.value)
    assert sent.bodies == []


# -- extra --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_operation_veo_extra_merges_into_request_parameters(sent: _Sent) -> None:
    """`numberOfVideos=2` plus `extra={'parameters': {'fooBar': 1}}` sends both under `parameters`."""
    await _googleai().generate_operation(
        model='googleai/veo-3.0-generate-001',
        prompt='a cat',
        config={'numberOfVideos': 2, 'extra': {'parameters': {'fooBar': 1}}},
    )

    assert sent.body['parameters'] == {'sampleCount': 2, 'fooBar': 1}


@pytest.mark.asyncio
async def test_generate_operation_veo_generate_audio_reaches_request(sent: _Sent) -> None:
    """`generateAudio=True` still reaches the request now that unknown keys are rejected (control)."""
    await _vertexai().generate_operation(
        model='vertexai/veo-3.0-generate-001', prompt='a cat', config={'generateAudio': True}
    )

    assert sent.body['parameters'] == {'generateAudio': True}


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
async def test_generate_gemini_google_search_dict_passes_options_to_tool(sent: _Sent) -> None:
    """`{'google_search': {'exclude_domains': [...]}}` passes the options into the tool."""
    await _vertexai().generate(
        model='vertexai/gemini-2.5-flash', prompt='hi', config={'google_search': {'exclude_domains': ['example.com']}}
    )

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


@pytest.mark.asyncio
@pytest.mark.parametrize('old_key', ['google_search_retrieval', 'googleSearchRetrieval'])
async def test_generate_gemini_google_search_retrieval_raises_naming_google_search(sent: _Sent, old_key: str) -> None:
    """The pre-1.0 name (and the JS spelling) fails before sending and names `google_search` as the fix."""
    with pytest.raises(GenkitError) as raised:
        await _googleai().generate(model=_GEMINI, prompt='hi', config={old_key: True})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert f'{old_key} was renamed to google_search' in str(raised.value)
    assert sent.bodies == []


# -- Dev UI -------------------------------------------------------------------


@pytest.mark.parametrize('config_class', [GeminiConfig, GeminiImageConfig, VeoConfig])
def test_config_form_advertises_no_additional_properties(config_class: type[BaseModel]) -> None:
    """The Dev UI config form says `additionalProperties: false`, so it doesn't offer free-form keys."""
    assert config_class.model_json_schema(by_alias=True)['additionalProperties'] is False


def test_gemini_config_form_nested_settings_advertise_no_additional_properties() -> None:
    """Nested settings like `thinkingConfig` are closed in the form too."""
    properties = GeminiConfig.model_json_schema(by_alias=True)['properties']

    assert properties['thinkingConfig']['additionalProperties'] is False
    assert properties['functionCallingConfig']['additionalProperties'] is False


def test_gemini_config_form_shows_google_search_not_google_search_retrieval() -> None:
    """The Dev UI form shows `googleSearch` and not the old name."""
    properties = GeminiConfig.model_json_schema(by_alias=True)['properties']

    assert 'googleSearch' in properties
    assert 'googleSearchRetrieval' not in properties
