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

"""Config classes expose Genkit types only, and still send what the SDK types sent.

Enum fields are Literals of the google-genai enum values and nested messages
are Genkit mirrors. These tests pin the Literals to the installed SDK and check
the values that need conversion (bytes, datetimes) on the HTTP body.
"""

import enum
import json
import types
from collections.abc import Iterator
from typing import Any, Literal, get_args, get_origin
from unittest.mock import patch

import pytest
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GeminiTtsConfig, GemmaConfig, VeoConfig, VertexAI
from genkit_google_genai._models import _gemini, _veo
from google.auth.credentials import AnonymousCredentials
from google.genai import _api_client, types as genai_types
from pydantic import BaseModel

from genkit import Genkit

_PUBLIC_CONFIGS: list[type[BaseModel]] = [GeminiConfig, GeminiTtsConfig, GeminiImageConfig, GemmaConfig, VeoConfig]


@pytest.mark.parametrize(
    ('literal', 'sdk_enum'),
    [
        (_gemini.HarmBlockMethod, genai_types.HarmBlockMethod),
        (_gemini.ProminentPeople, genai_types.ProminentPeople),
        (_gemini.PhishBlockThreshold, genai_types.PhishBlockThreshold),
        (_veo.VideoCompressionQuality, genai_types.VideoCompressionQuality),
        (_veo.ImageResizeMode, genai_types.ImageResizeMode),
    ],
    ids=lambda v: getattr(v, '__name__', None),
)
def test_config_literal_lists_exactly_the_sdk_enum_values(literal: object, sdk_enum: type[enum.Enum]) -> None:
    """Each Literal alias equals its google-genai enum's value set, so a new SDK value fails CI."""
    assert set(get_args(literal)) == {e.value for e in sdk_enum}


def _reachable(annotation: object, seen: set[type]) -> None:
    """Collect every class a field annotation can hold, through unions, containers and nested models."""
    origin = get_origin(annotation)
    if origin is Literal:
        return
    if origin is not None or isinstance(annotation, types.UnionType):
        for arg in get_args(annotation):
            _reachable(arg, seen)
        return
    if not isinstance(annotation, type) or annotation in seen:
        return
    seen.add(annotation)
    if issubclass(annotation, BaseModel):
        for field in annotation.model_fields.values():
            _reachable(field.annotation, seen)


@pytest.mark.parametrize('config_class', _PUBLIC_CONFIGS, ids=lambda c: c.__name__)
def test_public_config_fields_hold_no_google_genai_types(config_class: type[BaseModel]) -> None:
    """No field, at any depth, is annotated with a google.genai class."""
    seen: set[type] = set()
    _reachable(config_class, seen)

    assert sorted(c.__qualname__ for c in seen if c.__module__.startswith('google.')) == []


@pytest.mark.parametrize('config_class', _PUBLIC_CONFIGS, ids=lambda c: c.__name__)
def test_public_config_schema_defs_are_genkit_classes(config_class: type[BaseModel]) -> None:
    """Every $defs entry in the Dev UI schema is a Genkit class reachable from the config."""
    seen: set[type] = set()
    _reachable(config_class, seen)
    genkit_names = {c.__name__ for c in seen if c.__module__.startswith('genkit_google_genai.')}

    defs = set(config_class.model_json_schema(by_alias=True).get('$defs', {}))

    assert defs <= genkit_names


def test_tool_options_build_the_sdk_tool() -> None:
    """A google_search options dict dumps to kwargs genai_types.GoogleSearch accepts unchanged."""
    config = GeminiConfig.model_validate({
        'googleSearch': {
            'excludeDomains': ['spam.example'],
            'timeRangeFilter': {'startTime': '2024-01-01T00:00:00Z'},
            'searchTypes': {'webSearch': {}},
            'blockingConfidence': 'BLOCK_LOW_AND_ABOVE',
        }
    })

    dumped = config.model_dump(exclude_none=True)['google_search']
    tool = genai_types.Tool(google_search=genai_types.GoogleSearch(**dumped))

    assert tool.google_search == genai_types.GoogleSearch.model_validate({
        'excludeDomains': ['spam.example'],
        'timeRangeFilter': {'startTime': '2024-01-01T00:00:00Z'},
        'searchTypes': {'webSearch': {}},
        'blockingConfidence': 'BLOCK_LOW_AND_ABOVE',
    })


@pytest.mark.parametrize(
    'replicated',
    [
        {'voiceSampleAudio': 'YWJj', 'consentAudio': 'ZGVm'},
        {'voice_sample_audio': b'abc', 'consent_audio': b'def'},
    ],
    ids=['base64-str', 'raw-bytes'],
)
def test_replicated_voice_audio_decodes_like_the_sdk(replicated: dict[str, Any]) -> None:
    """Base64 strings decode once and raw bytes pass through, matching genai_types.ReplicatedVoiceConfig."""
    config = GeminiTtsConfig.model_validate({'speechConfig': {'voiceConfig': {'replicatedVoiceConfig': replicated}}})

    dumped = config.model_dump(exclude_none=True)['speech_config']
    sent = genai_types.SpeechConfig(**dumped).voice_config
    assert sent is not None

    assert sent.replicated_voice_config == genai_types.ReplicatedVoiceConfig.model_validate(replicated)
    assert sent.replicated_voice_config is not None
    assert sent.replicated_voice_config.voice_sample_audio == b'abc'


def test_replicated_voice_audio_json_round_trip() -> None:
    """JSON input decodes base64 and JSON output re-encodes it."""
    voice = _gemini.ReplicatedVoiceConfig.model_validate_json('{"voiceSampleAudio": "YWJj"}')

    assert voice.voice_sample_audio == b'abc'
    assert voice.model_dump(mode='json', by_alias=True, exclude_none=True) == {'voiceSampleAudio': 'YWJj'}


def test_sdk_enum_member_still_validates_as_its_literal() -> None:
    """A google-genai enum member passed at runtime is stored as its plain string value."""
    image = _gemini.ImageConfig.model_validate({'prominent_people': genai_types.ProminentPeople.BLOCK_PROMINENT_PEOPLE})

    assert image.prominent_people == 'BLOCK_PROMINENT_PEOPLE'
    assert type(image.prominent_people) is str


# -- on the wire ---------------------------------------------------------------


@pytest.fixture
def sent_bodies() -> Iterator[list[dict[str, Any]]]:
    """Capture every request body the SDK sends, and answer with a canned success."""
    bodies: list[dict[str, Any]] = []

    async def fake_request(
        _self: object, http_request: _api_client.HttpRequest, http_options: object = None, stream: bool = False
    ) -> _api_client.HttpResponse:
        assert isinstance(http_request.data, dict)
        bodies.append(http_request.data)
        if ':predictLongRunning' in http_request.url:
            body: dict[str, Any] = {'name': 'operations/1', 'done': False}
        else:
            body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}
        return _api_client.HttpResponse(headers={}, response_stream=[json.dumps(body)])

    with patch.object(_api_client.BaseApiClient, '_async_request', fake_request):
        yield bodies


def _vertexai() -> Genkit:
    return Genkit(plugins=[VertexAI(project='p', location='us-central1', credentials=AnonymousCredentials())])


@pytest.mark.asyncio
async def test_generate_google_search_time_range_reaches_request(sent_bodies: list[dict[str, Any]]) -> None:
    """`timeRangeFilter` with an ISO start time is sent as the SDK sends an Interval."""
    await _vertexai().generate(
        model='vertexai/gemini-2.5-flash',
        prompt='hi',
        config={
            'google_search': {
                'exclude_domains': ['spam.example'],
                'time_range_filter': {'start_time': '2024-01-01T00:00:00Z'},
            }
        },
    )

    # Key spelling inside googleSearch is the SDK's business (it varies by
    # backend); pin the values.
    [tool] = sent_bodies[-1]['tools']
    exclude_domains, time_range = tool['googleSearch'].values()
    assert exclude_domains == ['spam.example']
    [start_time] = time_range.values()
    assert start_time.startswith('2024-01-01T00:00:00')


@pytest.mark.asyncio
async def test_generate_tts_replicated_voice_sends_audio_base64_once(sent_bodies: list[dict[str, Any]]) -> None:
    """`voiceSampleAudio='YWJj'` goes out as 'YWJj', not base64 of the base64 text."""
    await _vertexai().generate(
        model='vertexai/gemini-2.5-flash-preview-tts',
        prompt='hi',
        config={
            'speechConfig': {
                'voiceConfig': {'replicatedVoiceConfig': {'voiceSampleAudio': 'YWJj', 'mimeType': 'audio/wav'}}
            }
        },
    )

    voice = sent_bodies[-1]['generationConfig']['speechConfig']['voiceConfig']['replicatedVoiceConfig']
    assert voice['voiceSampleAudio'] == 'YWJj'


@pytest.mark.asyncio
async def test_generate_operation_veo_literal_fields_reach_request(sent_bodies: list[dict[str, Any]]) -> None:
    """`compressionQuality` and `resizeMode` strings are sent as the SDK enum values."""
    await _vertexai().generate_operation(
        model='vertexai/veo-3.0-generate-001',
        prompt='a cat',
        config={'compressionQuality': 'LOSSLESS', 'resizeMode': 'PAD'},
    )

    parameters = sent_bodies[-1]['parameters']
    assert parameters['compressionQuality'] == 'LOSSLESS'
    assert parameters['resizeMode'] == 'PAD'
