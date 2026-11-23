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

"""Which flat Gemini settings combine, and where a bad combination raises.

- Settings that can't both be set raise in the config constructor, so a
  single call fails on the constructor line and a default plus a call fails
  when generate builds the merged config.
- Settings that need a companion, and Vertex-only settings on the Gemini API,
  raise when the request is built: a ModelRef may set one half legitimately.
"""

import json
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest
from genkit_google_genai import GeminiConfig, GeminiImageConfig, GeminiTtsConfig, GoogleAI, VertexAI
from google.auth.credentials import AnonymousCredentials
from google.genai import _api_client
from pydantic import ValidationError

from genkit import Genkit, GenkitError, ModelResponse
from genkit.model import ModelRef

_GEMINI = 'googleai/gemini-2.5-flash'


class _Sent:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []


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
        body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}
        return _api_client.HttpResponse(headers={}, response_stream=[json.dumps(body)])

    with patch.object(_api_client.BaseApiClient, '_async_request', fake_request):
        yield captured


def _failure(response: ModelResponse) -> str:
    """The INVALID_ARGUMENT text a failed generate carries, including per-field details."""
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    details = response.error.details or {}
    return ' '.join([response.error.message, *(e.get('message', '') for e in details.get('errors', []))])


def _googleai() -> Genkit:
    return Genkit(plugins=[GoogleAI(api_key='test-key')])


def _vertexai() -> Genkit:
    return Genkit(plugins=[VertexAI(project='p', location='us-central1', credentials=AnonymousCredentials())])


# -- can't both be set: constructor ---------------------------------------------


def test_thinking_level_and_budget_raise_at_construction() -> None:
    """Both in one config fail on the constructor line, naming the None escape."""
    with pytest.raises(ValidationError, match='thinking_level=None or thinking_budget=None'):
        GeminiConfig(thinking_level='HIGH', thinking_budget=1024)


def test_google_search_false_with_options_raises_at_construction() -> None:
    """An explicit False next to search options is a contradiction."""
    with pytest.raises(ValidationError, match='google_search=False with google_search_exclude_domains set'):
        GeminiConfig(google_search=False, google_search_exclude_domains=['example-reviews.com'])


def test_google_search_options_alone_validate() -> None:
    """Options with google_search left unset are fine; the plugin turns search on."""
    config = GeminiConfig(google_search_exclude_domains=['example-reviews.com'])

    assert config.google_search is None


@pytest.mark.asyncio
async def test_default_budget_plus_call_level_raises_at_generate(sent: _Sent) -> None:
    """A ModelRef budget and a call level each pass alone; the merged config fails before sending."""
    ref = ModelRef(name=_GEMINI, config_schema=GeminiConfig, config=GeminiConfig(thinking_budget=1024))

    with pytest.raises(GenkitError, match='thinking_level=None or thinking_budget=None'):
        await _googleai().generate(model=ref, prompt='hi', config={'thinking_level': 'HIGH'})

    assert sent.bodies == []


@pytest.mark.asyncio
async def test_call_clears_default_budget_with_none(sent: _Sent) -> None:
    """`thinking_budget=None` on the call clears the default, so the level goes out alone."""
    ref = ModelRef(name=_GEMINI, config_schema=GeminiConfig, config=GeminiConfig(thinking_budget=1024))

    await _googleai().generate(model=ref, prompt='hi', config={'thinking_level': 'HIGH', 'thinking_budget': None})

    assert sent.bodies[-1]['generationConfig']['thinkingConfig'] == {'thinking_level': 'HIGH'}


@pytest.mark.asyncio
async def test_call_turns_off_default_search_by_clearing_options(sent: _Sent) -> None:
    """google_search=False plus None for each default option sends no search tool."""
    ref = ModelRef(
        name='vertexai/gemini-2.5-flash',
        config_schema=GeminiConfig,
        config=GeminiConfig(google_search_exclude_domains=['example-reviews.com']),
    )

    await _vertexai().generate(
        model=ref, prompt='hi', config={'google_search': False, 'google_search_exclude_domains': None}
    )

    assert 'tools' not in sent.bodies[-1]


# -- needs a companion: request build -------------------------------------------


@pytest.mark.asyncio
async def test_search_start_time_without_end_time_raises(sent: _Sent) -> None:
    """A time range needs both ends; the Gemini API rejects one."""
    config = GeminiConfig(google_search_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc))

    response = await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    assert re.search('set google_search_start_time and google_search_end_time together', _failure(response))
    assert sent.bodies == []


@pytest.mark.asyncio
async def test_search_time_range_drops_sub_second_precision(sent: _Sent) -> None:
    """`datetime.now()` works: the API rejects sub-second times, so the window is rounded outward."""
    start = datetime(2026, 1, 1, 8, 30, 15, 123456, tzinfo=timezone.utc)
    config = GeminiConfig(google_search_start_time=start, google_search_end_time=start.replace(day=2))

    await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    [tool] = sent.bodies[-1]['tools']
    assert tool['googleSearch']['timeRangeFilter'] == {
        'start_time': '2026-01-01T08:30:15+00:00',
        'end_time': '2026-01-02T08:30:16+00:00',
    }


@pytest.mark.asyncio
async def test_search_time_range_within_one_second_stays_non_empty(sent: _Sent) -> None:
    """Start and end in the same second don't collapse to an empty range."""
    start = datetime(2024, 1, 1, 0, 0, 0, 100, tzinfo=timezone.utc)
    config = GeminiConfig(google_search_start_time=start, google_search_end_time=start.replace(microsecond=900))

    await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    [tool] = sent.bodies[-1]['tools']
    assert tool['googleSearch']['timeRangeFilter'] == {
        'start_time': '2024-01-01T00:00:00+00:00',
        'end_time': '2024-01-01T00:00:01+00:00',
    }


@pytest.mark.asyncio
async def test_search_time_range_whole_second_end_is_unchanged(sent: _Sent) -> None:
    """An end time on a whole second is sent as is."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    config = GeminiConfig(google_search_start_time=start, google_search_end_time=start.replace(hour=12))

    await _googleai().generate(model=_GEMINI, prompt='hi', config=config)

    [tool] = sent.bodies[-1]['tools']
    assert tool['googleSearch']['timeRangeFilter']['end_time'] == '2024-01-01T12:00:00+00:00'


@pytest.mark.asyncio
async def test_file_search_top_k_without_store_names_raises(sent: _Sent) -> None:
    """File Search options with no store to search name the missing field."""
    response = await _googleai().generate(model=_GEMINI, prompt='hi', config=GeminiConfig(file_search_top_k=5))

    assert re.search('file_search_top_k needs file_search_store_names', _failure(response))
    assert sent.bodies == []


@pytest.mark.asyncio
async def test_file_search_empty_store_names_attach_nothing(sent: _Sent) -> None:
    """An untouched Dev UI list (empty strings) leaves File Search off without an error."""
    await _googleai().generate(model=_GEMINI, prompt='hi', config=GeminiConfig(file_search_store_names=['']))

    assert 'tools' not in sent.bodies[-1]


@pytest.mark.asyncio
async def test_google_search_empty_list_options_attach_nothing(sent: _Sent) -> None:
    """Empty google_search_types / google_search_exclude_domains lists neither turn search on nor trip Vertex checks."""
    await _googleai().generate(
        model=_GEMINI,
        prompt='hi',
        config=GeminiConfig(google_search_types=[], google_search_exclude_domains=[]),
    )

    assert 'tools' not in sent.bodies[-1]


@pytest.mark.asyncio
async def test_safety_setting_method_none_passes_on_gemini_api(sent: _Sent) -> None:
    """A SafetySetting entry with method=None is stripped and not rejected as Vertex-only."""
    await _googleai().generate(
        model=_GEMINI,
        prompt='hi',
        config=GeminiConfig(
            safety_settings=[{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_ONLY_HIGH', 'method': None}]
        ),
    )

    assert sent.bodies[-1]['safetySettings'] == [
        {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_ONLY_HIGH'}
    ]


# -- Vertex-only settings on the Gemini API -------------------------------------


@pytest.mark.parametrize(
    ('model', 'config', 'field'),
    [
        (_GEMINI, GeminiConfig(google_search_exclude_domains=['x.com']), 'google_search_exclude_domains'),
        (
            _GEMINI,
            GeminiConfig(google_search_blocking_confidence='BLOCK_HIGH_AND_ABOVE'),
            'google_search_blocking_confidence',
        ),
        (
            _GEMINI,
            GeminiConfig(
                safety_settings=[
                    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_ONLY_HIGH', 'method': 'SEVERITY'}
                ]
            ),
            'safety_settings[].method',
        ),
        ('googleai/gemini-2.5-flash-image', GeminiImageConfig(output_mime_type='image/jpeg'), 'output_mime_type'),
        ('googleai/gemini-2.5-flash-image', GeminiImageConfig(person_generation='ALLOW_ADULT'), 'person_generation'),
    ],
    ids=['exclude_domains', 'blocking_confidence', 'safety_method', 'output_mime_type', 'person_generation'],
)
@pytest.mark.asyncio
async def test_vertex_only_setting_raises_on_gemini_api(
    sent: _Sent, model: str, config: GeminiConfig, field: str
) -> None:
    """The plugin names the flat field instead of letting google-genai raise on the nested one."""
    response = await _googleai().generate(model=model, prompt='hi', config=config)

    assert re.search(f'{re.escape(field)}.*Vertex AI only', _failure(response))
    assert sent.bodies == []


@pytest.mark.parametrize(
    ('config', 'field'),
    [
        (GeminiConfig(file_search_store_names=['fileSearchStores/s1']), 'file_search_store_names'),
        (
            GeminiConfig(file_search_store_names=['fileSearchStores/s1'], file_search_top_k=5),
            'file_search_store_names, file_search_top_k',
        ),
    ],
    ids=['store_names', 'store_names_and_top_k'],
)
@pytest.mark.asyncio
async def test_gemini_api_only_file_search_raises_on_vertex_ai(sent: _Sent, config: GeminiConfig, field: str) -> None:
    """File Search is Gemini-API-only; Vertex AI rejects it naming the flat field."""
    response = await _vertexai().generate(model='vertexai/gemini-2.5-flash', prompt='hi', config=config)

    assert re.search(f'{re.escape(field)}: Gemini API only, not supported on Vertex AI', _failure(response))
    assert sent.bodies == []


# -- TTS voices -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_voice_plus_call_speakers_raises_at_generate(sent: _Sent) -> None:
    """A ModelRef voice and call speakers fail on the merged config, naming the None escape."""
    ref = ModelRef(
        name='googleai/gemini-2.5-flash-preview-tts',
        config_schema=GeminiTtsConfig,
        config=GeminiTtsConfig(voice='Kore'),
    )

    with pytest.raises(GenkitError, match='voice=None or speakers=None'):
        await _googleai().generate(model=ref, prompt='hi', config={'speakers': {'Host': 'Kore', 'Chef': 'Puck'}})

    assert sent.bodies == []
