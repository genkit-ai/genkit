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

"""GoogleAI / VertexAI network arguments: headers, timeout_seconds, http_client."""

import math
from typing import Any

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI, VertexAI
from google.oauth2.credentials import Credentials

from genkit import Genkit
from genkit.plugin_api import GENKIT_CLIENT_HEADER

MODEL_LIST = {'models': [{'name': 'models/gemini-2.5-flash', 'supportedActions': ['generateContent']}]}
VERTEX_MODEL_LIST = {'models': [{'name': 'publishers/google/models/gemini-2.5-flash'}]}
CANDIDATE = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Smoked Salmon Tartine'}]}}]}
ANTIGRAVITY_DONE = {
    'id': 'ag-1',
    'status': 'completed',
    'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'Built it'}]}],
}
GENERATE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'


class FakeGoogleApi:
    """Mock transport that records each request and answers like the Google APIs."""

    def __init__(self, model_list: dict[str, Any] = MODEL_LIST) -> None:
        self.requests: list[httpx.Request] = []
        self.model_list = model_list

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json=self.model_list)
        if request.url.path.endswith('/interactions'):
            return httpx.Response(200, json=ANTIGRAVITY_DONE)
        return httpx.Response(200, json=CANDIDATE)

    def client(self, **kwargs: Any) -> httpx.AsyncClient:  # noqa: ANN401
        return httpx.AsyncClient(transport=httpx.MockTransport(self), **kwargs)

    def urls(self, method: str) -> list[str]:
        return [str(r.url) for r in self.requests if r.method == method]

    def last_post(self) -> httpx.Request:
        return [r for r in self.requests if r.method == 'POST'][-1]


def vertex_ai(**kwargs: Any) -> VertexAI:  # noqa: ANN401
    return VertexAI(project='p', location='us-central1', credentials=Credentials(token='test-token'), **kwargs)


@pytest.mark.asyncio
async def test_google_ai_sends_requests_through_your_http_client() -> None:
    """GoogleAI(http_client=...) lists models and generates through that client."""
    api = FakeGoogleApi()
    plugin = GoogleAI(api_key='k', http_client=api.client())
    ai = Genkit(plugins=[plugin])

    await plugin.init()
    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.text == 'Smoked Salmon Tartine'
    assert any('/models' in url for url in api.urls('GET'))
    assert api.urls('POST') == [GENERATE_URL]


@pytest.mark.asyncio
async def test_vertex_ai_sends_requests_through_your_http_client() -> None:
    """VertexAI(http_client=...) lists models and generates through it, even when a call picks another location."""
    api = FakeGoogleApi(VERTEX_MODEL_LIST)
    plugin = vertex_ai(http_client=api.client())
    ai = Genkit(plugins=[plugin])

    await plugin.init()
    response = await ai.generate(
        model='vertexai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig(location='europe-west1'),
    )

    assert response.text == 'Smoked Salmon Tartine'
    assert any('/publishers/google/models' in url for url in api.urls('GET'))
    assert api.urls('POST') == [
        'https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west1'
        '/publishers/google/models/gemini-2.5-flash:generateContent'
    ]


@pytest.mark.asyncio
async def test_http_client_still_used_when_a_call_overrides_base_url() -> None:
    """A per-call base_url and api_version still go through your http_client."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=api.client())])

    response = await ai.generate(
        model='googleai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig(base_url='https://proxy.example.com', api_version='v1alpha'),
    )

    assert response.text == 'Smoked Salmon Tartine'
    assert api.urls('POST') == ['https://proxy.example.com/v1alpha/models/gemini-2.5-flash:generateContent']


@pytest.mark.asyncio
async def test_antigravity_sends_requests_through_your_http_client() -> None:
    """Antigravity, an Interactions model, also goes through GoogleAI(http_client=...)."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', headers={'x-team': 'search'}, http_client=api.client())])

    response = await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')

    assert response.text == 'Built it'
    assert api.urls('POST') == ['https://generativelanguage.googleapis.com/v1beta/interactions']
    assert api.last_post().headers['x-team'] == 'search'


@pytest.mark.asyncio
async def test_your_headers_are_sent_with_genkit_attribution_appended() -> None:
    """GoogleAI(headers=...) sends your headers as-is and appends Genkit's attribution to User-Agent."""
    api = FakeGoogleApi()
    ai = Genkit(
        plugins=[GoogleAI(api_key='k', headers={'X-Team': 'search', 'User-Agent': 'acme'}, http_client=api.client())]
    )

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    sent = api.last_post().headers
    assert sent['x-team'] == 'search'
    assert len(sent.get_list('user-agent')) == 1
    assert sent['user-agent'].endswith(f'acme {GENKIT_CLIENT_HEADER}')
    assert GENKIT_CLIENT_HEADER in sent['x-goog-api-client']


@pytest.mark.asyncio
async def test_timeout_seconds_reaches_the_request() -> None:
    """GoogleAI(timeout_seconds=30) puts a 30 second timeout on Gemini and Interactions requests."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', timeout_seconds=30, http_client=api.client())])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout == {'connect': 30.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}
    assert interactions_timeout == {'connect': 30.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}


@pytest.mark.asyncio
async def test_timeout_set_on_your_http_client_is_not_used() -> None:
    """Without timeout_seconds, the 1 second timeout on your own client doesn't cut off model calls."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=api.client(timeout=1.0))])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout == {'connect': None, 'read': None, 'write': None, 'pool': None}
    assert interactions_timeout == {'connect': 10.0, 'read': None, 'write': None, 'pool': None}


@pytest.mark.asyncio
async def test_one_http_client_shared_by_two_plugins() -> None:
    """GoogleAI and VertexAI can share one http_client; each sends only its own headers and the client stays open."""
    google_api = FakeGoogleApi()
    vertex_api = FakeGoogleApi(VERTEX_MODEL_LIST)

    def route(request: httpx.Request) -> httpx.Response:
        api = vertex_api if 'aiplatform' in request.url.host else google_api
        return api(request)

    shared = httpx.AsyncClient(transport=httpx.MockTransport(route))
    ai = Genkit(
        plugins=[
            GoogleAI(api_key='k', headers={'x-team': 'search'}, http_client=shared),
            vertex_ai(headers={'x-team': 'ads'}, http_client=shared),
        ]
    )

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    await ai.generate(model='vertexai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert google_api.last_post().headers.get_list('x-team') == ['search']
    assert vertex_api.last_post().headers.get_list('x-team') == ['ads']
    assert google_api.last_post().headers['user-agent'].count(GENKIT_CLIENT_HEADER) == 1
    assert vertex_api.last_post().headers['user-agent'].count(GENKIT_CLIENT_HEADER) == 1
    assert shared.is_closed is False


@pytest.mark.asyncio
async def test_no_network_arguments_sends_attribution_headers_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """GoogleAI(api_key=...) alone sends Gemini's default endpoint with only Genkit's attribution added."""
    api = FakeGoogleApi()

    async def send(_client: httpx.AsyncClient, request: httpx.Request, **_: object) -> httpx.Response:
        return api(request)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    ai = Genkit(plugins=[GoogleAI(api_key='k')])

    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.text == 'Smoked Salmon Tartine'
    assert api.urls('POST') == [GENERATE_URL]
    sent = api.last_post().headers
    assert len(sent.get_list('user-agent')) == 1
    assert sent['user-agent'].endswith(GENKIT_CLIENT_HEADER)
    assert GENKIT_CLIENT_HEADER in sent['x-goog-api-client']
    assert 'x-team' not in sent


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
def test_http_options_argument_raises_type_error(plugin_class: Any) -> None:  # noqa: ANN401
    """GoogleAI(http_options=...) and VertexAI(http_options=...) raise TypeError."""
    removed: dict[str, Any] = {'http_options': {'timeout': 30000}}
    with pytest.raises(TypeError, match='http_options'):
        plugin_class(api_key='k', **removed)


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
def test_debug_config_argument_raises_type_error(plugin_class: Any) -> None:  # noqa: ANN401
    """GoogleAI(debug_config=...) and VertexAI(debug_config=...) raise TypeError."""
    removed: dict[str, Any] = {'debug_config': None}
    with pytest.raises(TypeError, match='debug_config'):
        plugin_class(api_key='k', **removed)


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
def test_timeout_argument_raises_type_error(plugin_class: Any) -> None:  # noqa: ANN401
    """GoogleAI(timeout=30) raises TypeError; the argument is timeout_seconds."""
    removed: dict[str, Any] = {'timeout': 30}
    with pytest.raises(TypeError, match='timeout'):
        plugin_class(api_key='k', **removed)


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
def test_sync_httpx_client_raises_type_error_naming_async_client(plugin_class: Any) -> None:  # noqa: ANN401
    """http_client=httpx.Client() raises TypeError telling you to pass an httpx.AsyncClient."""
    sync_client: Any = httpx.Client()
    with pytest.raises(TypeError, match=r'http_client must be an httpx\.AsyncClient, got Client'):
        plugin_class(api_key='k', http_client=sync_client)


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
@pytest.mark.parametrize('timeout_seconds', [0, -1, math.nan, math.inf])
def test_timeout_seconds_not_positive_and_finite_raises_value_error(
    plugin_class: Any,  # noqa: ANN401
    timeout_seconds: float,
) -> None:
    """timeout_seconds=0, -1, nan or inf raises ValueError at construction."""
    with pytest.raises(ValueError, match='timeout_seconds must be a positive, finite number'):
        plugin_class(api_key='k', timeout_seconds=timeout_seconds)


@pytest.mark.asyncio
async def test_sub_millisecond_timeout_seconds_rounds_up_to_one_millisecond() -> None:
    """timeout_seconds=0.0001 sends a 1 ms timeout on Gemini and Interactions instead of none or zero."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', timeout_seconds=0.0001, http_client=api.client())])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout['read'] == 0.001
    assert interactions_timeout['read'] == 0.001
