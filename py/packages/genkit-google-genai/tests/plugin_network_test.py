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

"""GoogleAI / VertexAI network arguments: headers, timeout, http_client."""

import asyncio
import math
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI, VertexAI
from google.oauth2.credentials import Credentials

from genkit import FinishReason, Genkit
from genkit.plugin_api import GENKIT_USER_AGENT

MODEL_LIST = {'models': [{'name': 'models/gemini-2.5-flash', 'supportedActions': ['generateContent']}]}
VERTEX_MODEL_LIST = {'models': [{'name': 'publishers/google/models/gemini-2.5-flash'}]}
CANDIDATE = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Smoked Salmon Tartine'}]}}]}
ANTIGRAVITY_DONE = {
    'id': 'ag-1',
    'status': 'completed',
    'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'Built it'}]}],
}
EMBEDDING = {'embeddings': [{'values': [0.1, 0.2, 0.3]}]}
UNAVAILABLE = {'error': {'code': 503, 'message': 'The model is overloaded.', 'status': 'UNAVAILABLE'}}
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
        if 'embedcontent' in request.url.path.lower():
            return httpx.Response(200, json=EMBEDDING)
        return httpx.Response(200, json=CANDIDATE)

    def client(self, **kwargs: Any) -> httpx.AsyncClient:  # noqa: ANN401
        """Build a client over this fake; pass the bound method as ``http_client=``."""
        return httpx.AsyncClient(transport=httpx.MockTransport(self), **kwargs)

    def urls(self, method: str) -> list[str]:
        return [str(r.url) for r in self.requests if r.method == method]

    def last_post(self) -> httpx.Request:
        return [r for r in self.requests if r.method == 'POST'][-1]


class FlakyGoogleApi(FakeGoogleApi):
    """FakeGoogleApi whose first ``failures`` POSTs return 503 UNAVAILABLE."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == 'POST' and self.failures:
            self.failures -= 1
            self.requests.append(request)
            return httpx.Response(503, json=UNAVAILABLE)
        return super().__call__(request)


def vertex_ai(**kwargs: Any) -> VertexAI:  # noqa: ANN401
    return VertexAI(project='p', location='us-central1', credentials=Credentials(token='test-token'), **kwargs)


@pytest.fixture
def local_google_api() -> Iterator[tuple[FakeGoogleApi, str]]:
    """Serve FakeGoogleApi over real HTTP/1.1 keep-alive sockets on localhost."""
    api = FakeGoogleApi()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def _respond(self) -> None:
            body = self.rfile.read(int(self.headers.get('content-length') or 0))
            request = httpx.Request(
                self.command, f'http://local{self.path}', headers=self.headers.items(), content=body
            )
            payload = api(request).content
            self.send_response(200)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        # http.server dispatches on these exact names.
        do_GET = _respond  # noqa: N815
        do_POST = _respond  # noqa: N815

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, ANN401
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield api, f'http://127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.asyncio
async def test_google_ai_sends_requests_through_your_http_client() -> None:
    """GoogleAI(http_client=...) lists models and generates through the client it returns."""
    api = FakeGoogleApi()
    plugin = GoogleAI(api_key='k', http_client=api.client)
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
    plugin = vertex_ai(http_client=api.client)
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
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=api.client)])

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
    ai = Genkit(plugins=[GoogleAI(api_key='k', headers={'x-team': 'search'}, http_client=api.client)])

    response = await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')

    assert response.text == 'Built it'
    assert api.urls('POST') == ['https://generativelanguage.googleapis.com/v1beta/interactions']
    assert api.last_post().headers['x-team'] == 'search'


@pytest.mark.asyncio
async def test_your_headers_are_sent_with_genkit_attribution_appended() -> None:
    """GoogleAI(headers=...) sends your headers as-is and appends Genkit's attribution to User-Agent."""
    api = FakeGoogleApi()
    ai = Genkit(
        plugins=[GoogleAI(api_key='k', headers={'X-Team': 'search', 'User-Agent': 'acme'}, http_client=api.client)]
    )

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    sent = api.last_post().headers
    assert sent['x-team'] == 'search'
    assert len(sent.get_list('user-agent')) == 1
    assert sent['user-agent'].endswith(f'acme {GENKIT_USER_AGENT}')
    assert GENKIT_USER_AGENT in sent['x-goog-api-client']


@pytest.mark.asyncio
async def test_timeout_in_seconds_reaches_the_request() -> None:
    """GoogleAI(timeout=30) puts a 30 second timeout on Gemini and Interactions requests."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', timeout=30, http_client=api.client)])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout == {'connect': 30.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}
    assert interactions_timeout == {'connect': 30.0, 'read': 30.0, 'write': 30.0, 'pool': 30.0}


@pytest.mark.asyncio
async def test_timeout_set_on_your_http_client_is_not_used() -> None:
    """Without timeout, the 1 second timeout on your own client doesn't cut off model calls."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=lambda: api.client(timeout=1.0))])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout == {'connect': None, 'read': None, 'write': None, 'pool': None}
    assert interactions_timeout == {'connect': 10.0, 'read': None, 'write': None, 'pool': None}


@pytest.mark.asyncio
async def test_one_http_client_factory_shared_by_two_plugins() -> None:
    """GoogleAI and VertexAI can share one http_client factory; each sends only its own headers."""
    google_api = FakeGoogleApi()
    vertex_api = FakeGoogleApi(VERTEX_MODEL_LIST)

    def route(request: httpx.Request) -> httpx.Response:
        api = vertex_api if 'aiplatform' in request.url.host else google_api
        return api(request)

    def http_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(route))

    ai = Genkit(
        plugins=[
            GoogleAI(api_key='k', headers={'x-team': 'search'}, http_client=http_client),
            vertex_ai(headers={'x-team': 'ads'}, http_client=http_client),
        ]
    )

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    await ai.generate(model='vertexai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert google_api.last_post().headers.get_list('x-team') == ['search']
    assert vertex_api.last_post().headers.get_list('x-team') == ['ads']
    assert google_api.last_post().headers['user-agent'].count(GENKIT_USER_AGENT) == 1
    assert vertex_api.last_post().headers['user-agent'].count(GENKIT_USER_AGENT) == 1


@pytest.mark.asyncio
async def test_http_client_factory_called_once_per_event_loop() -> None:
    """Gemini, Interactions and per-call override clients on one loop all share the one client the factory built."""
    api = FakeGoogleApi()
    built: list[httpx.AsyncClient] = []

    def http_client() -> httpx.AsyncClient:
        built.append(api.client())
        return built[-1]

    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=http_client)])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    await ai.generate(
        model='googleai/gemini-2.5-flash', prompt='Suggest a dish.', config=GeminiConfig(api_version='v1')
    )
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')

    assert len(built) == 1
    assert len(api.urls('POST')) == 3


def test_http_client_works_across_event_loops(local_google_api: tuple[FakeGoogleApi, str]) -> None:
    """Two asyncio.run calls over a pooled transport each get their own client, so the second loop doesn't fail."""
    api, base_url = local_google_api
    built: list[httpx.AsyncClient] = []

    def http_client() -> httpx.AsyncClient:
        built.append(httpx.AsyncClient(transport=httpx.AsyncHTTPTransport()))
        return built[-1]

    ai = Genkit(plugins=[GoogleAI(api_key='k', base_url=base_url, timeout=5, http_client=http_client)])

    async def order() -> tuple[str, str]:
        dish = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
        page = await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
        assert dish.error is None and page.error is None
        return dish.text, page.text

    assert asyncio.run(order()) == ('Smoked Salmon Tartine', 'Built it')
    assert asyncio.run(order()) == ('Smoked Salmon Tartine', 'Built it')
    assert len(built) == 2
    assert len(api.urls('POST')) == 4


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
    assert sent['user-agent'].endswith(GENKIT_USER_AGENT)
    assert GENKIT_USER_AGENT in sent['x-goog-api-client']
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
@pytest.mark.parametrize('client_class', [httpx.AsyncClient, httpx.Client])
def test_client_instance_raises_type_error_asking_for_a_factory(
    plugin_class: Any,  # noqa: ANN401
    client_class: type[httpx.AsyncClient | httpx.Client],
) -> None:
    """http_client=httpx.AsyncClient() raises TypeError telling you to pass lambda: httpx.AsyncClient(...)."""
    instance: Any = client_class()
    with pytest.raises(TypeError, match=r'Pass http_client=lambda: httpx\.AsyncClient\(\.\.\.\)'):
        plugin_class(api_key='k', http_client=instance)


@pytest.mark.asyncio
async def test_factory_returning_sync_client_fails_naming_async_client() -> None:
    """http_client=httpx.Client builds a sync client, so the first model call fails naming httpx.AsyncClient."""
    sync_factory: Any = httpx.Client
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=sync_factory)])

    with pytest.raises(TypeError, match=r'http_client must return an httpx\.AsyncClient, got Client'):
        await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
@pytest.mark.parametrize('timeout', [0, -1, math.nan, math.inf])
def test_timeout_not_positive_and_finite_raises_value_error(
    plugin_class: Any,  # noqa: ANN401
    timeout: float,
) -> None:
    """timeout=0, -1, nan or inf raises ValueError at construction."""
    with pytest.raises(ValueError, match='timeout must be a positive, finite number of seconds'):
        plugin_class(api_key='k', timeout=timeout)


@pytest.mark.asyncio
async def test_sub_millisecond_timeout_rounds_up_to_one_millisecond() -> None:
    """timeout=0.0001 sends a 1 ms timeout on Gemini and Interactions instead of none or zero."""
    api = FakeGoogleApi()
    ai = Genkit(plugins=[GoogleAI(api_key='k', timeout=0.0001, http_client=api.client)])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    gemini_timeout = api.last_post().extensions['timeout']
    await ai.generate(model='googleai/antigravity-preview-05-2026', prompt='Build a landing page.')
    interactions_timeout = api.last_post().extensions['timeout']

    assert gemini_timeout['read'] == 0.001
    assert interactions_timeout['read'] == 0.001


@pytest.mark.asyncio
async def test_max_retries_retries_an_overloaded_model_call() -> None:
    """GoogleAI(max_retries=1) retries a 503 on generate and returns the second attempt's answer."""
    api = FlakyGoogleApi(failures=1)
    ai = Genkit(plugins=[GoogleAI(api_key='k', max_retries=1, http_client=api.client)])

    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.text == 'Smoked Salmon Tartine'
    assert api.urls('POST') == [GENERATE_URL, GENERATE_URL]


@pytest.mark.asyncio
async def test_max_retries_covers_embeddings() -> None:
    """max_retries also retries ai.embed, which genkit_middleware.Retry doesn't cover."""
    api = FlakyGoogleApi(failures=1)
    ai = Genkit(plugins=[GoogleAI(api_key='k', max_retries=1, http_client=api.client)])

    embeddings = await ai.embed(embedder='googleai/gemini-embedding-001', content='Smoked Salmon Tartine')

    assert embeddings[0].embedding == [0.1, 0.2, 0.3]
    assert len(api.urls('POST')) == 2


@pytest.mark.asyncio
async def test_no_max_retries_makes_one_attempt() -> None:
    """Without max_retries, a 503 fails after one request, as google-genai does by default."""
    api = FlakyGoogleApi(failures=1)
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_client=api.client)])

    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'UNAVAILABLE'
    assert api.urls('POST') == [GENERATE_URL]


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
def test_max_retries_maps_to_attempts_including_the_first(plugin_class: Any) -> None:  # noqa: ANN401
    """max_retries=4 is HttpRetryOptions(attempts=5); max_retries=0 sets no retry options."""
    retrying = plugin_class(api_key='k', max_retries=4)
    no_retries = plugin_class(api_key='k', max_retries=0)

    assert retrying._client_kwargs['http_options'].retry_options.attempts == 5
    assert no_retries._client_kwargs['http_options'].retry_options is None


@pytest.mark.parametrize('plugin_class', [GoogleAI, vertex_ai])
@pytest.mark.parametrize('max_retries', [-1, 1.5, True])
def test_max_retries_not_a_non_negative_int_raises_value_error(
    plugin_class: Any,  # noqa: ANN401
    max_retries: Any,  # noqa: ANN401
) -> None:
    """max_retries=-1, 1.5 or True raises ValueError at construction."""
    with pytest.raises(ValueError, match='max_retries must be a non-negative integer'):
        plugin_class(api_key='k', max_retries=max_retries)
