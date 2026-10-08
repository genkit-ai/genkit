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

"""User httpx clients in ``http_options`` carry every plugin request.

google-genai lets callers pass their own httpx clients (proxies, mTLS,
custom transports). The plugin must use them as-is, not deep-copy them, and
per-request overrides must not leak back into the caller's ``HttpOptions``.
"""

import json
from typing import Any, cast

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI, VertexAI
from google.genai import types as genai_types
from google.oauth2.credentials import Credentials

from genkit import Genkit

_MODEL_LIST = {'models': [{'name': 'models/gemini-2.5-flash', 'supportedActions': ['generateContent']}]}
_VERTEX_MODEL_LIST = {'models': [{'name': 'publishers/google/models/gemini-2.5-flash'}]}
_CANDIDATE = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Smoked Salmon Tartine'}]}}]}


class _Recorder:
    """httpx transport handler that records requests and answers with canned bodies."""

    def __init__(self, model_list: dict[str, Any]) -> None:
        self.requests: list[httpx.Request] = []
        self._model_list = model_list

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json=self._model_list)
        return httpx.Response(200, json=_CANDIDATE)

    def urls(self, method: str) -> list[str]:
        return [str(r.url) for r in self.requests if r.method == method]


def _user_options(recorder: _Recorder) -> genai_types.HttpOptions:
    return genai_types.HttpOptions(
        headers={'x-tenant': 'bistro-42'},
        extra_body={'labels': {'team': 'kitchen'}},
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(recorder)),
    )


def _assert_untouched(opts: genai_types.HttpOptions, client: httpx.AsyncClient | None) -> None:
    assert opts.base_url is None
    assert opts.api_version is None
    assert opts.headers == {'x-tenant': 'bistro-42'}
    assert opts.extra_body == {'labels': {'team': 'kitchen'}}
    assert opts.httpx_async_client is client


@pytest.mark.asyncio
async def test_googleai_routes_discovery_and_generate_through_user_client() -> None:
    """GoogleAI sends models.list and generateContent through the caller's httpx client."""
    recorder = _Recorder(_MODEL_LIST)
    opts = _user_options(recorder)
    user_client = opts.httpx_async_client
    plugin = GoogleAI(api_key='k', http_options=opts)
    ai = Genkit(plugins=[plugin])

    await plugin.init()
    response = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.text == 'Smoked Salmon Tartine'
    assert any('/models' in url for url in recorder.urls('GET'))
    assert any(url.endswith('/models/gemini-2.5-flash:generateContent') for url in recorder.urls('POST'))
    sent = recorder.requests[-1]
    assert sent.headers['x-tenant'] == 'bistro-42'
    assert 'genkit-python' in sent.headers['x-goog-api-client']
    _assert_untouched(opts, user_client)


@pytest.mark.asyncio
async def test_googleai_request_overrides_reuse_user_client_without_mutation() -> None:
    """Per-request base_url/api_version build a temp client on the same transport.

    The caller's HttpOptions keeps its original fields afterwards.
    """
    recorder = _Recorder(_MODEL_LIST)
    opts = _user_options(recorder)
    user_client = opts.httpx_async_client
    ai = Genkit(plugins=[GoogleAI(api_key='k', http_options=opts)])

    await ai.generate(
        model='googleai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig.model_validate({'base_url': 'https://proxy.example.com', 'api_version': 'v1alpha'}),
    )

    assert recorder.urls('POST') == ['https://proxy.example.com/v1alpha/models/gemini-2.5-flash:generateContent']
    _assert_untouched(opts, user_client)


@pytest.mark.asyncio
async def test_googleai_dict_http_options_with_user_client() -> None:
    """Dict-form http_options with a user httpx client works the same way."""
    recorder = _Recorder(_MODEL_LIST)
    user_client = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    headers = {'x-tenant': 'bistro-42'}
    ai = Genkit(
        plugins=[GoogleAI(api_key='k', http_options=cast(Any, {'headers': headers, 'httpx_async_client': user_client}))]
    )

    await ai.generate(
        model='googleai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig.model_validate({'api_version': 'v1alpha'}),
    )

    assert recorder.urls('POST') == [
        'https://generativelanguage.googleapis.com/v1alpha/models/gemini-2.5-flash:generateContent'
    ]
    assert headers == {'x-tenant': 'bistro-42'}


def test_sync_httpx_client_is_accepted() -> None:
    """A user sync httpx client survives plugin construction by reference."""
    sync_client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    opts = genai_types.HttpOptions(httpx_client=sync_client)
    plugin = GoogleAI(api_key='k', http_options=opts)
    assert plugin._client_kwargs['http_options'].httpx_client is sync_client
    assert opts.headers is None


@pytest.mark.asyncio
async def test_vertexai_routes_discovery_and_generate_through_user_client() -> None:
    """VertexAI sends models.list and generateContent through the caller's httpx client."""
    recorder = _Recorder(_VERTEX_MODEL_LIST)
    opts = _user_options(recorder)
    user_client = opts.httpx_async_client
    plugin = VertexAI(
        project='p', location='us-central1', credentials=Credentials(token='test-token'), http_options=opts
    )
    ai = Genkit(plugins=[plugin])

    await plugin.init()
    await ai.generate(
        model='vertexai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig.model_validate({'api_version': 'v1beta1', 'location': 'europe-west1'}),
    )

    assert any('/publishers/google/models' in url for url in recorder.urls('GET'))
    assert recorder.urls('POST') == [
        'https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/p/locations/europe-west1'
        '/publishers/google/models/gemini-2.5-flash:generateContent'
    ]
    _assert_untouched(opts, user_client)


def test_copy_http_options_shares_transport_and_copies_containers() -> None:
    """copy_http_options keeps client identity but gives the copy its own dicts."""
    from genkit_google_genai._models._sdk_config import copy_http_options

    client = httpx.AsyncClient()
    opts = genai_types.HttpOptions(
        headers={'x-tenant': 'bistro-42'},
        extra_body={'labels': {'team': 'kitchen'}},
        httpx_async_client=client,
    )

    copied = copy_http_options(opts)
    assert copied.headers is not None
    assert copied.extra_body is not None
    copied.headers['x-tenant'] = 'other'
    copied.extra_body['labels']['team'] = 'other'
    copied.base_url = 'https://proxy.example.com'

    assert copied.httpx_async_client is client
    assert json.dumps(opts.model_dump(exclude={'httpx_async_client'}, exclude_none=True)) == json.dumps({
        'headers': {'x-tenant': 'bistro-42'},
        'extra_body': {'labels': {'team': 'kitchen'}},
    })
