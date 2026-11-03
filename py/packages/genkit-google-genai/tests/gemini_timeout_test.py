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

"""GeminiConfig.timeout: per-request seconds, sent as this request's http_options.timeout in ms."""

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from google.genai._api_client import patch_http_options
from pydantic import ValidationError

from genkit import Genkit, Message, Part, Role
from genkit.model import ModelRequest


async def _cfg(config: GeminiConfig, client_kwargs: dict[str, Any] | None = None) -> genai_types.GenerateContentConfig:
    model = GeminiModel('gemini-2.5-flash', MagicMock(), client_kwargs=client_kwargs)
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])], config=config)
    cfg = await model._genkit_to_googleai_cfg(request=request)
    assert cfg is not None
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize(('seconds', 'ms', 'server_seconds'), [(30, 30000, '30'), (2.007, 2007, '3'), (0.0001, 1, '1')])
async def test_timeout_seconds_become_request_http_options_ms(seconds: float, ms: int, server_seconds: str) -> None:
    """`timeout=30` builds a config whose http_options.timeout is 30000 ms and X-Server-Timeout is '30'."""
    cfg = await _cfg(GeminiConfig(timeout=seconds))

    assert cfg.http_options is not None
    assert cfg.http_options.timeout == ms
    assert cfg.http_options.headers == {'X-Server-Timeout': server_seconds}


@pytest.mark.asyncio
async def test_timeout_is_not_sent_in_the_body() -> None:
    """timeout is transport only: nothing named timeout reaches the generation config or extra_body."""
    cfg = await _cfg(GeminiConfig(timeout=30, temperature=0.2))

    dumped = cfg.model_dump(exclude_none=True, exclude={'http_options'})
    assert 'timeout' not in dumped
    assert cfg.http_options is not None
    assert cfg.http_options.extra_body is None


@pytest.mark.asyncio
async def test_timeout_and_extra_share_the_request_http_options() -> None:
    """`extra` rides on http_options.extra_body; a timeout on the same request keeps it."""
    cfg = await _cfg(GeminiConfig.model_validate({'timeout': 15, 'extra': {'labels': {'team': 'search'}}}))

    assert cfg.http_options is not None
    assert cfg.http_options.timeout == 15000
    assert cfg.http_options.headers == {'X-Server-Timeout': '15'}
    assert cfg.http_options.extra_body == {'labels': {'team': 'search'}}


@pytest.mark.asyncio
async def test_request_timeout_overrides_only_the_plugin_timeout() -> None:
    """google-genai layers request http_options over the client's, so base_url and headers stay."""
    plugin_http = genai_types.HttpOptions(timeout=60000, base_url='https://proxy.example', headers={'x-team': 'a'})
    cfg = await _cfg(GeminiConfig(timeout=15), client_kwargs={'http_options': plugin_http})

    assert cfg.http_options is not None
    effective = patch_http_options(plugin_http, cfg.http_options)
    assert effective.timeout == 15000
    assert effective.base_url == 'https://proxy.example'
    assert effective.headers is not None
    assert effective.headers['x-team'] == 'a'
    assert effective.headers['X-Server-Timeout'] == '15'


@pytest.mark.asyncio
async def test_no_timeout_leaves_http_options_alone() -> None:
    """Without timeout or extra, the request carries no http_options, so the plugin's apply unchanged."""
    cfg = await _cfg(GeminiConfig(temperature=0.2))

    assert cfg.http_options is None


@pytest.mark.asyncio
async def test_generate_and_stream_apply_per_request_timeout_and_server_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate() and generate_stream() override both httpx timeout and X-Server-Timeout above a 10s plugin client."""
    captured: list[httpx.Request] = []
    candidate = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Smoked Salmon Tartine'}]}}]}

    async def send(_client: httpx.AsyncClient, request: httpx.Request, **_: object) -> httpx.Response:
        captured.append(request)
        if 'streamGenerateContent' in request.url.path:
            sse = f'data: {json.dumps(candidate)}\r\n\r\n'
            return httpx.Response(200, headers={'content-type': 'text/event-stream'}, content=sse.encode('utf-8'))
        return httpx.Response(200, json=candidate)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    ai = Genkit(
        plugins=[
            GoogleAI(
                api_key='test-key',
                timeout=10,
            )
        ]
    )

    # 1. Baseline call with no per-request timeout populates X-Server-Timeout='10' on the shared client
    baseline = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    assert baseline.text == 'Smoked Salmon Tartine'

    # 2. generate() with timeout=120 raises both httpx timeout and X-Server-Timeout to 120s
    slow = await ai.generate(
        model='googleai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config=GeminiConfig(timeout=120),
    )
    assert slow.text == 'Smoked Salmon Tartine'

    # 3. generate_stream() with timeout=90 applies 90s on the streaming call
    handle = ai.generate_stream(
        model='googleai/gemini-2.5-flash',
        prompt='Suggest a dish.',
        config={'timeout': 90},
    )
    chunks = [chunk.text async for chunk in handle.stream]
    streamed = await handle.response
    assert chunks == ['Smoked Salmon Tartine']
    assert streamed.text == 'Smoked Salmon Tartine'

    # 4. A subsequent call with no timeout falls back to the plugin's 10s deadline
    after = await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    assert after.text == 'Smoked Salmon Tartine'

    post_calls = [req for req in captured if req.method == 'POST']
    assert [(req.headers['x-server-timeout'], req.extensions['timeout']) for req in post_calls] == [
        ('10', {'connect': 10.0, 'read': 10.0, 'write': 10.0, 'pool': 10.0}),
        ('120', {'connect': 120.0, 'read': 120.0, 'write': 120.0, 'pool': 120.0}),
        ('90', {'connect': 90.0, 'read': 90.0, 'write': 90.0, 'pool': 90.0}),
        ('10', {'connect': 10.0, 'read': 10.0, 'write': 10.0, 'pool': 10.0}),
    ]


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan'), True, 'soon'])
def test_timeout_must_be_positive_finite_seconds(bad: object) -> None:
    """0, negative, inf, NaN, bool and non-numbers raise at validation."""
    with pytest.raises(ValidationError, match='timeout'):
        GeminiConfig.model_validate({'timeout': bad})


def test_timeout_is_in_the_dev_ui_schema() -> None:
    """The Dev UI form shows timeout with its seconds description."""
    schema = GeminiConfig.model_json_schema(by_alias=True)
    assert 'seconds' in schema['properties']['timeout']['description']
