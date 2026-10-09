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

"""Integer sampling knobs reach the generateContent body as JSON integers."""

import json
from typing import Any

import httpx
import pytest
from genkit_google_genai import GeminiConfig, GoogleAI, VertexAI

from genkit import Genkit
from genkit.plugin_api import Plugin

_OK = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}


@pytest.mark.parametrize(
    ('plugin', 'model'),
    [
        (lambda: GoogleAI(api_key='fake-key'), 'googleai/gemini-2.5-flash'),
        (lambda: VertexAI(api_key='fake-key'), 'vertexai/gemini-2.5-flash'),
    ],
    ids=['googleai', 'vertexai'],
)
@pytest.mark.parametrize(
    'config',
    [
        GeminiConfig(top_k=40, max_output_tokens=500, candidate_count=1, seed=7, logprobs=2),
        {'topK': 40, 'maxOutputTokens': 500, 'candidateCount': 1, 'seed': 7, 'logprobs': 2},
    ],
    ids=['typed', 'dict'],
)
@pytest.mark.asyncio
async def test_generate_sends_integer_top_k(
    monkeypatch: pytest.MonkeyPatch,
    plugin: Any,
    model: str,
    config: Any,  # noqa: ANN401
) -> None:
    """`top_k=40` is sent as `"topK": 40`; google-genai types it as float and would send 40.0."""
    bodies: list[dict[str, Any]] = []

    async def send(client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:  # noqa: ANN401
        if request.method == 'GET':
            return httpx.Response(200, json={'models': []}, request=request)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_OK, request=request)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    p: Plugin = plugin()
    ai = Genkit(plugins=[p])

    await ai.generate(model=model, prompt='hi', config=config)

    gen = bodies[-1]['generationConfig']
    assert gen['topK'] == 40
    for key in ('topK', 'maxOutputTokens', 'candidateCount', 'seed', 'logprobs'):
        assert type(gen[key]) is int, key
