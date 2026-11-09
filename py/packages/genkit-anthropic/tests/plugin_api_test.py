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

"""genkit_anthropic.plugin_api: the hooks genkit-vertexai's Model Garden calls."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from genkit_anthropic._model_info import get_model_info
from genkit_anthropic.plugin_api import anthropic_model, anthropic_model_info

from genkit import Message, Part, Role
from genkit.model import ModelContext, ModelRequest


@pytest.mark.asyncio
async def test_anthropic_model_sends_through_the_given_client() -> None:
    """The returned function posts to the caller's base URL with the given model ID."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-5',
                'content': [{'type': 'text', 'text': 'Risotto'}],
                'stop_reason': 'end_turn',
                'stop_sequence': None,
                'usage': {'input_tokens': 3, 'output_tokens': 1},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncAnthropic(api_key='k', base_url='https://garden.example', http_client=http_client, max_retries=0)
        generate = anthropic_model('claude-sonnet-4-5', client)
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])],
            config={},
        )

        response = await generate(request, ModelContext())

    assert response.text == 'Risotto'
    assert seen[0].url.host == 'garden.example'
    assert json.loads(seen[0].content)['model'].startswith('claude-sonnet-4-5')


@pytest.mark.parametrize('model', ['claude-sonnet-4-5', 'claude-not-in-catalog'])
def test_anthropic_model_info_matches_the_catalog(model: str) -> None:
    """Known IDs return the catalog entry; unknown IDs return the defaults."""
    assert anthropic_model_info(model) == get_model_info(model)
