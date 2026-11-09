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

"""genkit_openai.plugin_api: the hook genkit-vertexai's Model Garden calls."""

import json

import httpx
import pytest
from genkit_openai.plugin_api import openai_compatible_model
from openai import AsyncOpenAI

from genkit import Message, Part, Role
from genkit.model import ModelContext, ModelRequest


@pytest.mark.asyncio
async def test_openai_compatible_model_sends_through_the_given_client() -> None:
    """The returned function posts to the caller's base URL with the given model ID."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 1700000000,
                'model': 'meta/llama-3.3-70b-instruct-maas',
                'choices': [
                    {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Risotto'}}
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncOpenAI(api_key='k', base_url='https://garden.example/v1', http_client=http_client)
        generate = openai_compatible_model('meta/llama-3.3-70b-instruct-maas', client)
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])],
            config={},
        )

        response = await generate(request, ModelContext())

    assert response.text == 'Risotto'
    assert str(seen[0].url) == 'https://garden.example/v1/chat/completions'
    assert json.loads(seen[0].content)['model'] == 'meta/llama-3.3-70b-instruct-maas'
