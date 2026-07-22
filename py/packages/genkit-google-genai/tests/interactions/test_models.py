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

"""Tests for Interactions-backed Google AI models."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from genkit_google_genai.google import GoogleAI, googleai_name
from genkit_google_genai.models.antigravity import AntigravityModel
from genkit_google_genai.models.deep_research import DeepResearchModel
from genkit_google_genai.models.googleai_lyria import GoogleAILyriaModel
from genkit_google_genai.models.interactions_utils import downgrade_system_messages

from genkit import ActionKind, Message, ModelRequest, Part, Role, TextPart


def test_downgrade_system_messages_maps_system_to_user() -> None:
    messages = [
        Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='Be helpful'))]),
        Message(role=Role.USER, content=[Part(root=TextPart(text='Hi'))]),
    ]
    downgraded = downgrade_system_messages(messages)
    assert downgraded[0].role == Role.USER
    assert downgraded[1].role == Role.USER


@pytest.mark.asyncio
async def test_deep_research_start_sends_background_request() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content.decode())
        return httpx.Response(200, json={'id': 'dr-1', 'status': 'in_progress'})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        model = DeepResearchModel(
            'deep-research-preview-04-2026',
            plugin_api_key='plugin-key',
            client_options={},
        )
        request = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='sys'))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='research this'))]),
            ],
            config={'thinkingSummaries': 'AUTO', 'googleSearch': True},
        )
        with patch(
            'genkit_google_genai.models.deep_research.InteractionsClient',
            wraps=lambda **kwargs: __import__(
                'genkit_google_genai._interactions.client', fromlist=['InteractionsClient']
            ).InteractionsClient(http_client=http_client, **kwargs),
        ):
            operation = await model.start(request, MagicMock())

    body = captured['body']
    assert isinstance(body, dict)
    assert body['background'] is True
    assert body['agent'] == 'deep-research-preview-04-2026'
    assert body['agent_config'] == {
        'type': 'deep-research',
        'thinking_summaries': 'auto',
    }
    assert body['tools'] == [{'type': 'google_search'}]
    assert body['input'][0]['type'] == 'user_input'
    assert operation.id == 'dr-1'
    assert operation.done is False
    assert 'apiKey' not in (operation.metadata or {}).get('clientOptions', {})
    assert 'api_key' not in (operation.metadata or {}).get('clientOptions', {})


@pytest.mark.asyncio
async def test_deep_research_check_rederives_api_key() -> None:
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured['api_key'] = request.headers.get('x-goog-api-key', '')
        return httpx.Response(
            200,
            json={
                'id': 'dr-1',
                'status': 'completed',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'done'}]}],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        model = DeepResearchModel('deep-research-preview-04-2026', plugin_api_key='plugin-key', client_options={})
        from genkit.model import Operation

        operation = Operation.model_construct(
            id='dr-1',
            metadata={'clientOptions': {'base_url': 'https://example.test'}},
        )
        with patch(
            'genkit_google_genai.models.deep_research.InteractionsClient',
            wraps=lambda **kwargs: __import__(
                'genkit_google_genai._interactions.client', fromlist=['InteractionsClient']
            ).InteractionsClient(http_client=http_client, **kwargs),
        ):
            updated = await model.check(operation)

    assert captured['api_key'] == 'plugin-key'
    assert updated.done is True
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].root.text == 'done'


@pytest.mark.asyncio
async def test_antigravity_generate_downgrades_system_and_uses_agent() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                'id': 'ag-1',
                'status': 'completed',
                'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]}],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        model = AntigravityModel('antigravity-preview-05-2026', plugin_api_key='key', client_options={})
        request = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content=[Part(root=TextPart(text='sys'))]),
                Message(role=Role.USER, content=[Part(root=TextPart(text='build'))]),
            ],
            config={'responseModalities': ['TEXT', 'IMAGE']},
        )
        with patch(
            'genkit_google_genai.models.antigravity.InteractionsClient',
            wraps=lambda **kwargs: __import__(
                'genkit_google_genai._interactions.client', fromlist=['InteractionsClient']
            ).InteractionsClient(http_client=http_client, **kwargs),
        ):
            response = await model.generate(request, MagicMock())

    body = captured['body']
    assert isinstance(body, dict)
    assert body['agent'] == 'antigravity-preview-05-2026'
    assert body['response_modalities'] == ['text', 'image']
    assert 'background' not in body
    assert response.message is not None
    assert response.message.content[0].root.text == 'hello'


@pytest.mark.asyncio
async def test_googleai_lyria_defaults_audio_and_text_modalities() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured['body'] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                'id': 'ly-1',
                'status': 'completed',
                'steps': [
                    {
                        'type': 'model_output',
                        'content': [{'type': 'audio', 'data': 'abc', 'mime_type': 'audio/wav'}],
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        model = GoogleAILyriaModel('lyria-3-clip-preview', plugin_api_key='key', client_options={})
        request = ModelRequest(
            messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='jazz riff'))])],
        )
        with patch(
            'genkit_google_genai.models.googleai_lyria.InteractionsClient',
            wraps=lambda **kwargs: __import__(
                'genkit_google_genai._interactions.client', fromlist=['InteractionsClient']
            ).InteractionsClient(http_client=http_client, **kwargs),
        ):
            response = await model.generate(request, MagicMock())

    body = captured['body']
    assert isinstance(body, dict)
    assert body['model'] == 'lyria-3-clip-preview'
    assert body['response_modalities'] == ['audio', 'text']
    assert response.message is not None


@pytest.mark.asyncio
async def test_googleai_plugin_registers_interactions_models() -> None:
    mock_client = MagicMock()
    mock_client.models.list.return_value = iter([])

    with patch('genkit_google_genai.google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')
        actions = await plugin.init()

    kinds_by_name = {action.name: action.kind for action in actions}
    dr_name = googleai_name('deep-research-preview-04-2026')
    ag_name = googleai_name('antigravity-preview-05-2026')
    ly_name = googleai_name('lyria-3-clip-preview')

    assert kinds_by_name[dr_name] == ActionKind.BACKGROUND_MODEL
    assert kinds_by_name[f'{dr_name}/check'] == ActionKind.CHECK_OPERATION
    assert kinds_by_name[f'{dr_name}/cancel'] == ActionKind.CANCEL_OPERATION
    assert kinds_by_name[ag_name] == ActionKind.MODEL
    assert kinds_by_name[ly_name] == ActionKind.MODEL


@pytest.mark.asyncio
async def test_googleai_resolve_routes_interactions_models() -> None:
    mock_client = MagicMock()
    mock_client.models.list.return_value = iter([])

    with patch('genkit_google_genai.google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')

    dr_name = googleai_name('deep-research-pro-preview-12-2025')
    bg = await plugin.resolve(ActionKind.BACKGROUND_MODEL, dr_name)
    assert bg is not None
    assert bg.kind == ActionKind.BACKGROUND_MODEL

    check = await plugin.resolve(ActionKind.CHECK_OPERATION, f'{dr_name}/check')
    assert check is not None

    cancel = await plugin.resolve(ActionKind.CANCEL_OPERATION, f'{dr_name}/cancel')
    assert cancel is not None

    ag = await plugin.resolve(ActionKind.MODEL, googleai_name('antigravity-preview-05-2026'))
    assert ag is not None
    assert ag.kind == ActionKind.MODEL

    ly = await plugin.resolve(ActionKind.MODEL, googleai_name('lyria-3-pro-preview'))
    assert ly is not None


@pytest.mark.asyncio
async def test_googleai_list_actions_includes_interactions_models() -> None:
    mock_client = MagicMock()
    mock_client.models.list.return_value = iter([])

    with patch('genkit_google_genai.google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')
        actions = await plugin.list_actions()

    names = {action.name for action in actions}
    assert googleai_name('deep-research-max-preview-04-2026') in names
    assert googleai_name('antigravity-preview-05-2026') in names
    assert googleai_name('lyria-3-pro-preview') in names
