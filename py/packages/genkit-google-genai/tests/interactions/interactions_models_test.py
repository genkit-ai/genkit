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

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai.google import GoogleAI, googleai_name
from genkit_google_genai.models.antigravity import AntigravityModel
from genkit_google_genai.models.deep_research import (
    DeepResearchModel,
    create_deep_research_background_action,
    deep_research_model,
)
from genkit_google_genai.models.googleai_lyria import GoogleAILyriaModel
from genkit_google_genai.models.interactions_utils import downgrade_system_messages

from genkit import ActionKind, Message, ModelRequest, Part, Role, TextPart
from genkit.model import Operation


def test_downgrade_system_messages_maps_system_to_user() -> None:
    messages = [
        Message(role=Role.SYSTEM, content=[Part(TextPart(text='Be helpful'))]),
        Message(role=Role.USER, content=[Part(TextPart(text='Hi'))]),
    ]
    downgraded = downgrade_system_messages(messages)
    assert downgraded[0].role == Role.USER
    assert downgraded[1].role == Role.USER


def _patch_interactions(
    module: str,
    *,
    create_result: dict[str, Any] | None = None,
    get_result: dict[str, Any] | None = None,
    cancel_result: dict[str, Any] | None = None,
    captured: dict[str, Any] | None = None,
):
    """Patch interactions_client on a model module with fake create/get/cancel."""
    create_calls: list[dict[str, Any]] = []
    get_calls: list[str] = []
    cancel_calls: list[str] = []

    async def create(**kwargs: Any) -> dict[str, Any]:
        create_calls.append(kwargs)
        if captured is not None:
            captured['create'] = kwargs
            captured['api_key'] = captured.get('_api_key')
        return create_result or {'id': 'ix-1', 'status': 'in_progress'}

    async def get(interaction_id: str, **_kwargs: Any) -> dict[str, Any]:
        get_calls.append(interaction_id)
        if captured is not None:
            captured['get'] = interaction_id
            captured['api_key'] = captured.get('_api_key')
        return get_result or {'id': interaction_id, 'status': 'completed', 'steps': []}

    async def cancel(interaction_id: str, **_kwargs: Any) -> dict[str, Any]:
        cancel_calls.append(interaction_id)
        if captured is not None:
            captured['cancel'] = interaction_id
        return cancel_result or {'id': interaction_id, 'status': 'cancelled'}

    mock_client = MagicMock()
    mock_client.aio.interactions.create = AsyncMock(side_effect=create)
    mock_client.aio.interactions.get = AsyncMock(side_effect=get)
    mock_client.aio.interactions.cancel = AsyncMock(side_effect=cancel)
    mock_client.aio.aclose = AsyncMock()

    @asynccontextmanager
    async def fake_interactions_client(**kwargs: Any):
        if captured is not None:
            captured['_api_key'] = kwargs.get('api_key')
            captured['client_options'] = kwargs.get('client_options')
        yield mock_client

    return (
        patch(f'{module}.resolve_interactions_client', fake_interactions_client),
        create_calls,
        get_calls,
        cancel_calls,
    )


@pytest.mark.asyncio
async def test_deep_research_start_sends_background_request() -> None:
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = _patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-1', 'status': 'in_progress'},
        captured=captured,
    )
    model = DeepResearchModel(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options={},
    )
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(TextPart(text='sys'))]),
            Message(role=Role.USER, content=[Part(TextPart(text='research this'))]),
        ],
        config={'thinking_summaries': 'auto', 'google_search': True},
    )
    with patcher:
        operation = await model.start(request, MagicMock())

    body = create_calls[0]
    assert body['background'] is True
    assert body['agent'] == 'deep-research-preview-04-2026'
    assert body['agent_config'] == {
        'type': 'deep-research',
        'thinking_summaries': 'auto',
    }
    assert body['tools'] == [{'type': 'google_search'}]
    input_steps = body['input']
    assert isinstance(input_steps, list)
    assert isinstance(input_steps[0], dict)
    assert input_steps[0]['type'] == 'user_input'
    assert operation.id == 'dr-1'
    assert operation.done is False
    assert (operation.metadata or {}).get('clientOptions', {}).get('api_key') == 'plugin-key'


@pytest.mark.asyncio
async def test_deep_research_check_uses_stored_api_key() -> None:
    captured: dict[str, Any] = {}
    patcher, _, get_calls, _ = _patch_interactions(
        'genkit_google_genai.models.deep_research',
        get_result={
            'id': 'dr-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'done'}]}],
        },
        captured=captured,
    )
    model = DeepResearchModel('deep-research-preview-04-2026', plugin_api_key='plugin-key', client_options={})
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'base_url': 'https://example.test', 'api_key': 'override-key'}},
    )
    with patcher:
        updated = await model.check(operation)

    assert captured['api_key'] == 'override-key'
    assert get_calls == ['dr-1']
    assert updated.done is True
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].root.text == 'done'


@pytest.mark.asyncio
async def test_deep_research_check_falls_back_to_plugin_api_key() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, _ = _patch_interactions(
        'genkit_google_genai.models.deep_research',
        get_result={'id': 'dr-1', 'status': 'in_progress'},
        captured=captured,
    )
    model = DeepResearchModel('deep-research-preview-04-2026', plugin_api_key='plugin-key', client_options={})
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'base_url': 'https://example.test'}},
    )
    with patcher:
        await model.check(operation)

    assert captured['api_key'] == 'plugin-key'


@pytest.mark.asyncio
async def test_antigravity_generate_downgrades_system_and_uses_agent() -> None:
    patcher, create_calls, _, _ = _patch_interactions(
        'genkit_google_genai.models.antigravity',
        create_result={
            'id': 'ag-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]}],
        },
    )
    model = AntigravityModel('antigravity-preview-05-2026', plugin_api_key='key', client_options={})
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(TextPart(text='sys'))]),
            Message(role=Role.USER, content=[Part(TextPart(text='build'))]),
        ],
        config={'response_modalities': ['text', 'image']},
    )
    with patcher:
        response = await model.generate(request, MagicMock())

    body = create_calls[0]
    assert body['agent'] == 'antigravity-preview-05-2026'
    assert body['response_modalities'] == ['text', 'image']
    assert 'background' not in body
    assert response.message is not None
    assert response.message.content[0].root.text == 'hello'


@pytest.mark.asyncio
async def test_googleai_lyria_defaults_audio_and_text_modalities() -> None:
    patcher, create_calls, _, _ = _patch_interactions(
        'genkit_google_genai.models.googleai_lyria',
        create_result={
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
    model = GoogleAILyriaModel('lyria-3-clip-preview', plugin_api_key='key', client_options={})
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='jazz riff'))])],
    )
    with patcher:
        response = await model.generate(request, MagicMock())

    body = create_calls[0]
    assert body['model'] == 'lyria-3-clip-preview'
    assert body['response_modalities'] == ['audio', 'text']
    assert response.message is not None


def test_deep_research_model_ref_is_namespaced() -> None:
    ref = deep_research_model('deep-research-preview-04-2026')
    assert ref.name == 'googleai/deep-research-preview-04-2026'
    assert ref.config_schema is not None


@pytest.mark.asyncio
async def test_deep_research_define_background_model_sets_action() -> None:
    """define_background_model must stamp Operation.action for check/cancel."""
    patcher, _, _, _ = _patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-action-1', 'status': 'in_progress'},
    )
    ref = deep_research_model('deep-research-preview-04-2026')
    bg = create_deep_research_background_action(
        ref,
        plugin_api_key='plugin-key',
        client_options={},
    )
    with patcher:
        operation = await bg.start(
            ModelRequest(messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])]),
        )

    assert isinstance(operation, Operation)
    assert operation.action == f'/background-model/{ref.name}'
    assert operation.id == 'dr-action-1'
    model_meta = (bg.start_action.metadata or {}).get('model')
    assert isinstance(model_meta, dict)
    supports = model_meta.get('supports')
    assert isinstance(supports, dict)
    assert supports.get('longRunning') is True


@pytest.mark.asyncio
async def test_googleai_resolve_model_skips_deep_research_foreground() -> None:
    mock_client = MagicMock()
    mock_client.models.list.return_value = iter([])

    with patch('genkit_google_genai.google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')

    dr_name = googleai_name('deep-research-preview-04-2026')
    assert await plugin.resolve(ActionKind.MODEL, dr_name) is None
    bg = await plugin.resolve(ActionKind.BACKGROUND_MODEL, dr_name)
    assert bg is not None
    assert bg.kind == ActionKind.BACKGROUND_MODEL


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

    # Legacy Vertex name must not fall through to Gemini capabilities metadata.
    ly_legacy = await plugin.resolve(ActionKind.MODEL, googleai_name('lyria-002'))
    assert ly_legacy is not None
    model_meta = (ly_legacy.metadata or {}).get('model')
    assert isinstance(model_meta, dict)
    supports = model_meta.get('supports')
    assert isinstance(supports, dict)
    assert supports.get('media') is True
    assert supports.get('multiturn') is not True


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
