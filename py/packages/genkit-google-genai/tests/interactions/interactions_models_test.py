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

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai._interactions.converters import split_system_instruction
from genkit_google_genai._interactions.options import ClientOptions
from genkit_google_genai.google import GoogleAI, VertexAI, googleai_name
from genkit_google_genai.models.antigravity import AntigravityConfig, create_antigravity_action
from genkit_google_genai.models.deep_research import (
    DeepResearchConfig,
    create_deep_research_background_action,
    deep_research_model,
    response_format_from_request,
)
from genkit_google_genai.models.interactions_lyria import LyriaConfig, create_lyria_action
from genkit_google_genai.models.interactions_registry import deep_research_model_info
from google.genai.interactions import Interaction

from genkit import ActionKind, GenkitError, Message, ModelRequest, Part, Role, TextPart
from genkit.model import Operation


def test_split_system_instruction_folds_system_turns() -> None:
    messages = [
        Message(role=Role.SYSTEM, content=[Part(TextPart(text='Be helpful'))]),
        Message(role=Role.USER, content=[Part(TextPart(text='Hi'))]),
        Message(role=Role.SYSTEM, content=[Part(TextPart(text='Be terse'))]),
    ]
    instruction, turns = split_system_instruction(messages)
    assert instruction == 'Be helpful\n\nBe terse'
    assert [message.role for message in turns] == [Role.USER]


def test_split_system_instruction_without_system_turns() -> None:
    messages = [Message(role=Role.USER, content=[Part(TextPart(text='Hi'))])]
    instruction, turns = split_system_instruction(messages)
    assert instruction is None
    assert turns == messages


def patch_interactions(
    module: str,
    *,
    create_result: dict[str, Any] | None = None,
    get_result: dict[str, Any] | None = None,
    cancel_result: dict[str, Any] | None = None,
    captured: dict[str, Any] | None = None,
):
    """Patch raw HTTP Interactions helpers on a model module."""
    create_calls: list[dict[str, Any]] = []
    get_calls: list[str] = []
    cancel_calls: list[str] = []

    async def create(
        api_key: str,
        body: dict[str, Any],
        client_options: ClientOptions | None = None,
    ) -> Interaction:
        create_calls.append(body)
        if captured is not None:
            captured['create'] = body
            captured['api_key'] = api_key
            captured['client_options'] = client_options
        return Interaction.model_validate(create_result or {'id': 'ix-1', 'status': 'in_progress'})

    async def get(
        api_key: str,
        interaction_id: str,
        client_options: ClientOptions | None = None,
    ) -> Interaction:
        get_calls.append(interaction_id)
        if captured is not None:
            captured['get'] = interaction_id
            captured['api_key'] = api_key
            captured['client_options'] = client_options
        return Interaction.model_validate(get_result or {'id': interaction_id, 'status': 'completed', 'steps': []})

    async def cancel(
        api_key: str,
        interaction_id: str,
        client_options: ClientOptions | None = None,
    ) -> Interaction:
        cancel_calls.append(interaction_id)
        if captured is not None:
            captured['cancel'] = interaction_id
            captured['api_key'] = api_key
            captured['client_options'] = client_options
        return Interaction.model_validate(cancel_result or {'id': interaction_id, 'status': 'cancelled'})

    patches: dict[str, Any] = {
        'create_interaction': AsyncMock(side_effect=create),
    }
    # Only deep_research imports get/cancel; patch those when present.
    if module.endswith('deep_research'):
        patches['get_interaction'] = AsyncMock(side_effect=get)
        patches['cancel_interaction'] = AsyncMock(side_effect=cancel)

    return (
        patch.multiple(module, **patches),
        create_calls,
        get_calls,
        cancel_calls,
    )


@pytest.mark.asyncio
async def test_deep_research_start_sends_background_request() -> None:
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-1', 'status': 'in_progress'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part(TextPart(text='sys'))]),
            Message(role=Role.USER, content=[Part(TextPart(text='research this'))]),
        ],
        config={'thinking_summaries': 'auto', 'google_search': True},
    )
    with patcher:
        operation = await action.start(request)

    body = create_calls[0]
    assert body['background'] is True
    assert body['agent'] == 'deep-research-preview-04-2026'
    assert body['agent_config'] == {
        'type': 'deep-research',
        'thinking_summaries': 'auto',
    }
    assert body['tools'] == [{'type': 'google_search'}]
    # Deep Research rejects system_instruction; system text lands as a leading input step.
    assert 'system_instruction' not in body
    assert body['input'] == [
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'sys'}]},
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'research this'}]},
    ]
    assert operation.id == 'dr-1'
    assert operation.done is False
    persisted = (operation.metadata or {}).get('clientOptions') or {}
    assert 'apiKey' not in persisted
    assert 'baseUrl' not in persisted


@pytest.mark.asyncio
async def test_deep_research_check_reads_secrets_not_ticket() -> None:
    captured: dict[str, Any] = {}
    patcher, _, get_calls, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        get_result={
            'id': 'dr-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'done'}]}],
        },
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'baseUrl': 'https://evil.test', 'apiKey': 'ticket-key'}},
    )
    with patcher:
        updated = await action.check(operation, context={'secrets': {'api_key': 'tenant-key'}})

    assert captured['api_key'] == 'tenant-key'
    assert captured['client_options'].base_url is None
    assert get_calls == ['dr-1']
    assert updated.done is True
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].root.text == 'done'
    assert isinstance(updated.output, type(updated.output))
    assert updated.output.message.role == 'model'


@pytest.mark.asyncio
async def test_deep_research_cancel_reads_secrets_not_ticket() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, cancel_calls = patch_interactions(
        'genkit_google_genai.models.deep_research',
        cancel_result={'id': 'dr-1', 'status': 'cancelled'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'baseUrl': 'https://evil.test', 'apiKey': 'ticket-key'}},
    )
    with patcher:
        updated = await action.cancel(operation, context={'secrets': {'api_key': 'tenant-key'}})

    assert captured['api_key'] == 'tenant-key'
    assert captured['client_options'].base_url is None
    assert cancel_calls == ['dr-1']
    assert updated.done is True
    assert updated.id == 'dr-1'


@pytest.mark.asyncio
async def test_deep_research_check_falls_back_to_plugin_api_key() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        get_result={'id': 'dr-1', 'status': 'in_progress'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'baseUrl': 'https://example.test'}},
    )
    with patcher:
        await action.check(operation)

    assert captured['api_key'] == 'plugin-key'


@pytest.mark.asyncio
async def test_deep_research_passes_previous_interaction_id() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-2', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='follow up'))])],
        config={'previous_interaction_id': 'v1_prior'},
    )
    with patcher:
        await action.start(request)

    assert create_calls[0]['previous_interaction_id'] == 'v1_prior'


@pytest.mark.asyncio
async def test_deep_research_rejects_config_api_key() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-key', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='context.secrets') as exc_info:
            await action.start(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])],
                    config={'api_key': 'request-key'},
                )
            )
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert create_calls == []


@pytest.mark.asyncio
async def test_deep_research_start_uses_context_secrets() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-secret', 'status': 'in_progress'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])])
    with patcher:
        operation = await action.start_action.run(request, context={'secrets': {'api_key': 'tenant-key'}})

    assert captured['api_key'] == 'tenant-key'
    persisted = (operation.response.metadata or {}).get('clientOptions') or {}
    assert 'apiKey' not in persisted


@pytest.mark.asyncio
async def test_antigravity_passes_previous_interaction_id() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.antigravity',
        create_result={
            'id': 'ag-2',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'ok'}]}],
        },
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(
            ModelRequest[AntigravityConfig](
                messages=[Message(role=Role.USER, content=[Part(TextPart(text='continue'))])],
                config=AntigravityConfig(previous_interaction_id='v1_prior'),
            )
        )

    assert create_calls[0]['previous_interaction_id'] == 'v1_prior'


@pytest.mark.asyncio
async def test_antigravity_rejects_empty_messages() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.antigravity',
        create_result={'id': 'ag-empty', 'status': 'completed', 'steps': []},
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='Missing input') as exc_info:
            await action.run(ModelRequest(messages=[]))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert create_calls == []


@pytest.mark.asyncio
async def test_deep_research_rejects_empty_messages() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-empty', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='Missing input') as exc_info:
            await action.start(ModelRequest(messages=[]))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert create_calls == []


@pytest.mark.asyncio
async def test_antigravity_generate_folds_system_and_uses_agent() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.antigravity',
        create_result={
            'id': 'ag-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]}],
        },
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    request = ModelRequest[AntigravityConfig](
        messages=[
            Message(role=Role.SYSTEM, content=[Part(TextPart(text='sys'))]),
            Message(role=Role.USER, content=[Part(TextPart(text='build'))]),
        ],
        config=AntigravityConfig(response_modalities=['text', 'image']),
    )
    with patcher:
        response = await action.run(request)

    body = create_calls[0]
    assert body['agent'] == 'antigravity-preview-05-2026'
    assert body['response_modalities'] == ['text', 'image']
    assert 'background' not in body
    assert body['environment'] == {'type': 'remote'}
    # Antigravity rejects system_instruction; system text lands as a leading input step.
    assert 'system_instruction' not in body
    assert body['input'] == [
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'sys'}]},
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'build'}]},
    ]
    assert response.response.message is not None
    assert response.response.message.content[0].root.text == 'hello'


def test_bare_model_request_accepts_lyria_config_instance() -> None:
    """Bare ModelRequest(config=LyriaConfig(...)) should not reject the plugin schema."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='riff'))])],
        config=LyriaConfig(response_modalities=['audio']),
    )
    assert isinstance(request.config, LyriaConfig)
    assert request.config.response_modalities == ['audio']


@pytest.mark.asyncio
async def test_lyria_defaults_audio_and_text_modalities() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.interactions_lyria',
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
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='jazz riff'))])],
    )
    with patcher:
        response = await action.run(request)

    body = create_calls[0]
    assert body['model'] == 'lyria-3-clip-preview'
    assert body['response_modalities'] == ['audio', 'text']
    assert response.response.message is not None


@pytest.mark.asyncio
async def test_lyria_passes_through_unknown_config_fields() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.interactions_lyria',
        create_result={
            'id': 'ly-2',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'ok'}]}],
        },
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(
            ModelRequest(
                messages=[Message(role=Role.USER, content=[Part(TextPart(text='riff'))])],
                config={'temperature': 0.4},
            )
        )

    body = create_calls[0]
    assert body['temperature'] == 0.4
    assert 'api_key' not in body
    assert 'apiKey' not in body


def test_deep_research_model_ref_is_namespaced() -> None:
    ref = deep_research_model('deep-research-preview-04-2026')
    assert ref.name == 'googleai/deep-research-preview-04-2026'
    assert ref.config_schema is not None
    assert ref.info == deep_research_model_info('deep-research-preview-04-2026')


def test_googleai_family_constructors() -> None:
    dr = GoogleAI.deep_research_model('deep-research-preview-04-2026')
    assert dr.name == 'googleai/deep-research-preview-04-2026'
    assert dr.config_schema is DeepResearchConfig
    assert dr.info == deep_research_model_info('deep-research-preview-04-2026')

    ag = GoogleAI.antigravity_model('antigravity-preview-05-2026')
    assert ag.name == 'googleai/antigravity-preview-05-2026'
    assert ag.config_schema is AntigravityConfig

    ly = GoogleAI.lyria_model('lyria-3-clip-preview')
    assert ly.name == 'googleai/lyria-3-clip-preview'
    assert ly.config_schema is LyriaConfig


@pytest.mark.asyncio
async def test_deep_research_background_action_sets_action() -> None:
    """Start stamps Operation.action so check/cancel can find the companions."""
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-action-1', 'status': 'in_progress'},
    )
    ref = deep_research_model('deep-research-preview-04-2026')
    bg = create_deep_research_background_action(
        ref,
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
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


@pytest.mark.asyncio
async def test_vertex_keeps_interactions_families_fail_closed() -> None:
    mock_client = MagicMock()
    mock_client.models.list.return_value = iter([])

    with patch('genkit_google_genai.google.genai.client.Client', return_value=mock_client):
        plugin = VertexAI(project='p', location='us-central1')

    assert await plugin.resolve(ActionKind.MODEL, 'vertexai/deep-research-preview-04-2026') is None
    assert await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'vertexai/deep-research-preview-04-2026') is None
    assert await plugin.resolve(ActionKind.MODEL, 'vertexai/antigravity-preview-05-2026') is None
    assert await plugin.resolve(ActionKind.MODEL, 'vertexai/lyria-3-clip-preview') is None
    assert await plugin.resolve(ActionKind.MODEL, 'vertexai/lyria-002') is None


@pytest.mark.asyncio
async def test_deep_research_file_search_and_mcp_dump_snake_case() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-tools', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])],
        config={
            'file_search': {'file_search_store_names': ['stores/one']},
            'mcp_servers': [{'name': 'docs', 'url': 'https://mcp.example', 'allowed_tools': ['search']}],
        },
    )
    with patcher:
        await action.start(request)

    tools = create_calls[0]['tools']
    assert {'type': 'file_search', 'file_search_store_names': ['stores/one']} in tools
    assert {
        'type': 'mcp_server',
        'name': 'docs',
        'url': 'https://mcp.example',
        'allowed_tools': ['search'],
    } in tools
    assert 'fileSearchStoreNames' not in tools[0]
    assert 'allowedTools' not in tools[1]


def test_response_format_from_request_keeps_caller_schema() -> None:
    schema = {'type': 'object', 'properties': {'title': {'type': 'string'}}}
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])],
        output={'format': 'json', 'schema': schema},
    )
    assert response_format_from_request(request) == {
        'type': 'text',
        'mime_type': 'application/json',
        'schema': schema,
    }


def test_deep_research_accepts_uppercase_choice_labels() -> None:
    config = DeepResearchConfig.model_validate({
        'thinking_summaries': 'AUTO',
        'visualization': 'OFF',
        'response_modalities': ['TEXT', 'IMAGE'],
    })
    assert config.thinking_summaries == 'auto'
    assert config.visualization == 'off'
    assert config.response_modalities == ['text', 'image']


@pytest.mark.asyncio
async def test_lyria_system_only_is_enough() -> None:
    """A system prompt is enough to start a clip — no user turn required."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.interactions_lyria',
        create_result={
            'id': 'ly-sys',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'ok'}]}],
        },
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(ModelRequest(messages=[Message(role=Role.SYSTEM, content=[Part(TextPart(text='play jazz'))])]))

    assert create_calls[0]['system_instruction'] == 'play jazz'
    assert create_calls[0]['input'] == []


@pytest.mark.asyncio
async def test_antigravity_rejects_config_api_key() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.antigravity',
        create_result={'id': 'ag-key', 'status': 'completed', 'steps': []},
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='context.secrets'):
            await action.run(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part(TextPart(text='hi'))])],
                    config={'api_key': 'nope'},
                )
            )
    assert create_calls == []


@pytest.mark.asyncio
async def test_lyria_rejects_config_api_key() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.interactions_lyria',
        create_result={'id': 'ly-key', 'status': 'completed', 'steps': []},
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='context.secrets'):
            await action.run(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part(TextPart(text='riff'))])],
                    config={'api_key': 'nope'},
                )
            )
    assert create_calls == []
