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

from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai._google import GenaiModels, GoogleAI, VertexAI, googleai_name
from genkit_google_genai._interactions._converters import split_system_instruction
from genkit_google_genai._interactions._options import ClientOptions
from genkit_google_genai._models._antigravity import AntigravityConfig, create_antigravity_action
from genkit_google_genai._models._deep_research import (
    DeepResearchConfig,
    create_deep_research_background_action,
    deep_research_model,
    response_format_from_request,
)
from genkit_google_genai._models._interactions_lyria import LyriaConfig, create_lyria_action
from genkit_google_genai._models._interactions_registry import deep_research_model_info, lyria_model_info
from google.genai.interactions import Interaction
from pydantic import BaseModel, ValidationError

from genkit import Genkit, GenkitError, Message, Operation, Part, Role
from genkit.model import ModelRequest
from genkit.plugin_api import ActionKind


async def _empty_model_pager() -> AsyncIterator[Any]:
    for model in ():
        yield model


def _set_empty_async_model_list(mock_client: MagicMock) -> None:
    mock_client.aio.models.list = AsyncMock(side_effect=_empty_model_pager)


def test_split_system_instruction_folds_system_turns() -> None:
    messages = [
        Message(role=Role.SYSTEM, content=[Part.from_text('Be helpful')]),
        Message(role=Role.USER, content=[Part.from_text('Hi')]),
        Message(role=Role.SYSTEM, content=[Part.from_text('Be terse')]),
    ]
    instruction, turns = split_system_instruction(messages)
    assert instruction == 'Be helpful\n\nBe terse'
    assert [message.role for message in turns] == [Role.USER]


def test_split_system_instruction_without_system_turns() -> None:
    messages = [Message(role=Role.USER, content=[Part.from_text('Hi')])]
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
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-1', 'status': 'in_progress'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(
            base_url='https://plugin.example',
            api_version='v1',
            timeout=1000,
            custom_headers={'x-request-id': 'plugin'},
        ),
    )
    request = ModelRequest(
        messages=[
            Message(role=Role.SYSTEM, content=[Part.from_text('sys')]),
            Message(role=Role.USER, content=[Part.from_text('research this')]),
        ],
        config={
            'thinking_summaries': 'auto',
            'google_search': True,
            'base_url': 'https://start.example',
            'api_version': 'v1',
            'timeout': 1500,
            'custom_headers': {'x-request-id': 'start'},
        },
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
    assert not operation.metadata
    options = captured['client_options']
    assert options.base_url == 'https://start.example'
    assert options.api_version == 'v1'
    assert options.timeout == 1500
    assert options.custom_headers == {'x-request-id': 'start'}
    assert 'base_url' not in body
    assert 'api_version' not in body
    assert 'timeout' not in body
    assert 'custom_headers' not in body


@pytest.mark.asyncio
async def test_deep_research_check_reads_secrets_not_ticket() -> None:
    captured: dict[str, Any] = {}
    patcher, _, get_calls, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
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
        client_options=ClientOptions(
            base_url='https://plugin.example',
            api_version='v1',
            timeout=1000,
            custom_headers={'x-request-id': 'plugin'},
        ),
    )
    operation = Operation.model_construct(
        id='dr-1',
        metadata={'clientOptions': {'baseUrl': 'https://evil.test', 'apiKey': 'ticket-key'}},
    )
    with patcher:
        updated = await action.check(
            operation,
            context={
                'secrets': {'api_key': 'tenant-key'},
                'config': {
                    'base_url': 'https://poll.example',
                    'api_version': 'v1beta',
                    'timeout': 2000,
                    'custom_headers': {'x-request-id': 'check'},
                },
            },
        )

    assert captured['api_key'] == 'tenant-key'
    options = captured['client_options']
    assert options.base_url == 'https://poll.example'
    assert options.api_version == 'v1beta'
    assert options.timeout == 2000
    assert options.custom_headers == {'x-request-id': 'check'}
    assert get_calls == ['dr-1']
    assert not updated.metadata
    assert updated.done is True
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].text == 'done'
    assert isinstance(updated.output, type(updated.output))
    assert updated.output.message.role == 'model'


@pytest.mark.asyncio
async def test_deep_research_cancel_reads_secrets_not_ticket() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, cancel_calls = patch_interactions(
        'genkit_google_genai._models._deep_research',
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
        updated = await action.cancel(
            operation,
            context={
                'secrets': {'api_key': 'tenant-key'},
                'config': {
                    'base_url': 'https://cancel.example',
                    'api_version': 'v1alpha',
                    'timeout': 2500,
                    'custom_headers': {'x-request-id': 'cancel'},
                },
            },
        )

    assert captured['api_key'] == 'tenant-key'
    options = captured['client_options']
    assert options.base_url == 'https://cancel.example'
    assert options.api_version == 'v1alpha'
    assert options.timeout == 2500
    assert options.custom_headers == {'x-request-id': 'cancel'}
    assert cancel_calls == ['dr-1']
    assert updated.done is True
    assert updated.id == 'dr-1'
    assert not updated.metadata


@pytest.mark.asyncio
async def test_deep_research_check_falls_back_to_plugin_api_key() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
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
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-2', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('follow up')])],
        config={'previous_interaction_id': 'v1_prior'},
    )
    with patcher:
        await action.start(request)

    assert create_calls[0]['previous_interaction_id'] == 'v1_prior'


@pytest.mark.asyncio
async def test_deep_research_rejects_config_api_key() -> None:
    """`config={'api_key': ...}` on Deep Research is an unknown key and starts no job."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-key', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='api_key') as exc_info:
            await action.start(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part.from_text('q')])],
                    config={'api_key': 'request-key'},
                )
            )
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert create_calls == []


@pytest.mark.asyncio
async def test_deep_research_start_uses_context_secrets() -> None:
    captured: dict[str, Any] = {}
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-secret', 'status': 'in_progress'},
        captured=captured,
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('q')])])
    with patcher:
        operation = await action.start_action.run(request, context={'secrets': {'api_key': 'tenant-key'}})

    assert captured['api_key'] == 'tenant-key'
    persisted = (operation.response.metadata or {}).get('clientOptions') or {}
    assert 'apiKey' not in persisted


@pytest.mark.asyncio
async def test_antigravity_passes_previous_interaction_id() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
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
                messages=[Message(role=Role.USER, content=[Part.from_text('continue')])],
                config=AntigravityConfig(previous_interaction_id='v1_prior'),
            )
        )

    assert create_calls[0]['previous_interaction_id'] == 'v1_prior'


@pytest.mark.asyncio
async def test_antigravity_rejects_empty_messages() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
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
        'genkit_google_genai._models._deep_research',
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
        'genkit_google_genai._models._antigravity',
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
            Message(role=Role.SYSTEM, content=[Part.from_text('sys')]),
            Message(role=Role.USER, content=[Part.from_text('build')]),
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
    assert response.response.message.content[0].text == 'hello'


@pytest.mark.asyncio
async def test_antigravity_keeps_custom_environment() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
        create_result={
            'id': 'ag-env',
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
            ModelRequest(
                messages=[Message(role=Role.USER, content=[Part.from_text('build')])],
                config=AntigravityConfig(environment={'type': 'custom', 'name': 'custom-env'}),
            )
        )

    assert create_calls[0]['environment'] == {'type': 'custom', 'name': 'custom-env'}


def test_bare_model_request_accepts_lyria_config_instance() -> None:
    """Bare ModelRequest(config=LyriaConfig(...)) should not reject the plugin schema."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('riff')])],
        config=LyriaConfig(response_modalities=['audio']),
    )
    assert isinstance(request.config, LyriaConfig)
    assert request.config.response_modalities == ['audio']


@pytest.mark.asyncio
async def test_lyria_defaults_audio_and_text_modalities() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
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
        messages=[Message(role=Role.USER, content=[Part.from_text('jazz riff')])],
    )
    with patcher:
        response = await action.run(request)

    body = create_calls[0]
    assert body['model'] == 'lyria-3-clip-preview'
    assert body['response_modalities'] == ['audio', 'text']
    assert response.response.message is not None


@pytest.mark.asyncio
async def test_lyria_extra_temperature_lands_in_create_body() -> None:
    """`extra={'temperature': 0.4}` on the Lyria action is sent on the create body."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
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
                messages=[Message(role=Role.USER, content=[Part.from_text('riff')])],
                config={'extra': {'temperature': 0.4}},
            )
        )

    body = create_calls[0]
    assert body['temperature'] == 0.4
    assert 'extra' not in body
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
    ly_supports = lyria_model_info('lyria-3-clip-preview').supports
    assert ly_supports is not None
    assert ly_supports.system_role is False
    assert GoogleAI.lyria_model('lyria-002').name == 'googleai/lyria-002'

    with_config = GoogleAI.deep_research_model(
        'deep-research-preview-04-2026',
        config=DeepResearchConfig(thinking_summaries='auto'),
    )
    assert isinstance(with_config.config, DeepResearchConfig)
    assert with_config.config.thinking_summaries == 'auto'


def test_package_root_lyria_config_is_interactions() -> None:
    from genkit_google_genai import LyriaConfig as RootLyriaConfig

    ref = GoogleAI.lyria_model(
        'lyria-3-clip-preview',
        config=RootLyriaConfig(response_modalities=['audio']),
    )
    assert ref.config_schema is LyriaConfig
    assert isinstance(ref.config, RootLyriaConfig)


@pytest.mark.asyncio
async def test_deep_research_background_action_sets_action() -> None:
    """Start stamps Operation.action so check/cancel can find the companions."""
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
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
            ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('q')])]),
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
    _set_empty_async_model_list(mock_client)

    with patch('genkit_google_genai._google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')

    dr_name = 'deep-research-preview-04-2026'
    assert await plugin.resolve(ActionKind.MODEL, dr_name) is None
    bg = await plugin.resolve(ActionKind.BACKGROUND_MODEL, dr_name)
    assert bg is not None
    assert bg.kind == ActionKind.BACKGROUND_MODEL


@pytest.mark.asyncio
async def test_googleai_plugin_registers_interactions_models() -> None:
    mock_client = MagicMock()
    _set_empty_async_model_list(mock_client)

    with patch('genkit_google_genai._google.genai.client.Client', return_value=mock_client):
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
    _set_empty_async_model_list(mock_client)

    with patch('genkit_google_genai._google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')

    dr_name = 'deep-research-pro-preview-12-2025'
    bg = await plugin.resolve(ActionKind.BACKGROUND_MODEL, dr_name)
    assert bg is not None
    assert bg.kind == ActionKind.BACKGROUND_MODEL

    check = await plugin.resolve(ActionKind.CHECK_OPERATION, f'{dr_name}/check')
    assert check is not None

    cancel = await plugin.resolve(ActionKind.CANCEL_OPERATION, f'{dr_name}/cancel')
    assert cancel is not None

    ag = await plugin.resolve(ActionKind.MODEL, 'antigravity-preview-05-2026')
    assert ag is not None
    assert ag.kind == ActionKind.MODEL

    ly = await plugin.resolve(ActionKind.MODEL, 'lyria-3-pro-preview')
    assert ly is not None

    # Unknown lyria-* ids still resolve here so a version we have not
    # catalogued is not minted as Gemini.
    ly_passthrough = await plugin.resolve(ActionKind.MODEL, 'lyria-002')
    assert ly_passthrough is not None
    model_meta = (ly_passthrough.metadata or {}).get('model')
    assert isinstance(model_meta, dict)
    supports = model_meta.get('supports')
    assert isinstance(supports, dict)
    assert supports.get('media') is True
    assert supports.get('multiturn') is not True


@pytest.mark.asyncio
async def test_googleai_list_actions_includes_interactions_models() -> None:
    mock_client = MagicMock()
    _set_empty_async_model_list(mock_client)

    with patch('genkit_google_genai._google.genai.client.Client', return_value=mock_client):
        plugin = GoogleAI(api_key='test-key')
        actions = await plugin.list_actions()

    names = {action.name for action in actions}
    assert googleai_name('deep-research-max-preview-04-2026') in names
    assert googleai_name('antigravity-preview-05-2026') in names
    assert googleai_name('lyria-3-pro-preview') in names


@pytest.mark.asyncio
async def test_vertex_keeps_interactions_families_fail_closed() -> None:
    mock_client = MagicMock()
    _set_empty_async_model_list(mock_client)

    with patch('genkit_google_genai._google.genai.client.Client', return_value=mock_client):
        plugin = VertexAI(project='p', location='us-central1')

    assert await plugin.resolve(ActionKind.MODEL, 'deep-research-preview-04-2026') is None
    assert await plugin.resolve(ActionKind.BACKGROUND_MODEL, 'deep-research-preview-04-2026') is None
    assert await plugin.resolve(ActionKind.MODEL, 'antigravity-preview-05-2026') is None
    assert await plugin.resolve(ActionKind.MODEL, 'lyria-3-clip-preview') is None
    assert await plugin.resolve(ActionKind.MODEL, 'lyria-002') is None


@pytest.mark.asyncio
async def test_deep_research_file_search_and_mcp_dump_snake_case() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-tools', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('q')])],
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
        messages=[Message(role=Role.USER, content=[Part.from_text('q')])],
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
        'genkit_google_genai._models._interactions_lyria',
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
        await action.run(ModelRequest(messages=[Message(role=Role.SYSTEM, content=[Part.from_text('play jazz')])]))

    assert create_calls[0]['system_instruction'] == 'play jazz'
    assert create_calls[0]['input'] == []


@pytest.mark.asyncio
async def test_lyria_rejects_empty_messages_without_system() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={'id': 'ly-empty', 'status': 'completed', 'steps': []},
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='Missing input') as exc_info:
            await action.run(ModelRequest(messages=[]))
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert create_calls == []


@pytest.mark.asyncio
async def test_lyria_keeps_system_instruction_and_user_input() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={
            'id': 'ly-both',
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
                messages=[
                    Message(role=Role.SYSTEM, content=[Part.from_text('cinematic')]),
                    Message(role=Role.USER, content=[Part.from_text('short sting')]),
                ]
            )
        )

    assert create_calls[0]['system_instruction'] == 'cinematic'
    assert create_calls[0]['input'] == [
        {'type': 'user_input', 'content': [{'type': 'text', 'text': 'short sting'}]},
    ]


@pytest.mark.asyncio
async def test_antigravity_rejects_config_api_key() -> None:
    """`config={'api_key': ...}` on Antigravity is an unknown key and sends nothing."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
        create_result={'id': 'ag-key', 'status': 'completed', 'steps': []},
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='api_key'):
            await action.run(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
                    config={'api_key': 'nope'},
                )
            )
    assert create_calls == []


@pytest.mark.asyncio
async def test_lyria_rejects_config_api_key() -> None:
    """`config={'api_key': ...}` on Interactions Lyria is an unknown key and sends nothing."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={'id': 'ly-key', 'status': 'completed', 'steps': []},
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='key',
        client_options=ClientOptions(),
    )
    with patcher:
        with pytest.raises(GenkitError, match='api_key'):
            await action.run(
                ModelRequest(
                    messages=[Message(role=Role.USER, content=[Part.from_text('riff')])],
                    config={'api_key': 'nope'},
                )
            )
    assert create_calls == []


@pytest.mark.asyncio
async def test_antigravity_generate_uses_context_secrets() -> None:
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
        create_result={
            'id': 'ag-secret',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'ok'}]}],
        },
        captured=captured,
    )
    action = create_antigravity_action(
        'antigravity-preview-05-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(
            ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('plan')])]),
            context={'secrets': {'api_key': 'tenant-key'}},
        )

    assert captured['api_key'] == 'tenant-key'
    assert create_calls[0]['agent'] == 'antigravity-preview-05-2026'


@pytest.mark.asyncio
async def test_lyria_generate_uses_context_secrets() -> None:
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={'id': 'ly-secret', 'status': 'completed', 'steps': []},
        captured=captured,
    )
    action = create_lyria_action(
        'lyria-3-clip-preview',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(
            ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('sting')])]),
            context={'secrets': {'api_key': 'tenant-key'}},
        )

    assert captured['api_key'] == 'tenant-key'
    assert create_calls[0]['model'] == 'lyria-3-clip-preview'


@pytest.mark.asyncio
async def test_lyria_002_hits_interactions_wire() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={'id': 'ly-002', 'status': 'completed', 'steps': []},
    )
    action = create_lyria_action(
        'lyria-002',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        await action.run(ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('sting')])]))

    assert create_calls[0]['model'] == 'lyria-002'


@patch('genkit_google_genai._google.genai.client.Client')
@patch('genkit_google_genai._google._list_genai_models')
@pytest.mark.asyncio
async def test_ai_generate_operation_deep_research_uses_context_secrets(
    mock_list_models: MagicMock, mock_client: MagicMock
) -> None:
    mock_list_models.return_value = GenaiModels()
    captured: dict[str, Any] = {}
    patcher, create_calls, get_calls, cancel_calls = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-ai-1', 'status': 'in_progress'},
        get_result={
            'id': 'dr-ai-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'report'}]}],
        },
        cancel_result={'id': 'dr-ai-1', 'status': 'cancelled'},
        captured=captured,
    )
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    tenant = {'secrets': {'api_key': 'tenant-key'}}
    with patcher:
        operation = await ai.generate_operation(
            model=GoogleAI.deep_research_model('deep-research-preview-04-2026'),
            prompt='Summarize recent advances in quantum error correction.',
            config={
                'base_url': 'https://start.example',
                'api_version': 'v1',
                'timeout': 1500,
                'custom_headers': {'x-request-id': 'start'},
            },
            context=tenant,
        )
        assert captured['api_key'] == 'tenant-key'
        assert captured['client_options'] == ClientOptions(
            base_url='https://start.example',
            api_version='v1',
            timeout=1500,
            custom_headers={'x-request-id': 'start'},
        )
        assert create_calls[0]['background'] is True
        assert operation.id == 'dr-ai-1'
        assert operation.done is False
        assert not operation.metadata

        updated = await ai.check_operation(
            operation,
            context=tenant,
            config={
                'base_url': 'https://poll.example',
                'api_version': 'v1beta',
                'timeout': 2000,
                'custom_headers': {'x-request-id': 'check'},
            },
        )
        assert captured['client_options'] == ClientOptions(
            base_url='https://poll.example',
            api_version='v1beta',
            timeout=2000,
            custom_headers={'x-request-id': 'check'},
        )

        cancelled = await ai.cancel_operation(
            operation,
            context=tenant,
            config={
                'base_url': 'https://cancel.example',
                'api_version': 'v1alpha',
                'timeout': 2500,
                'custom_headers': {'x-request-id': 'cancel'},
            },
        )

    assert captured['api_key'] == 'tenant-key'
    assert captured['client_options'] == ClientOptions(
        base_url='https://cancel.example',
        api_version='v1alpha',
        timeout=2500,
        custom_headers={'x-request-id': 'cancel'},
    )
    assert get_calls == ['dr-ai-1']
    assert cancel_calls == ['dr-ai-1']
    assert updated.done is True
    assert cancelled.done is True
    assert not updated.metadata
    assert not cancelled.metadata
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].text == 'report'


@patch('genkit_google_genai._google.genai.client.Client')
@patch('genkit_google_genai._google._list_genai_models')
@pytest.mark.asyncio
async def test_ai_generate_antigravity_uses_context_secrets(
    mock_list_models: MagicMock, mock_client: MagicMock
) -> None:
    mock_list_models.return_value = GenaiModels()
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity',
        create_result={
            'id': 'ag-ai-1',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'hello'}]}],
        },
        captured=captured,
    )
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with patcher:
        response = await ai.generate(
            model=GoogleAI.antigravity_model('antigravity-preview-05-2026'),
            prompt='Plan a three-day itinerary',
            config={
                'base_url': 'https://antigravity.example',
                'api_version': 'v1',
                'timeout': 1500,
                'custom_headers': {'x-request-id': 'antigravity'},
            },
            context={'secrets': {'api_key': 'tenant-key'}},
        )

    assert captured['api_key'] == 'tenant-key'
    assert captured['client_options'] == ClientOptions(
        base_url='https://antigravity.example',
        api_version='v1',
        timeout=1500,
        custom_headers={'x-request-id': 'antigravity'},
    )
    assert create_calls[0]['agent'] == 'antigravity-preview-05-2026'
    assert 'background' not in create_calls[0]
    assert response.text == 'hello'


@patch('genkit_google_genai._google.genai.client.Client')
@patch('genkit_google_genai._google._list_genai_models')
@pytest.mark.asyncio
async def test_ai_generate_lyria_002_hits_interactions_with_tenant_key(
    mock_list_models: MagicMock, mock_client: MagicMock
) -> None:
    mock_list_models.return_value = GenaiModels()
    captured: dict[str, Any] = {}
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria',
        create_result={'id': 'ly-ai-002', 'status': 'completed', 'steps': []},
        captured=captured,
    )
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with patcher:
        await ai.generate(
            model=GoogleAI.lyria_model('lyria-002'),
            prompt='a short cinematic sting',
            config={
                'base_url': 'https://lyria.example',
                'api_version': 'v1beta',
                'timeout': 2000,
                'custom_headers': {'x-request-id': 'lyria'},
            },
            context={'secrets': {'api_key': 'tenant-key'}},
        )

    assert captured['api_key'] == 'tenant-key'
    assert captured['client_options'] == ClientOptions(
        base_url='https://lyria.example',
        api_version='v1beta',
        timeout=2000,
        custom_headers={'x-request-id': 'lyria'},
    )
    assert create_calls[0]['model'] == 'lyria-002'


@patch('genkit_google_genai._google.genai.client.Client')
@patch('genkit_google_genai._google._list_genai_models')
@pytest.mark.asyncio
async def test_generate_operation_deep_research_handle_has_one_prefix(
    mock_list_models: MagicMock, mock_client: MagicMock
) -> None:
    """A Deep Research job's handle is `/background-model/googleai/deep-research-…` after start, check, and cancel."""
    mock_list_models.return_value = GenaiModels()
    patcher, _, get_calls, cancel_calls = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-1', 'status': 'in_progress'},
        get_result={'id': 'dr-1', 'status': 'in_progress'},
        cancel_result={'id': 'dr-1', 'status': 'cancelled'},
    )
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    key = '/background-model/googleai/deep-research-preview-04-2026'
    with patcher:
        operation = await ai.generate_operation(model='googleai/deep-research-preview-04-2026', prompt='research')
        checked = await ai.check_operation(operation)
        cancelled = await ai.cancel_operation(operation)

    assert operation.action == key
    assert checked.action == key
    assert cancelled.action == key
    assert get_calls == ['dr-1']
    assert cancel_calls == ['dr-1']


@patch('genkit_google_genai._google.genai.client.Client')
@patch('genkit_google_genai._google._list_genai_models')
@pytest.mark.asyncio
async def test_check_operation_saved_deep_research_handle_checks_on_a_fresh_app(
    mock_list_models: MagicMock, mock_client: MagicMock
) -> None:
    """A saved Deep Research handle checks by its stored `operation.action` on a new `Genkit`."""
    mock_list_models.return_value = GenaiModels()
    patcher, _, get_calls, _ = patch_interactions(
        'genkit_google_genai._models._deep_research',
        get_result={'id': 'dr-saved', 'status': 'completed', 'steps': []},
    )
    saved = Operation(id='dr-saved', done=False, action='/background-model/googleai/deep-research-preview-04-2026')
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with patcher:
        checked = await ai.check_operation(saved)

    assert get_calls == ['dr-saved']
    assert checked.done is True
    assert checked.action == '/background-model/googleai/deep-research-preview-04-2026'


# What an Antigravity, Interactions Lyria, or Deep Research config does at the
# call: another plugin's class or an unknown key fails before anything is sent,
# and `extra` is merged into the create body.

_ANTIGRAVITY = 'googleai/antigravity-preview-05-2026'
_LYRIA = 'googleai/lyria-3-clip-preview'
_DEEP_RESEARCH = 'googleai/deep-research-preview-04-2026'
_DEEP_RESEARCH_KEY = f'/background-model/{_DEEP_RESEARCH}'
_TEXT_REPLY = {
    'id': 'ix-ok',
    'status': 'completed',
    'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'ok'}]}],
}


class OtherPluginConfig(BaseModel):
    """Stands in for another plugin's config class (an OpenAI or Anthropic config, say)."""

    temperature: float | None = None


def _config_form(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """The config form the Dev UI renders for a model: `metadata.model.customOptions`."""
    model_meta = cast(dict[str, Any], (metadata or {})['model'])
    return cast(dict[str, Any], model_meta['customOptions'])


@pytest.fixture
def googleai() -> Iterator[Genkit]:
    """A Genkit app with the Google AI plugin and no model listing over the network."""
    with (
        patch('genkit_google_genai._google.genai.client.Client'),
        patch('genkit_google_genai._google._list_genai_models', return_value=GenaiModels()),
    ):
        yield Genkit(plugins=[GoogleAI(api_key='plugin-key')])


@pytest.mark.asyncio
async def test_generate_antigravity_with_other_config_class_raises_invalid_argument(googleai: Genkit) -> None:
    """`config=OtherPluginConfig()` on Antigravity raises INVALID_ARGUMENT naming `AntigravityConfig`."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._antigravity')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate(model=_ANTIGRAVITY, prompt='hi', config=OtherPluginConfig(temperature=0.2))

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'config must be genkit_google_genai.AntigravityConfig or a mapping' in str(raised.value)
    assert 'OtherPluginConfig' in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_lyria_with_other_config_class_raises_invalid_argument(googleai: Genkit) -> None:
    """`config=OtherPluginConfig()` on Interactions Lyria raises INVALID_ARGUMENT naming `LyriaConfig`."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._interactions_lyria')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate(model=_LYRIA, prompt='riff', config=OtherPluginConfig(temperature=0.2))

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'config must be genkit_google_genai.LyriaConfig or a mapping' in str(raised.value)
    assert 'OtherPluginConfig' in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_operation_deep_research_with_other_config_class_raises_invalid_argument(
    googleai: Genkit,
) -> None:
    """`generate_operation` on Deep Research with `OtherPluginConfig()` raises and starts no job."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._deep_research')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate_operation(model=_DEEP_RESEARCH, prompt='q', config=OtherPluginConfig(temperature=0.2))

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'config must be genkit_google_genai.DeepResearchConfig or a mapping' in str(raised.value)
    assert 'OtherPluginConfig' in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_antigravity_with_its_own_config_class_runs(googleai: Genkit) -> None:
    """`config=AntigravityConfig(store=False)` still runs and sends `store` (control)."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        response = await googleai.generate(model=_ANTIGRAVITY, prompt='hi', config=AntigravityConfig(store=False))

    assert response.text == 'ok'
    assert create_calls[0]['store'] is False


@pytest.mark.asyncio
async def test_generate_antigravity_with_dict_config_runs(googleai: Genkit) -> None:
    """`config={'store': False}` still runs and sends `store` (control)."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        response = await googleai.generate(model=_ANTIGRAVITY, prompt='hi', config={'store': False})

    assert response.text == 'ok'
    assert create_calls[0]['store'] is False


@pytest.mark.asyncio
async def test_check_operation_deep_research_after_start_still_resolves(googleai: Genkit) -> None:
    """A started Deep Research job checks to its report and cancels by its stored `operation.action`."""
    patcher, _, get_calls, cancel_calls = patch_interactions(
        'genkit_google_genai._models._deep_research',
        create_result={'id': 'dr-run', 'status': 'in_progress'},
        get_result={
            'id': 'dr-run',
            'status': 'completed',
            'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 'report'}]}],
        },
    )
    with patcher:
        operation = await googleai.generate_operation(
            model=_DEEP_RESEARCH, prompt='q', config={'thinking_summaries': 'auto'}
        )
        checked = await googleai.check_operation(operation)
        cancelled = await googleai.cancel_operation(operation)

    assert operation.action == _DEEP_RESEARCH_KEY
    assert checked.action == _DEEP_RESEARCH_KEY
    assert checked.done is True
    assert checked.output is not None
    assert checked.output.message is not None
    assert checked.output.message.content[0].text == 'report'
    assert cancelled.action == _DEEP_RESEARCH_KEY
    assert get_calls == ['dr-run']
    assert cancel_calls == ['dr-run']


@pytest.mark.asyncio
async def test_check_operation_deep_research_job_saved_before_upgrade_still_resolves(googleai: Genkit) -> None:
    """A handle saved as `/background-model/googleai/deep-research-…` checks and cancels."""
    patcher, _, get_calls, cancel_calls = patch_interactions(
        'genkit_google_genai._models._deep_research',
        get_result={'id': 'dr-old', 'status': 'in_progress'},
    )
    saved = Operation.model_validate({'id': 'dr-old', 'done': False, 'action': _DEEP_RESEARCH_KEY})
    with patcher:
        checked = await googleai.check_operation(saved)
        cancelled = await googleai.cancel_operation(saved)

    assert checked.action == _DEEP_RESEARCH_KEY
    assert checked.done is False
    assert cancelled.action == _DEEP_RESEARCH_KEY
    assert get_calls == ['dr-old']
    assert cancel_calls == ['dr-old']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('kind', 'name'),
    [
        (ActionKind.MODEL, _ANTIGRAVITY),
        (ActionKind.MODEL, _LYRIA),
        (ActionKind.BACKGROUND_MODEL, _DEEP_RESEARCH),
    ],
)
async def test_list_actions_interactions_models_advertise_same_config_schema(kind: ActionKind, name: str) -> None:
    """The Dev UI list and the resolved model show the same config form."""
    with patch('genkit_google_genai._google.genai.client.Client') as mock_client:
        _set_empty_async_model_list(mock_client.return_value)
        plugin = GoogleAI(api_key='plugin-key')
        listed = {meta.name: meta for meta in await plugin.list_actions()}
        resolved = await plugin.resolve(kind, name)

    assert resolved is not None
    assert _config_form(listed[name].metadata) == _config_form(resolved.metadata)


@pytest.mark.parametrize('config_class', [AntigravityConfig, LyriaConfig, DeepResearchConfig])
def test_interactions_configs_with_unknown_key_raise_validation_error(config_class: type[BaseModel]) -> None:
    """`AntigravityConfig(temprature=0.2)` and the Lyria and Deep Research equivalents fail naming `temprature`."""
    with pytest.raises(ValidationError, match='temprature'):
        config_class.model_validate({'temprature': 0.2})


@pytest.mark.asyncio
async def test_generate_antigravity_unknown_config_key_raises_and_sends_nothing(googleai: Genkit) -> None:
    """`config={'temprature': 0.2}` on Antigravity raises INVALID_ARGUMENT naming the key; nothing is sent."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._antigravity')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate(model=_ANTIGRAVITY, prompt='hi', config={'temprature': 0.2})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "unknown config key 'temprature'" in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_lyria_unknown_config_key_raises_and_sends_nothing(googleai: Genkit) -> None:
    """`config={'temperature': 0.4}` on Interactions Lyria raises INVALID_ARGUMENT naming the key; nothing is sent."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._interactions_lyria')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate(model=_LYRIA, prompt='riff', config={'temperature': 0.4})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "unknown config key 'temperature'" in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_operation_deep_research_unknown_config_key_raises_and_starts_no_job(googleai: Genkit) -> None:
    """`config={'thinking_summary': 'auto'}` on Deep Research raises naming the key and starts no job."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._deep_research')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate_operation(model=_DEEP_RESEARCH, prompt='q', config={'thinking_summary': 'auto'})

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert "unknown config key 'thinking_summary'" in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'path'),
    [
        ({'mcp_servers': [{'name': 'docs', 'url': 'https://mcp.example', 'alowed_tools': ['x']}]}, 'alowed_tools'),
        ({'file_search': {'file_search_store_names': ['stores/one'], 'top_k': 3}}, 'file_search.top_k'),
    ],
)
async def test_generate_operation_deep_research_unknown_nested_key_raises_and_starts_no_job(
    googleai: Genkit, config: dict[str, Any], path: str
) -> None:
    """A typo inside `mcp_servers` or `file_search` raises like a top-level one and starts no job."""
    patcher, create_calls, _, _ = patch_interactions('genkit_google_genai._models._deep_research')
    with patcher, pytest.raises(GenkitError) as raised:
        await googleai.generate_operation(model=_DEEP_RESEARCH, prompt='q', config=config)

    assert raised.value.status == 'INVALID_ARGUMENT'
    assert 'unknown config key' in str(raised.value)
    assert path in str(raised.value)
    assert create_calls == []


@pytest.mark.asyncio
async def test_generate_antigravity_extra_lands_in_create_body(googleai: Genkit) -> None:
    """`extra={'agent_config': {...}}` appears at the top level of the Antigravity create body."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        await googleai.generate(
            model=_ANTIGRAVITY, prompt='hi', config={'extra': {'agent_config': {'type': 'dynamic'}}}
        )

    body = create_calls[0]
    assert body['agent_config'] == {'type': 'dynamic'}
    assert body['agent'] == 'antigravity-preview-05-2026'
    assert body['environment'] == {'type': 'remote'}
    assert 'extra' not in body


@pytest.mark.asyncio
async def test_generate_lyria_extra_lands_in_create_body(googleai: Genkit) -> None:
    """`extra={'generation_config': {...}}` appears at the top level of the Lyria create body."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._interactions_lyria', create_result=_TEXT_REPLY
    )
    with patcher:
        await googleai.generate(
            model=_LYRIA, prompt='riff', config={'extra': {'generation_config': {'temperature': 0.4}}}
        )

    body = create_calls[0]
    assert body['generation_config'] == {'temperature': 0.4}
    assert body['model'] == 'lyria-3-clip-preview'
    assert body['response_modalities'] == ['audio', 'text']
    assert 'extra' not in body


@pytest.mark.asyncio
async def test_generate_operation_deep_research_extra_lands_in_create_body(googleai: Genkit) -> None:
    """`extra={'webhook_config': {...}}` appears at the top level of the Deep Research create body."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research', create_result={'id': 'dr-x', 'status': 'in_progress'}
    )
    with patcher:
        await googleai.generate_operation(
            model=_DEEP_RESEARCH, prompt='q', config={'extra': {'webhook_config': {'uri': 'https://hook.example'}}}
        )

    body = create_calls[0]
    assert body['webhook_config'] == {'uri': 'https://hook.example'}
    assert body['background'] is True
    assert body['agent_config'] == {'type': 'deep-research'}
    assert 'extra' not in body


@pytest.mark.asyncio
async def test_generate_operation_deep_research_extra_nested_key_wins_and_keeps_siblings(googleai: Genkit) -> None:
    """`extra` agent_config visualization `auto` over `visualization='off'` sends `auto` and keeps the rest."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._deep_research', create_result={'id': 'dr-x', 'status': 'in_progress'}
    )
    with patcher:
        await googleai.generate_operation(
            model=_DEEP_RESEARCH,
            prompt='q',
            config={
                'visualization': 'off',
                'thinking_summaries': 'auto',
                'extra': {'agent_config': {'visualization': 'auto'}},
            },
        )

    assert create_calls[0]['agent_config'] == {
        'type': 'deep-research',
        'thinking_summaries': 'auto',
        'visualization': 'auto',
    }


@pytest.mark.asyncio
async def test_generate_antigravity_extra_environment_overrides_default(googleai: Genkit) -> None:
    """`extra={'environment': {'type': 'custom', ...}}` wins over the default remote environment."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        await googleai.generate(
            model=_ANTIGRAVITY,
            prompt='hi',
            config={'extra': {'environment': {'type': 'custom', 'name': 'my-env'}}},
        )

    assert create_calls[0]['environment'] == {'type': 'custom', 'name': 'my-env'}


@pytest.mark.asyncio
async def test_generate_antigravity_extra_contents_are_not_checked(googleai: Genkit) -> None:
    """`extra={'temprature': 1}` is sent as written; what's inside `extra` is the caller's."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        await googleai.generate(model=_ANTIGRAVITY, prompt='hi', config={'extra': {'temprature': 1}})

    assert create_calls[0]['temprature'] == 1


@pytest.mark.asyncio
async def test_generate_antigravity_without_extra_sends_only_declared_settings(googleai: Genkit) -> None:
    """A config with no `extra` sends only the agent, input, environment, and declared settings (control)."""
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai._models._antigravity', create_result=_TEXT_REPLY
    )
    with patcher:
        await googleai.generate(
            model=_ANTIGRAVITY,
            prompt='hi',
            config={'response_modalities': ['TEXT'], 'timeout': 1500},
        )

    body = create_calls[0]
    assert set(body) == {'agent', 'input', 'environment', 'response_modalities'}
    assert body['response_modalities'] == ['text']


@pytest.mark.asyncio
@pytest.mark.parametrize('name', [_ANTIGRAVITY, _LYRIA, _DEEP_RESEARCH])
async def test_interactions_config_forms_advertise_no_additional_properties(name: str) -> None:
    """The Dev UI config form says `additionalProperties: false`, including Deep Research's MCP and file search."""
    with patch('genkit_google_genai._google.genai.client.Client') as mock_client:
        _set_empty_async_model_list(mock_client.return_value)
        listed = {meta.name: meta for meta in await GoogleAI(api_key='plugin-key').list_actions()}

    form = _config_form(listed[name].metadata)
    assert form['additionalProperties'] is False
    assert 'extra' in form['properties']
    for nested in (form.get('$defs') or {}).values():
        if nested.get('type') == 'object':
            assert nested['additionalProperties'] is False
