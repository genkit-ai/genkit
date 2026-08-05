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
from unittest.mock import AsyncMock, patch

import pytest
from genkit_google_genai._interactions.converters import split_system_instruction
from genkit_google_genai._interactions.options import ClientOptions
from genkit_google_genai.models.antigravity import AntigravityConfig, create_antigravity_action
from genkit_google_genai.models.deep_research import (
    create_deep_research_background_action,
    deep_research_model,
)
from genkit_google_genai.models.lyria import LyriaConfig, create_lyria_action
from google.genai.interactions import Interaction

from genkit import GenkitError, Message, ModelRequest, Part, Role, TextPart
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
    assert (operation.metadata or {}).get('clientOptions', {}).get('apiKey') == 'plugin-key'


@pytest.mark.asyncio
async def test_deep_research_check_uses_stored_api_key() -> None:
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
        metadata={'clientOptions': {'baseUrl': 'https://example.test', 'apiKey': 'override-key'}},
    )
    with patcher:
        updated = await action.check(operation)

    assert captured['api_key'] == 'override-key'
    assert get_calls == ['dr-1']
    assert updated.done is True
    assert updated.output is not None
    assert updated.output.message is not None
    assert updated.output.message.content[0].root.text == 'done'


@pytest.mark.asyncio
async def test_deep_research_cancel_uses_stored_api_key() -> None:
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
        metadata={'clientOptions': {'baseUrl': 'https://example.test', 'apiKey': 'override-key'}},
    )
    with patcher:
        updated = await action.cancel(operation)

    assert captured['api_key'] == 'override-key'
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
async def test_deep_research_start_stores_request_api_key_override() -> None:
    """Per-request api_key wins and is what check will reuse from the Operation."""
    patcher, _, _, _ = patch_interactions(
        'genkit_google_genai.models.deep_research',
        create_result={'id': 'dr-key', 'status': 'in_progress'},
    )
    action = create_deep_research_background_action(
        'deep-research-preview-04-2026',
        plugin_api_key='plugin-key',
        client_options=ClientOptions(),
    )
    with patcher:
        operation = await action.start(
            ModelRequest(
                messages=[Message(role=Role.USER, content=[Part(TextPart(text='q'))])],
                config={'api_key': 'request-key'},
            )
        )

    assert (operation.metadata or {}).get('clientOptions', {}).get('apiKey') == 'request-key'


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
        config=LyriaConfig(api_key='k', response_modalities=['audio']),
    )
    assert isinstance(request.config, LyriaConfig)
    assert request.config.api_key == 'k'


@pytest.mark.asyncio
async def test_lyria_defaults_audio_and_text_modalities() -> None:
    patcher, create_calls, _, _ = patch_interactions(
        'genkit_google_genai.models.lyria',
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
        'genkit_google_genai.models.lyria',
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
                config={'temperature': 0.4, 'api_key': 'should-not-passthrough'},
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


@pytest.mark.asyncio
async def test_deep_research_define_background_model_sets_action() -> None:
    """define_background_model must stamp Operation.action for check/cancel."""
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
