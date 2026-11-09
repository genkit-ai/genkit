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

"""GoogleAI key resolution: env chain, keyless boot, context.secrets BYOK.

Order per call: ``context.secrets.api_key``, then the plugin key (argument or
``GEMINI_API_KEY`` -> ``GOOGLE_API_KEY`` -> ``GOOGLE_GENAI_API_KEY``), else
``FAILED_PRECONDITION``.
"""

import os
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_google_genai import GoogleAI, VertexAI
from genkit_google_genai._google import GenaiModels, _call_client
from genkit_google_genai._models._interactions_utils import api_key_for_context
from genkit_google_genai._models._secrets import context_api_key
from google.genai import types as genai_types

from genkit import Genkit, GenkitError

_ENV_KEYS = ('GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_API_KEY')


@pytest.fixture
def no_key_env() -> Iterator[None]:
    """Strip every Gemini key variable for the test."""
    env = {k: v for k, v in os.environ.items() if k not in _ENV_KEYS}
    with patch.dict(os.environ, env, clear=True):
        yield


def _text_response(text: str = 'Smoked Salmon Tartine') -> genai_types.GenerateContentResponse:
    return genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(parts=[genai_types.Part(text=text)], role='model'),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ]
    )


async def _generate_error(ai: Genkit, **kwargs: Any) -> Any:
    """The call's error, raised or boxed on the response (PR0 flips generate to raise)."""
    try:
        response = await ai.generate(**kwargs)
    except GenkitError as e:
        return e
    assert response.error is not None
    return response.error


def _no_listing() -> Any:
    return patch('genkit_google_genai._google._list_genai_models', new_callable=AsyncMock, return_value=GenaiModels())


def _keyed_clients(response: genai_types.EmbedContentResponse) -> tuple[dict[str, MagicMock], Any]:
    """A Client factory that hands out one mock per api_key, so tests see which key embedded."""
    clients: dict[str, MagicMock] = {}

    def make(**kwargs: Any) -> MagicMock:
        client = MagicMock()
        client.aio.models.embed_content = AsyncMock(return_value=response)
        clients[kwargs['api_key']] = client
        return client

    return clients, make


def _embed_response() -> genai_types.EmbedContentResponse:
    return genai_types.EmbedContentResponse(embeddings=[genai_types.ContentEmbedding(values=[0.1, 0.2])])


# Env lookup


@pytest.mark.parametrize(
    ('env', 'expected'),
    [
        ({'GEMINI_API_KEY': 'gemini', 'GOOGLE_API_KEY': 'google', 'GOOGLE_GENAI_API_KEY': 'genai'}, 'gemini'),
        ({'GOOGLE_API_KEY': 'google', 'GOOGLE_GENAI_API_KEY': 'genai'}, 'google'),
        ({'GOOGLE_GENAI_API_KEY': 'genai'}, 'genai'),
        ({'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': 'google'}, 'google'),
    ],
    ids=['all-three', 'google-over-genai', 'genai-only', 'empty-gemini-skipped'],
)
def test_env_lookup_order(no_key_env: None, env: dict[str, str], expected: str) -> None:
    """GEMINI_API_KEY, then GOOGLE_API_KEY, then GOOGLE_GENAI_API_KEY, as in JS."""
    with patch.dict(os.environ, env):
        plugin = GoogleAI()
    assert plugin._plugin_api_key() == expected


def test_explicit_api_key_wins_over_env(no_key_env: None) -> None:
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'}):
        plugin = GoogleAI(api_key='arg-key')
    assert plugin._plugin_api_key() == 'arg-key'


@pytest.mark.asyncio
async def test_resolved_env_key_is_passed_to_client_explicitly(no_key_env: None) -> None:
    """The SDK's own env fallback prefers GOOGLE_API_KEY; Genkit passes its pick explicitly."""
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'gemini', 'GOOGLE_API_KEY': 'google'}):
        plugin = GoogleAI()
        with patch('genkit_google_genai._google.genai.client.Client') as client_cls:
            plugin._runtime_client()
    assert client_cls.call_args.kwargs['api_key'] == 'gemini'


# Keyless boot


@pytest.mark.asyncio
async def test_keyless_init_makes_no_client_or_network_call(no_key_env: None) -> None:
    """No key and no credentials: init() skips model listing and builds no client."""
    plugin = GoogleAI()
    with (
        patch('genkit_google_genai._google._list_genai_models', new_callable=AsyncMock) as list_models,
        patch('genkit_google_genai._google.genai.client.Client') as client_cls,
    ):
        actions = await plugin.init()
        listed = await plugin.list_actions()
    list_models.assert_not_awaited()
    client_cls.assert_not_called()
    # Interactions models are a static catalog and still register.
    assert any(a.name.startswith('googleai/deep-research') for a in actions)
    assert listed


@pytest.mark.asyncio
async def test_credentials_without_key_still_lists_models(no_key_env: None) -> None:
    plugin = GoogleAI(credentials=MagicMock())
    with (
        patch(
            'genkit_google_genai._google._list_genai_models', new_callable=AsyncMock, return_value=GenaiModels()
        ) as list_models,
        patch('genkit_google_genai._google.genai.client.Client'),
    ):
        await plugin.init()
    list_models.assert_awaited_once()


@pytest.mark.asyncio
async def test_keyed_init_still_lists_models(no_key_env: None) -> None:
    plugin = GoogleAI(api_key='plugin-key')
    with (
        patch(
            'genkit_google_genai._google._list_genai_models', new_callable=AsyncMock, return_value=GenaiModels()
        ) as list_models,
        patch('genkit_google_genai._google.genai.client.Client'),
    ):
        await plugin.init()
    list_models.assert_awaited_once()


# Per-call resolution through ai.generate


@pytest.mark.asyncio
async def test_keyless_plugin_serves_byok_generate(no_key_env: None) -> None:
    """A keyless gateway serves a call that brings context.secrets.api_key."""
    ai = Genkit(plugins=[GoogleAI()])
    with (
        patch('genkit_google_genai._google.genai.client.Client') as base_cls,
        patch('genkit_google_genai._models._gemini.genai.Client') as request_cls,
    ):
        base_cls.return_value.vertexai = False
        request_cls.return_value.aio.models.generate_content = AsyncMock(return_value=_text_response())
        response = await ai.generate(
            model='googleai/gemini-2.5-flash',
            prompt='Suggest a dish.',
            context={'secrets': {'api_key': 'tenant-key'}},
        )
    assert response.text == 'Smoked Salmon Tartine'
    # Every client built for this call carries the tenant key, never None.
    for call in [*base_cls.call_args_list, *request_cls.call_args_list]:
        assert call.kwargs['api_key'] == 'tenant-key'
    request_cls.return_value.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_secret_beats_plugin_key_on_generate(no_key_env: None) -> None:
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with (
        _no_listing(),
        patch('genkit_google_genai._google.genai.client.Client') as base_cls,
        patch('genkit_google_genai._models._gemini.genai.Client') as request_cls,
    ):
        base_cls.return_value.vertexai = False
        request_cls.return_value.aio.models.generate_content = AsyncMock(return_value=_text_response())
        await ai.generate(
            model='googleai/gemini-2.5-flash',
            prompt='Suggest a dish.',
            context={'secrets': {'api_key': 'tenant-key'}},
        )
    assert request_cls.call_args.kwargs['api_key'] == 'tenant-key'
    base_cls.return_value.aio.models.generate_content.assert_not_called()


@pytest.mark.asyncio
async def test_app_secrets_without_api_key_fall_back_to_plugin_key(no_key_env: None) -> None:
    """secrets holding only app secrets (db password, CRM token) bill the plugin key."""
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with (
        _no_listing(),
        patch('genkit_google_genai._google.genai.client.Client') as base_cls,
        patch('genkit_google_genai._models._gemini.genai.Client') as request_cls,
    ):
        base_cls.return_value.vertexai = False
        base_cls.return_value.aio.models.generate_content = AsyncMock(return_value=_text_response())
        response = await ai.generate(
            model='googleai/gemini-2.5-flash',
            prompt='Suggest a dish.',
            context={'secrets': {'crm_token': 'crm-123', 'db_password': 'hunter2'}},
        )
    assert response.text == 'Smoked Salmon Tartine'
    assert base_cls.call_args.kwargs['api_key'] == 'plugin-key'
    request_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'context',
    [None, {'secrets': {'crm_token': 'crm-123'}}, {'secrets': {'api_key': ''}}],
    ids=['no-context', 'app-secrets-only', 'empty-key'],
)
async def test_keyless_call_without_secret_is_failed_precondition(
    no_key_env: None, context: dict[str, object] | None
) -> None:
    """No plugin key and no secret: FAILED_PRECONDITION with the setup help, not a raw ValueError."""
    ai = Genkit(plugins=[GoogleAI()])
    with patch('genkit_google_genai._google.genai.client.Client') as client_cls:
        error = await _generate_error(ai, model='googleai/gemini-2.5-flash', prompt='Suggest a dish.', context=context)
    assert error.status == 'FAILED_PRECONDITION'
    message = error.message
    assert 'GEMINI_API_KEY' in message
    assert 'GOOGLE_API_KEY' in message
    assert 'https://aistudio.google.com/app/apikey' in message
    assert "context={'secrets': {'api_key': ...}}" in message
    client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_misplaced_key_still_invalid_argument_on_keyless_plugin(no_key_env: None) -> None:
    """A key at the top level of context is a caller bug, not a missing key."""
    ai = Genkit(plugins=[GoogleAI()])
    error = await _generate_error(ai, model='googleai/gemini-2.5-flash', prompt='hi', context={'api_key': 'tenant-key'})
    assert error.status == 'INVALID_ARGUMENT'


def test_interactions_missing_key_uses_same_status_and_text(no_key_env: None) -> None:
    """Deep Research / Antigravity / Lyria and Gemini report a missing key the same way."""
    with pytest.raises(GenkitError) as interactions:
        api_key_for_context({}, None)
    with pytest.raises(GenkitError) as gemini:
        _call_client(GoogleAI(), {})
    assert interactions.value.status == gemini.value.status == 'FAILED_PRECONDITION'
    assert str(interactions.value) == str(gemini.value)


def test_context_api_key_returns_none_without_api_key() -> None:
    assert context_api_key({'secrets': {'crm_token': 'crm-123'}}) is None
    assert context_api_key({'secrets': {}}) is None
    assert context_api_key({'secrets': {'apiKey': 'tenant-key'}}) == 'tenant-key'


# Veo


@pytest.mark.asyncio
async def test_keyless_veo_start_uses_secret(no_key_env: None) -> None:
    sdk_op = MagicMock()
    sdk_op.name = 'operations/1'
    sdk_op.done = False
    sdk_op.error = None
    sdk_op.response = None
    ai = Genkit(plugins=[GoogleAI()])
    with (
        patch('genkit_google_genai._google.genai.client.Client') as base_cls,
        patch('genkit_google_genai._models._veo.genai.Client') as request_cls,
    ):
        base_cls.return_value.vertexai = False
        request_cls.return_value.aio.models.generate_videos = AsyncMock(return_value=sdk_op)
        operation = await ai.generate_operation(
            model='googleai/veo-3.1-generate-preview',
            prompt='a chef plating a dish',
            context={'secrets': {'api_key': 'tenant-key'}},
        )
    assert operation.id == 'operations/1'
    for call in [*base_cls.call_args_list, *request_cls.call_args_list]:
        assert call.kwargs['api_key'] == 'tenant-key'


@pytest.mark.asyncio
async def test_keyless_veo_without_secret_is_failed_precondition(no_key_env: None) -> None:
    ai = Genkit(plugins=[GoogleAI()])
    with pytest.raises(GenkitError) as raised:
        await ai.generate_operation(model='googleai/veo-3.1-generate-preview', prompt='a chef plating a dish')
    assert raised.value.status == 'FAILED_PRECONDITION'


# Embedders (PR4-06)


@pytest.mark.asyncio
async def test_embed_secret_bills_tenant_not_plugin(no_key_env: None) -> None:
    """ai.embed(context=...) runs on the tenant key even when the plugin has one."""
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    clients, make = _keyed_clients(_embed_response())
    with _no_listing(), patch('genkit_google_genai._google.genai.client.Client', side_effect=make):
        embeddings = await ai.embed(
            embedder='googleai/gemini-embedding-001',
            content='Peanut allergy: no satay.',
            context={'secrets': {'api_key': 'tenant-key'}},
        )
    assert embeddings[0].embedding == [0.1, 0.2]
    clients['tenant-key'].aio.models.embed_content.assert_awaited_once()
    if 'plugin-key' in clients:  # built by init() for model listing only
        clients['plugin-key'].aio.models.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_embed_many_secret_on_keyless_plugin(no_key_env: None) -> None:
    ai = Genkit(plugins=[GoogleAI()])
    with patch('genkit_google_genai._google.genai.client.Client') as client_cls:
        client_cls.return_value.aio.models.embed_content = AsyncMock(
            return_value=genai_types.EmbedContentResponse(
                embeddings=[genai_types.ContentEmbedding(values=[0.1]), genai_types.ContentEmbedding(values=[0.2])]
            )
        )
        embeddings = await ai.embed_many(
            embedder='googleai/gemini-embedding-001',
            content=['Smoked Salmon Tartine', 'Mushroom Risotto'],
            context={'secrets': {'api_key': 'tenant-key'}},
        )
    assert len(embeddings) == 2
    assert [c.kwargs['api_key'] for c in client_cls.call_args_list] == ['tenant-key']


@pytest.mark.asyncio
async def test_embed_without_secret_uses_plugin_key(no_key_env: None) -> None:
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])
    with _no_listing(), patch('genkit_google_genai._google.genai.client.Client') as client_cls:
        client_cls.return_value.aio.models.embed_content = AsyncMock(return_value=_embed_response())
        await ai.embed(
            embedder='googleai/gemini-embedding-001',
            content='Peanut allergy: no satay.',
            context={'secrets': {'crm_token': 'crm-123'}},
        )
    assert [c.kwargs['api_key'] for c in client_cls.call_args_list] == ['plugin-key']


@pytest.mark.asyncio
async def test_keyless_embed_without_secret_is_failed_precondition(no_key_env: None) -> None:
    ai = Genkit(plugins=[GoogleAI()])
    with pytest.raises(GenkitError) as raised:
        await ai.embed(embedder='googleai/gemini-embedding-001', content='Peanut allergy: no satay.')
    assert raised.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_embed_inside_flow_inherits_flow_context(no_key_env: None) -> None:
    """embed() without context= sees the enclosing flow's context, like generate()."""
    ai = Genkit(plugins=[GoogleAI(api_key='plugin-key')])

    @ai.flow()
    async def index_menu(text: str) -> int:
        return len(await ai.embed(embedder='googleai/gemini-embedding-001', content=text))

    clients, make = _keyed_clients(_embed_response())
    with _no_listing(), patch('genkit_google_genai._google.genai.client.Client', side_effect=make):
        await index_menu.run('Mushroom Risotto', context={'secrets': {'api_key': 'tenant-key'}})
    clients['tenant-key'].aio.models.embed_content.assert_awaited_once()
    if 'plugin-key' in clients:
        clients['plugin-key'].aio.models.embed_content.assert_not_called()


@pytest.mark.asyncio
async def test_vertex_embed_secret_uses_express_client() -> None:
    """VertexAI embedders honor the same secret, dropping project and location (express mode)."""
    plugin = VertexAI(project='plugin-project', location='us-central1')
    with patch('genkit_google_genai._google.genai.client.Client') as client_cls:
        _call_client(plugin, {'secrets': {'api_key': 'tenant-key'}}, prefer_secret=True)
    kwargs = client_cls.call_args.kwargs
    assert kwargs['api_key'] == 'tenant-key'
    assert kwargs['credentials'] is None
    assert 'project' not in kwargs
    assert 'location' not in kwargs
