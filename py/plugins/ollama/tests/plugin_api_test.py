# Copyright 2025 Google LLC
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

"""Unit tests for Ollama Plugin."""

import unittest
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from genkit import ActionKind
from genkit.plugin_api import to_json_schema
from genkit.plugins.ollama import Ollama, ollama_name
from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.embedders import EmbeddingDefinition
from genkit.plugins.ollama.models import ModelDefinition, OllamaConfig, OllamaSupports


class TestOllamaInit(unittest.TestCase):
    """Test cases for Ollama.__init__ plugin."""

    def test_init_with_models(self) -> None:
        """Test correct propagation of models param."""
        model_ref = ModelDefinition(name='test_model')
        plugin = Ollama(models=[model_ref])

        assert plugin.models[0] == model_ref

    def test_init_with_embedders(self) -> None:
        """Test correct propagation of embedders param."""
        embedder_ref = EmbeddingDefinition(name='test_embedder')
        plugin = Ollama(embedders=[embedder_ref])

        assert plugin.embedders[0] == embedder_ref

    def test_init_with_options(self) -> None:
        """Test correct propagation of other options param."""
        model_ref = ModelDefinition(name='test_model')
        embedder_ref = EmbeddingDefinition(name='test_embedder')
        server_address = 'new.server.address'
        headers = {'Content-Type': 'json'}

        plugin = Ollama(
            models=[model_ref],
            embedders=[embedder_ref],
            server_address=server_address,
            request_headers=headers,
        )

        assert plugin.embedders[0] == embedder_ref
        assert plugin.models[0] == model_ref
        assert plugin.server_address == server_address
        assert plugin.request_headers == headers


@pytest.mark.asyncio
async def test_initialize(ollama_plugin_instance: Ollama) -> None:
    """Test init method of Ollama plugin."""
    model_ref = ModelDefinition(name='test_model')
    embedder_ref = EmbeddingDefinition(name='test_embedder')
    ollama_plugin_instance.models = [model_ref]
    ollama_plugin_instance.embedders = [embedder_ref]

    result = await ollama_plugin_instance.init()

    # init returns actions for pre-configured models and embedders
    assert len(result) == 2
    assert result[0].kind == ActionKind.MODEL
    assert result[1].kind == ActionKind.EMBEDDER


# _initialize_models and _initialize_embedders methods no longer exist in new plugin architecture
# Models and embedders are now created lazily via the resolve() method


@pytest.mark.parametrize(
    'kind, name',
    [
        (ActionKind.MODEL, 'test_model'),
        (ActionKind.EMBEDDER, 'test_embedder'),
    ],
)
@pytest.mark.asyncio
async def test_resolve_action(kind: ActionKind, name: str, ollama_plugin_instance: Ollama) -> None:
    """Unit Tests for resolve action method."""
    action = await ollama_plugin_instance.resolve(kind, ollama_name(name))

    assert action is not None
    assert action.kind == kind
    assert action.name == ollama_name(name)
    assert action.metadata is not None
    metadata = cast(dict[str, Any], action.metadata)

    if kind == ActionKind.MODEL:
        model_meta = cast(dict[str, Any], metadata['model'])
        assert model_meta['label'] == f'Ollama - {name}'
        supports = cast(dict[str, Any], model_meta['supports'])
        # Default model is CHAT → multiturn, and always advertises system role.
        assert supports['multiturn']
        assert supports['systemRole']
    else:
        embedder_meta = cast(dict[str, Any], metadata['embedder'])
        assert embedder_meta['label'] == f'Ollama Embedding - {name}'
        assert embedder_meta['supports'] == {'input': ['text']}


@pytest.mark.asyncio
async def test_create_model_action_chat_with_media() -> None:
    """A CHAT model with media support advertises multiturn/tools/media."""
    plugin = Ollama(
        models=[ModelDefinition(name='llava', api_type=OllamaAPITypes.CHAT, supports=OllamaSupports(media=True))]
    )
    action = plugin._create_model_action(ollama_name('llava'))

    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['multiturn'] is True
    assert supports['tools'] is True
    assert supports['media'] is True


@pytest.mark.asyncio
async def test_create_model_action_generate_gates_capabilities() -> None:
    """A GENERATE model reports multiturn/tools/media all False."""
    plugin = Ollama(models=[ModelDefinition(name='gen', api_type=OllamaAPITypes.GENERATE)])
    action = plugin._create_model_action(ollama_name('gen'))

    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['multiturn'] is False
    assert supports['tools'] is False
    assert supports['media'] is False
    assert supports['systemRole'] is True


@pytest.mark.asyncio
async def test_dynamic_model_advertises_generic_capabilities() -> None:
    """A dynamically-resolved model (not pre-configured) advertises the full
    generic capability set, matching the JS GENERIC_MODEL_INFO and the Go
    defaultOllamaSupports for un-probed models."""
    plugin = Ollama()
    action = plugin._create_model_action(ollama_name('some-unconfigured-model'))

    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['multiturn'] is True
    assert supports['tools'] is True
    assert supports['media'] is True
    assert supports['systemRole'] is True


@pytest.mark.asyncio
async def test_create_model_action_custom_options_is_ollama_config() -> None:
    """The model action advertises OllamaConfig (with Ollama-only knobs) as its schema."""
    plugin = Ollama(models=[ModelDefinition(name='m')])
    action = plugin._create_model_action(ollama_name('m'))

    model_meta = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model'])
    assert model_meta['customOptions'] == to_json_schema(OllamaConfig)
    props = cast(dict[str, Any], model_meta['customOptions']['properties'])
    assert 'think' in props
    assert 'keepAlive' in props


# _define_ollama_model and _define_ollama_embedder methods no longer exist in new plugin architecture
# Actions are now created via _create_model_action and _create_embedder_action methods


@pytest.mark.asyncio
async def test_list_actions(ollama_plugin_instance: Ollama) -> None:
    """Unit tests for list_actions method."""

    class MockModelResponse(BaseModel):
        model: str

    class MockListResponse(BaseModel):
        models: list[MockModelResponse]

    client_mock = MagicMock()
    list_method_mock = AsyncMock()
    client_mock.list = list_method_mock

    list_method_mock.return_value = MockListResponse(
        models=[
            MockModelResponse(model='test_model'),
            MockModelResponse(model='test_embed'),
        ]
    )

    def mock_client() -> MagicMock:
        return client_mock

    ollama_plugin_instance.client = mock_client

    actions = await ollama_plugin_instance.list_actions()

    assert len(actions) == 2

    has_model = False
    for action in actions:
        if hasattr(action, 'name') and 'test_model' in action.name:
            has_model = True
            break

    assert has_model

    has_embedder = False
    for action in actions:
        if hasattr(action, 'name') and 'test_embed' in action.name:
            has_embedder = True
            break

    assert has_embedder
