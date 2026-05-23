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
# pyright: reportMissingImports=false

"""Unit tests for Ollama Plugin."""

import unittest
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from genkit import ActionKind
from genkit.plugins.ollama import Ollama, ollama_name
from genkit.plugins.ollama.constants import OllamaAPITypes
from genkit.plugins.ollama.embedders import EmbeddingDefinition
from genkit.plugins.ollama.models import ModelDefinition, OllamaSupports


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
async def test_init_passes_request_headers_to_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test request headers are passed to the Ollama client."""
    client_kwargs: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)

    headers = {'Authorization': 'Bearer token'}
    server_address = 'http://ollama.example.test'
    monkeypatch.setattr('genkit.plugins.ollama.plugin_api.ollama_api.AsyncClient', FakeAsyncClient)

    plugin = Ollama(server_address=server_address, request_headers=headers)
    plugin.client()

    assert client_kwargs == {'host': server_address, 'headers': headers}


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
        supports = cast(dict[str, Any], model_meta['supports'])
        assert model_meta['label'] == f'Ollama - {name}'
        assert supports['multiturn']
        assert supports['systemRole']
        assert supports['tools']
        assert supports['media'] is False
        assert supports['output'] == ['text', 'json']
        assert supports['constrained'] == 'all'
    else:
        embedder_meta = cast(dict[str, Any], metadata['embedder'])
        assert embedder_meta['label'] == f'Ollama Embedding - {name}'
        assert embedder_meta['supports'] == {'input': ['text']}


def test_generate_model_metadata_disables_chat_only_features() -> None:
    """Generate API models should not advertise chat-only capabilities."""
    plugin = Ollama(models=[ModelDefinition(name='test_model', api_type=OllamaAPITypes.GENERATE)])

    action = plugin._create_model_action(ollama_name('test_model'))  # noqa: SLF001
    metadata = cast(dict[str, Any], action.metadata)
    model_meta = cast(dict[str, Any], metadata['model'])
    supports = cast(dict[str, Any], model_meta['supports'])

    assert supports['multiturn'] is False
    assert supports['tools'] is False
    assert supports['media'] is False


def test_model_metadata_uses_declared_media_support() -> None:
    """Models should advertise media only when explicitly configured."""
    plugin = Ollama(models=[ModelDefinition(name='llava', supports=OllamaSupports(media=True))])

    action = plugin._create_model_action(ollama_name('llava'))  # noqa: SLF001
    metadata = cast(dict[str, Any], action.metadata)
    model_meta = cast(dict[str, Any], metadata['model'])
    supports = cast(dict[str, Any], model_meta['supports'])

    assert supports['media'] is True


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
            assert action.metadata is not None
            model_meta = cast(dict[str, Any], action.metadata['model'])
            supports = cast(dict[str, Any], model_meta['supports'])
            assert supports['tools']
            assert supports['media'] is False
            break

    assert has_model

    has_embedder = False
    for action in actions:
        if hasattr(action, 'name') and 'test_embed' in action.name:
            has_embedder = True
            break

    assert has_embedder


@pytest.mark.asyncio
async def test_async_request_headers_callback_resolved_on_init(monkeypatch: pytest.MonkeyPatch) -> None:
    """Async callable headers should be awaited during init()."""
    client_kwargs: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)

    async def headers_provider() -> dict[str, str]:
        return {'Authorization': 'Bearer async-token'}

    monkeypatch.setattr('genkit.plugins.ollama.plugin_api.ollama_api.AsyncClient', FakeAsyncClient)
    plugin = Ollama(request_headers=headers_provider)

    # Before init: empty (callable hasn't been resolved yet).
    assert plugin.request_headers == {}
    await plugin.init()
    assert plugin.request_headers == {'Authorization': 'Bearer async-token'}

    plugin.client()
    assert client_kwargs['headers'] == {'Authorization': 'Bearer async-token'}


@pytest.mark.asyncio
async def test_sync_request_headers_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sync callable headers should also be resolved during init()."""
    monkeypatch.setattr(
        'genkit.plugins.ollama.plugin_api.ollama_api.AsyncClient', MagicMock(spec=type)
    )
    plugin = Ollama(request_headers=lambda: {'X-Tenant': 'acme'})
    await plugin.init()
    assert plugin.request_headers == {'X-Tenant': 'acme'}


@pytest.mark.asyncio
async def test_timeout_propagated_to_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Custom timeout is forwarded to the underlying ollama AsyncClient."""
    client_kwargs: dict[str, Any] = {}

    class FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)

    monkeypatch.setattr('genkit.plugins.ollama.plugin_api.ollama_api.AsyncClient', FakeAsyncClient)
    plugin = Ollama(timeout=42.5)
    plugin.client()
    assert client_kwargs['timeout'] == 42.5


@pytest.mark.asyncio
async def test_connection_error_wrapped_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reaching the plugin when ollama isn't running surfaces a friendly error."""
    import httpx

    from genkit.plugins.ollama import OllamaConnectionError
    from genkit.plugins.ollama.models import OllamaModel

    monkeypatch.setattr('genkit.plugins.ollama.plugin_api.ollama_api.AsyncClient', MagicMock(spec=type))
    plugin = Ollama(models=[ModelDefinition(name='test_model')], server_address='http://unreachable.test')
    action = plugin._create_model_action(ollama_name('test_model'))  # noqa: SLF001

    async def boom(self: OllamaModel, request: Any, ctx: Any = None) -> None:
        raise httpx.ConnectError('boom')

    monkeypatch.setattr(OllamaModel, 'generate', boom)

    with pytest.raises(OllamaConnectionError) as excinfo:
        await action._fn(None)  # noqa: SLF001
    assert 'http://unreachable.test' in str(excinfo.value)
    assert 'ollama serve' in str(excinfo.value)


def test_model_metadata_uses_ollama_config_schema() -> None:
    """Generated model action exposes OllamaConfig (with think/keep_alive) in customOptions."""
    from genkit.plugins.ollama.models import OllamaConfig

    plugin = Ollama(models=[ModelDefinition(name='test_model')])
    action = plugin._create_model_action(ollama_name('test_model'))  # noqa: SLF001
    custom_options = cast(dict[str, Any], action.metadata['model']['customOptions'])  # type: ignore[index]
    properties = set(cast(dict[str, Any], custom_options.get('properties', {})).keys())
    # OllamaConfig adds these on top of ModelConfig.
    assert {'think', 'keepAlive', 'numCtx', 'minP', 'seed', 'numPredict'} <= properties
    # ModelConfig fields still present.
    assert 'temperature' in properties
    assert OllamaConfig is not None  # silence flake
