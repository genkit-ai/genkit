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
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import ollama as ollama_api
import pytest
from genkit_ollama import Ollama, OllamaConnectionError, RequestHeaderParams, ollama_name
from genkit_ollama._errors import wrap_connection_errors
from genkit_ollama.constants import OllamaAPITypes
from genkit_ollama.embedders import EmbeddingDefinition
from genkit_ollama.models import ModelDefinition, OllamaConfig, OllamaSupports
from pydantic import BaseModel

from genkit import (
    ActionKind,
    Document,
    EmbedRequest,
    Media,
    MediaPart,
    Message,
    ModelRequest,
    Part,
    Role,
    TextPart,
)
from genkit.plugin_api import to_json_schema


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


def test_timeout_stored() -> None:
    """A timeout kwarg is stored on the plugin."""
    plugin = Ollama(timeout=30.0)

    assert plugin.timeout == 30.0


def test_make_client_forwards_host_headers_and_timeout() -> None:
    """_make_client forwards host, headers, and a non-None timeout to AsyncClient."""
    plugin = Ollama(
        server_address='http://example:11434',
        request_headers={'Authorization': 'Bearer x'},
        timeout=30.0,
    )

    with patch('ollama.AsyncClient') as async_client:
        plugin._make_client()

    async_client.assert_called_once_with(
        host='http://example:11434',
        headers={'Authorization': 'Bearer x'},
        timeout=30.0,
    )


def test_make_client_omits_timeout_when_none() -> None:
    """With the default timeout (None) the timeout kwarg is omitted entirely."""
    plugin = Ollama(server_address='http://example:11434')

    with patch('ollama.AsyncClient') as async_client:
        plugin._make_client()

    _, kwargs = async_client.call_args
    assert 'timeout' not in kwargs
    assert kwargs == {'host': 'http://example:11434', 'headers': {}}


def test_make_client_propagates_static_headers() -> None:
    """A static-dict plugin propagates its headers through _make_client."""
    headers = {'X-Token': 'abc'}
    plugin = Ollama(request_headers=headers)

    with patch('ollama.AsyncClient') as async_client:
        plugin._make_client()

    _, kwargs = async_client.call_args
    assert kwargs['headers'] == headers


@pytest.mark.asyncio
async def test_sync_callable_headers_resolved_per_request() -> None:
    """A sync header callable is resolved on every request, not once at init()."""
    tokens = iter(['t1', 't2'])
    plugin = Ollama(request_headers=lambda params: {'Authorization': next(tokens)})

    # init() does not eagerly resolve a callable.
    assert await plugin.init() == []
    assert plugin.request_headers == {}

    client_mock = MagicMock()
    client_mock._client.aclose = AsyncMock()
    with patch('ollama.AsyncClient', return_value=client_mock) as async_client:
        async with plugin._client_for_request():
            pass
        async with plugin._client_for_request():
            pass

    assert async_client.call_args_list[0].kwargs['headers'] == {'Authorization': 't1'}
    assert async_client.call_args_list[1].kwargs['headers'] == {'Authorization': 't2'}
    # Each fresh per-request client's connection pool is closed on exit.
    assert client_mock._client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_async_callable_headers_resolved_per_request() -> None:
    """An async header callable is awaited on every request, not once at init()."""
    tokens = iter(['a1', 'a2'])

    async def headers(params: RequestHeaderParams) -> dict[str, str]:
        return {'Authorization': next(tokens)}

    plugin = Ollama(request_headers=headers)

    assert await plugin.init() == []
    assert plugin.request_headers == {}

    client_mock = MagicMock()
    client_mock._client.aclose = AsyncMock()
    with patch('ollama.AsyncClient', return_value=client_mock) as async_client:
        async with plugin._client_for_request():
            pass
        async with plugin._client_for_request():
            pass

    assert async_client.call_args_list[0].kwargs['headers'] == {'Authorization': 'a1'}
    assert async_client.call_args_list[1].kwargs['headers'] == {'Authorization': 'a2'}
    assert client_mock._client.aclose.await_count == 2


@pytest.mark.asyncio
async def test_model_action_passes_request_context_to_header_callable() -> None:
    """A model header callable receives the server address, model, and model request."""
    captured: dict[str, Any] = {}

    def make_headers(params: RequestHeaderParams) -> dict[str, str]:
        captured['params'] = params
        return {'Authorization': 'Bearer tok'}

    model_def = ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT)
    plugin = Ollama(models=[model_def], server_address='http://example:11434', request_headers=make_headers)

    sdk_client = AsyncMock()
    sdk_client.chat.return_value = ollama_api.ChatResponse(message=ollama_api.Message(role='assistant', content='hi'))
    sdk_client._client.aclose = AsyncMock()

    action = plugin._create_model_action(ollama_name('m'))
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])

    with patch('ollama.AsyncClient', return_value=sdk_client) as async_client:
        await action._fn(request, None)

    params = cast(RequestHeaderParams, captured['params'])
    assert params.server_address == 'http://example:11434'
    assert params.model is model_def
    assert params.model_request is request
    assert params.embed_request is None
    # The resolved header is applied to the freshly built per-request client.
    assert async_client.call_args.kwargs['headers'] == {'Authorization': 'Bearer tok'}
    # That fresh client's connection pool is closed once the request completes.
    sdk_client._client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_embedder_action_passes_request_context_to_header_callable() -> None:
    """An embedder header callable receives the server address, embedder, and embed request."""
    captured: dict[str, Any] = {}

    def make_headers(params: RequestHeaderParams) -> dict[str, str]:
        captured['params'] = params
        return {'X-Token': 'abc'}

    plugin = Ollama(
        embedders=[EmbeddingDefinition(name='e')],
        server_address='http://example:11434',
        request_headers=make_headers,
    )

    sdk_client = AsyncMock()
    sdk_client.embed.return_value = ollama_api.EmbedResponse(embeddings=[[0.1, 0.2]])
    sdk_client._client.aclose = AsyncMock()

    action = plugin._create_embedder_action(ollama_name('e'))
    request = EmbedRequest(input=[Document.from_text(text='hello')])

    with patch('ollama.AsyncClient', return_value=sdk_client):
        await action._fn(request)

    params = cast(RequestHeaderParams, captured['params'])
    assert params.server_address == 'http://example:11434'
    assert params.model is not None and params.model.name == 'e'
    assert params.embed_request is request
    assert params.model_request is None
    sdk_client._client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_static_headers_reuse_cached_client_and_keep_it_open() -> None:
    """Static headers reuse the per-event-loop cached client and never close it."""
    plugin = Ollama(request_headers={'X-Token': 'abc'})

    async with plugin._client_for_request() as first:
        pass
    async with plugin._client_for_request() as second:
        pass

    # Same shared instance both times, and it was not closed on context exit.
    assert first is second
    assert not first._client.is_closed


@pytest.mark.asyncio
async def test_missing_inner_client_logs_instead_of_leaking() -> None:
    """If a future SDK exposes no _client, cleanup warns rather than silently leaking."""
    plugin = Ollama(request_headers=lambda params: {'X-Token': 't'})

    sdk_client = MagicMock()
    sdk_client._client = None  # simulate an SDK without the private httpx client to close

    with patch('ollama.AsyncClient', return_value=sdk_client):
        with patch('genkit_ollama.plugin_api.logger') as mock_logger:
            async with plugin._client_for_request():
                pass

    cast(MagicMock, mock_logger.warning).assert_called_once()


@pytest.mark.asyncio
async def test_list_actions_wraps_connection_error(ollama_plugin_instance: Ollama) -> None:
    """list_actions surfaces transport failures as OllamaConnectionError."""
    client_mock = MagicMock()
    client_mock.list = AsyncMock(side_effect=httpx.ConnectError('refused'))
    ollama_plugin_instance.client = lambda: client_mock

    with pytest.raises(OllamaConnectionError):
        await ollama_plugin_instance.list_actions()


@pytest.mark.asyncio
async def test_list_actions_does_not_wrap_http_status_error(ollama_plugin_instance: Ollama) -> None:
    """A genuine HTTP status response is not masked as a connection error."""
    request = httpx.Request('GET', 'http://localhost:11434/api/tags')
    response = httpx.Response(500, request=request)
    client_mock = MagicMock()
    client_mock.list = AsyncMock(side_effect=httpx.HTTPStatusError('boom', request=request, response=response))
    ollama_plugin_instance.client = lambda: client_mock

    with pytest.raises(httpx.HTTPStatusError):
        await ollama_plugin_instance.list_actions()


@pytest.mark.asyncio
async def test_model_action_wraps_connection_error() -> None:
    """The model action callable surfaces a down server as OllamaConnectionError.

    The ollama SDK converts ``httpx.ConnectError`` into a builtin
    ``ConnectionError`` before our wrapper sees it, so that is what we simulate.
    """
    plugin = Ollama(models=[ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT)])

    client_mock = MagicMock()
    client_mock.chat = AsyncMock(side_effect=ConnectionError('Failed to connect to Ollama.'))
    # The model captures the client factory when the action is built, so swap it
    # in before resolving the action.
    plugin.client = lambda: client_mock

    action = plugin._create_model_action(ollama_name('m'))
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])

    with pytest.raises(OllamaConnectionError):
        await action._fn(request, None)


@pytest.mark.asyncio
async def test_model_action_wraps_transport_timeout() -> None:
    """Timeouts the SDK does not intercept (httpx.TransportError) are also wrapped."""
    plugin = Ollama(models=[ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT)])

    client_mock = MagicMock()
    client_mock.chat = AsyncMock(side_effect=httpx.ReadTimeout('timed out'))
    plugin.client = lambda: client_mock

    action = plugin._create_model_action(ollama_name('m'))
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])])

    with pytest.raises(OllamaConnectionError):
        await action._fn(request, None)


@pytest.mark.asyncio
async def test_model_action_does_not_wrap_media_fetch_error() -> None:
    """A failed media-URL fetch surfaces raw, not as an Ollama server outage.

    build_chat_messages resolves image URLs (an HTTP fetch) before any Ollama SDK
    call. That transport failure must not be relabelled "Cannot reach the Ollama
    server", which would point users at the wrong fix.
    """
    plugin = Ollama(
        models=[ModelDefinition(name='m', api_type=OllamaAPITypes.CHAT, supports=OllamaSupports(media=True))]
    )

    # The Ollama SDK client must never be reached: image resolution fails first.
    client_mock = MagicMock()
    client_mock.chat = AsyncMock()
    plugin.client = lambda: client_mock

    image_client = MagicMock()
    image_client.get = AsyncMock(side_effect=httpx.ConnectError('image host unreachable'))

    action = plugin._create_model_action(ollama_name('m'))
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[
                    Part(root=MediaPart(media=Media(url='http://imgs.example/cat.jpg', content_type='image/jpeg')))
                ],
            )
        ]
    )

    with patch('genkit_ollama.models.get_cached_client', return_value=image_client):
        # The raw httpx.ConnectError propagates; it is not wrapped as OllamaConnectionError.
        with pytest.raises(httpx.ConnectError):
            await action._fn(request, None)

    client_mock.chat.assert_not_called()


@pytest.mark.asyncio
async def test_embedder_action_wraps_connection_error() -> None:
    """The embedder action surfaces a down server as OllamaConnectionError.

    Mirrors the model/list_actions paths so the embedder endpoint's connection
    wrapping cannot silently regress.
    """
    plugin = Ollama(embedders=[EmbeddingDefinition(name='e')])

    client_mock = MagicMock()
    client_mock.embed = AsyncMock(side_effect=ConnectionError('Failed to connect to Ollama.'))
    plugin.client = lambda: client_mock

    action = plugin._create_embedder_action(ollama_name('e'))
    request = EmbedRequest(input=[Document.from_text(text='hello')])

    with pytest.raises(OllamaConnectionError):
        await action._fn(request)


@pytest.mark.asyncio
async def test_wrap_connection_errors_translates_transport_error() -> None:
    """wrap_connection_errors turns an httpx TransportError into OllamaConnectionError."""
    with pytest.raises(OllamaConnectionError) as exc_info:
        async with wrap_connection_errors('http://localhost:11434'):
            raise httpx.ConnectError('refused')

    assert 'http://localhost:11434' in str(exc_info.value)


@pytest.mark.asyncio
async def test_wrap_connection_errors_timeout_has_distinct_message() -> None:
    """A timeout gets its own 'timed out' message, not the generic unreachable one."""
    with pytest.raises(OllamaConnectionError) as exc_info:
        async with wrap_connection_errors('http://localhost:11434'):
            raise httpx.ReadTimeout('slow')

    message = str(exc_info.value)
    assert 'timed out' in message
    assert 'http://localhost:11434' in message


@pytest.mark.asyncio
async def test_wrap_connection_errors_translates_builtin_connection_error() -> None:
    """wrap_connection_errors turns the SDK's builtin ConnectionError into ours."""
    with pytest.raises(OllamaConnectionError) as exc_info:
        async with wrap_connection_errors('http://localhost:11434'):
            raise ConnectionError('Failed to connect to Ollama.')

    assert 'http://localhost:11434' in str(exc_info.value)


@pytest.mark.asyncio
async def test_wrap_connection_errors_does_not_double_wrap() -> None:
    """An already-actionable OllamaConnectionError passes through unchanged."""
    original = OllamaConnectionError('already wrapped')

    with pytest.raises(OllamaConnectionError) as exc_info:
        async with wrap_connection_errors('http://localhost:11434'):
            raise original

    assert exc_info.value is original


@pytest.mark.asyncio
async def test_wrap_connection_errors_passes_through_http_status_error() -> None:
    """wrap_connection_errors leaves HTTPStatusError untouched."""
    request = httpx.Request('GET', 'http://localhost:11434/api/tags')
    response = httpx.Response(500, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        async with wrap_connection_errors('http://localhost:11434'):
            raise httpx.HTTPStatusError('boom', request=request, response=response)
