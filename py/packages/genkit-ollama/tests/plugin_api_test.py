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

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import ollama as ollama_api
import pytest
from genkit_ollama import Ollama, OllamaConnectionError, _plugin as plugin_module
from genkit_ollama._errors import wrap_connection_errors
from genkit_ollama._models import OllamaConfig, OllamaModel, _ResolvedModel
from pydantic import BaseModel

from genkit import Document, Genkit, GenkitError, Message, ModelResponse, Part, Role
from genkit.embedder import EmbedRequest
from genkit.model import ModelRequest
from genkit.plugin_api import ActionKind, to_json_schema


def test_init_with_options() -> None:
    """server_address and static headers land on the plugin."""
    plugin = Ollama(server_address='new.server.address', request_headers={'Content-Type': 'json'})

    assert plugin.server_address == 'new.server.address'
    assert plugin.request_headers == {'Content-Type': 'json'}


@pytest.mark.asyncio
async def test_initialize_registers_nothing(ollama_plugin_instance: Ollama) -> None:
    """Models and embedders resolve on demand; nothing is registered eagerly."""
    # An eagerly registered model would carry generic metadata and win over
    # the probed listing row in the Dev UI catalog.
    assert await ollama_plugin_instance.init() == []


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
    action = await ollama_plugin_instance.resolve(kind, name)

    assert action is not None
    assert action.kind == kind
    assert action.name == f'ollama/{name}'
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
    plugin = Ollama()
    action = plugin._create_model_action(_ResolvedModel(name='llava', media=True))

    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['multiturn'] is True
    assert supports['tools'] is True
    assert supports['media'] is True


@pytest.mark.asyncio
async def test_dynamic_model_advertises_generic_capabilities() -> None:
    """The fallback definition advertises the full generic capability set,
    matching the JS GENERIC_MODEL_INFO and the Go defaultOllamaSupports for
    un-probed models."""
    plugin = Ollama()
    action = plugin._create_model_action(_ResolvedModel(name='some-unconfigured-model'))

    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['multiturn'] is True
    assert supports['tools'] is True
    assert supports['media'] is True
    assert supports['systemRole'] is True


@pytest.mark.asyncio
async def test_create_model_action_custom_options_is_ollama_config() -> None:
    """The model action advertises OllamaConfig (with Ollama-only knobs) as its schema."""
    plugin = Ollama()
    action = plugin._create_model_action(_ResolvedModel(name='m'))

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


def _fake_server(seen: list[httpx.Request], tags: list[str] | None = None) -> Callable[[httpx.Request], httpx.Response]:
    """A minimal Ollama server: records each request and answers the endpoints the plugin calls."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        match request.url.path:
            case '/api/show':
                return httpx.Response(200, json={'capabilities': ['completion'], 'template': '', 'model_info': {}})
            case '/api/tags':
                return httpx.Response(200, json={'models': [{'model': name, 'name': name} for name in tags or []]})
            case '/api/chat':
                return httpx.Response(
                    200, json={'model': 'm', 'message': {'role': 'assistant', 'content': 'Tartine'}, 'done': True}
                )
            case '/api/embed':
                return httpx.Response(200, json={'model': 'e', 'embeddings': [[0.1, 0.2]]})
        return httpx.Response(404)

    return handle


@contextmanager
def _real_client_on(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[MagicMock]:
    """Real ollama.AsyncClient (so httpx auth runs), with requests served by ``handler``."""
    real = ollama_api.AsyncClient
    with patch(
        'ollama.AsyncClient', side_effect=lambda **kwargs: real(transport=httpx.MockTransport(handler), **kwargs)
    ) as factory:
        yield factory


def _hello() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])])


def test_make_client_routes_a_header_callable_through_httpx_auth() -> None:
    """A callable becomes an httpx.Auth hook; nothing is baked into the static headers."""
    plugin = Ollama(request_headers=lambda: {'Authorization': 'Bearer tok'})

    with patch('ollama.AsyncClient') as async_client:
        plugin._make_client()

    kwargs = async_client.call_args.kwargs
    assert kwargs['headers'] == {}
    assert isinstance(kwargs['auth'], httpx.Auth)


@pytest.mark.asyncio
async def test_a_sync_header_callable_runs_per_request_on_one_client() -> None:
    """A rotating token reaches each request, and every request shares one pooled client."""
    tokens = iter(['Bearer t1', 'Bearer t2'])
    plugin = Ollama(request_headers=lambda: {'Authorization': next(tokens)})
    seen: list[httpx.Request] = []
    action = plugin._create_model_action(_ResolvedModel(name='m'))

    with _real_client_on(_fake_server(seen)) as factory:
        await action._fn(_hello(), None)
        await action._fn(_hello(), None)

    assert [r.headers['authorization'] for r in seen] == ['Bearer t1', 'Bearer t2']
    assert factory.call_count == 1


@pytest.mark.asyncio
async def test_an_async_header_callable_is_awaited_per_request() -> None:
    """An async callable is awaited before each request, model and embedder alike."""
    tokens = iter(['Bearer a1', 'Bearer a2'])

    async def mint() -> dict[str, str]:
        return {'Authorization': next(tokens)}

    plugin = Ollama(request_headers=mint)
    seen: list[httpx.Request] = []

    with _real_client_on(_fake_server(seen)):
        await plugin._create_model_action(_ResolvedModel(name='m'))._fn(_hello(), None)
        await plugin._create_embedder_action('e')._fn(EmbedRequest(input=[Document.from_text(text='menu')]))

    assert [(r.url.path, r.headers['authorization']) for r in seen] == [
        ('/api/chat', 'Bearer a1'),
        ('/api/embed', 'Bearer a2'),
    ]


@pytest.mark.asyncio
async def test_static_headers_are_sent_without_an_auth_hook() -> None:
    """A dict is baked into the shared client and sent on every request."""
    plugin = Ollama(request_headers={'X-Token': 'abc'})
    seen: list[httpx.Request] = []

    with _real_client_on(_fake_server(seen)) as factory:
        await plugin._create_model_action(_ResolvedModel(name='m'))._fn(_hello(), None)

    assert seen[0].headers['x-token'] == 'abc'
    assert 'auth' not in factory.call_args.kwargs


@pytest.mark.asyncio
async def test_the_client_is_cached_per_event_loop() -> None:
    """Repeated calls on one loop reuse the same client and its connection pool."""
    plugin = Ollama(request_headers=lambda: {'X-Token': 't'})

    assert plugin.client() is plugin.client()


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
async def test_list_actions_classifies_response_error(ollama_plugin_instance: Ollama) -> None:
    """An Ollama ResponseError from /api/tags carries the server's HTTP status."""
    error = ollama_api.ResponseError('unauthorized', 401)
    client_mock = MagicMock()
    client_mock.list = AsyncMock(side_effect=error)
    ollama_plugin_instance.client = lambda: client_mock

    with pytest.raises(GenkitError) as exc_info:
        await ollama_plugin_instance.list_actions()

    assert exc_info.value.status == 'UNAUTHENTICATED'
    assert exc_info.value.__cause__ is error


@pytest.mark.asyncio
async def test_model_action_wraps_connection_error() -> None:
    """The model action callable surfaces a down server as OllamaConnectionError.

    The ollama SDK converts ``httpx.ConnectError`` into a builtin
    ``ConnectionError`` before our wrapper sees it, so that is what we simulate.
    """
    plugin = Ollama()

    client_mock = MagicMock()
    client_mock.chat = AsyncMock(side_effect=ConnectionError('Failed to connect to Ollama.'))
    # The model captures the client factory when the action is built, so swap it
    # in before resolving the action.
    plugin.client = lambda: client_mock

    action = plugin._create_model_action(_ResolvedModel(name='m'))
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Hello')])])

    with pytest.raises(OllamaConnectionError):
        await action._fn(request, None)


@pytest.mark.asyncio
async def test_model_action_wraps_transport_timeout() -> None:
    """Timeouts the SDK does not intercept (httpx.TransportError) are also wrapped."""
    plugin = Ollama()

    client_mock = MagicMock()
    client_mock.chat = AsyncMock(side_effect=httpx.ReadTimeout('timed out'))
    plugin.client = lambda: client_mock

    action = plugin._create_model_action(_ResolvedModel(name='m'))
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Hello')])])

    with pytest.raises(OllamaConnectionError):
        await action._fn(request, None)


@pytest.mark.asyncio
async def test_model_action_does_not_wrap_media_fetch_error() -> None:
    """A failed media-URL fetch surfaces raw, not as an Ollama server outage.

    build_chat_messages resolves image URLs (an HTTP fetch) before any Ollama SDK
    call. That transport failure must not be relabelled "Cannot reach the Ollama
    server", which would point users at the wrong fix.
    """
    plugin = Ollama()

    # The Ollama SDK client must never be reached: image resolution fails first.
    client_mock = MagicMock()
    client_mock.chat = AsyncMock()
    plugin.client = lambda: client_mock

    image_client = MagicMock()
    image_client.get = AsyncMock(side_effect=httpx.ConnectError('image host unreachable'))

    action = plugin._create_model_action(_ResolvedModel(name='m'))
    request = ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part.from_media('http://imgs.example/cat.jpg', content_type='image/jpeg')],
            )
        ]
    )

    with patch('genkit_ollama._models._image_fetch_client', return_value=image_client):
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
    plugin = Ollama()

    client_mock = MagicMock()
    client_mock.embed = AsyncMock(side_effect=ConnectionError('Failed to connect to Ollama.'))
    plugin.client = lambda: client_mock

    action = plugin._create_embedder_action('e')
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


def test_connection_error_is_unclassified() -> None:
    """A down server has no reported status: Retry retries it, Fallback does not switch models.

    Matches Go and the other plugins' raw transport errors.
    """
    error = OllamaConnectionError('Cannot reach the Ollama server.')

    assert isinstance(error, ConnectionError)
    assert not isinstance(error, GenkitError)


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


@pytest.mark.asyncio
async def test_generate_ollama_id_with_ollama_segment_sends_id_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ollama/ollama/llama3` sends `ollama/llama3` to the Ollama server."""
    seen: list[str] = []

    async def fake_generate(self: OllamaModel, request: object, ctx: object, client: object = None) -> ModelResponse:
        seen.append(self.model_definition.name)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    monkeypatch.setattr(OllamaModel, 'generate', fake_generate)
    ai = Genkit(plugins=[Ollama()])

    response = await ai.generate(model='ollama/ollama/llama3', prompt='hi')

    assert response.text == 'ok'
    assert seen == ['ollama/llama3']


def _show(capabilities: list[str] | None, template: str | None = '{{ range .Messages }}{{ .Content }}{{ end }}') -> Any:
    """An /api/show answer as the SDK parses it."""
    body: dict[str, Any] = {'template': template, 'model_info': {}}
    if capabilities is not None:
        body['capabilities'] = capabilities
    return ollama_api.ShowResponse.model_validate(body)


def _plugin_with_show(show: AsyncMock, **kwargs: Any) -> tuple[Ollama, MagicMock]:
    plugin = Ollama(**kwargs)
    client_mock = MagicMock()
    client_mock.show = show
    plugin.client = lambda: client_mock
    return plugin, client_mock


async def _supports(plugin: Ollama, name: str) -> dict[str, Any]:
    action = await plugin.resolve(ActionKind.MODEL, name)
    assert action is not None
    return cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])


@pytest.mark.asyncio
async def test_resolve_reads_tools_and_vision_from_show() -> None:
    """A vision + tools model advertises both, on /api/chat."""
    show = AsyncMock(return_value=_show(['completion', 'tools', 'vision']))
    plugin, _ = _plugin_with_show(show)

    supports = await _supports(plugin, 'llava')

    assert supports['tools'] is True
    assert supports['media'] is True
    assert supports['multiturn'] is True
    show.assert_awaited_once_with('llava')


@pytest.mark.asyncio
async def test_resolve_turns_off_what_show_does_not_report() -> None:
    """A text-only chat model loses the generic tools and media flags."""
    show = AsyncMock(return_value=_show(['completion']))
    plugin, _ = _plugin_with_show(show)

    supports = await _supports(plugin, 'gemma2')

    assert supports['tools'] is False
    assert supports['media'] is False
    assert supports['multiturn'] is True


@pytest.mark.asyncio
async def test_resolve_routes_an_embedding_model_off_chat_capabilities() -> None:
    """An embedding model reports neither tools nor vision, so neither is advertised."""
    show = AsyncMock(return_value=_show(['embedding'], template=''))
    plugin, _ = _plugin_with_show(show)

    supports = await _supports(plugin, 'nomic-embed-text')

    assert supports['tools'] is False
    assert supports['media'] is False


@pytest.mark.parametrize('template', ['{{ .Prompt }}', '', None])
@pytest.mark.asyncio
async def test_a_template_less_model_stays_on_chat_with_format(template: str | None) -> None:
    """A model with no chat template still goes to /api/chat, so output_schema reaches it as `format`."""
    show = AsyncMock(return_value=_show(['completion'], template=template))
    plugin, client_mock = _plugin_with_show(show)
    client_mock.chat = AsyncMock(
        return_value=ollama_api.ChatResponse(
            message=ollama_api.Message(role='assistant', content='{"dish": "Tartine"}')
        )
    )
    client_mock.generate = AsyncMock()

    action = await plugin.resolve(ActionKind.MODEL, 'llama2-base')
    assert action is not None
    schema = {'type': 'object', 'properties': {'dish': {'type': 'string'}}}
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])])
    request.output_schema = schema
    await action._fn(request, None)

    assert client_mock.chat.await_args is not None
    assert client_mock.chat.await_args.kwargs['format'] == schema
    client_mock.generate.assert_not_called()


@pytest.mark.asyncio
async def test_a_natively_rendered_chat_model_stays_on_chat() -> None:
    """A raw template next to tools means the server renders chat itself."""
    show = AsyncMock(return_value=_show(['completion', 'tools', 'thinking'], template='{{ .Prompt }}'))
    plugin, _ = _plugin_with_show(show)

    supports = await _supports(plugin, 'gpt-oss')

    assert supports['multiturn'] is True
    assert supports['tools'] is True


@pytest.mark.parametrize(
    'error',
    [
        ollama_api.ResponseError('model "llava" not found', 404),
        ConnectionError('Failed to connect to Ollama.'),
        httpx.ConnectError('refused'),
    ],
)
@pytest.mark.asyncio
async def test_a_failed_probe_falls_back_to_generic_chat(error: Exception) -> None:
    """Server down or model missing: today's dynamic defaults, and resolve still succeeds."""
    plugin, _ = _plugin_with_show(AsyncMock(side_effect=error))

    action = await plugin.resolve(ActionKind.MODEL, 'llava')

    assert action is not None
    supports = cast(dict[str, Any], cast(dict[str, Any], action.metadata)['model']['supports'])
    assert supports['tools'] is True
    assert supports['media'] is True
    assert supports['multiturn'] is True


@pytest.mark.asyncio
async def test_an_old_server_without_capabilities_falls_back() -> None:
    """A show answer with no capabilities field keeps the generic defaults."""
    plugin, _ = _plugin_with_show(AsyncMock(return_value=SimpleNamespace(template='{{ .Prompt }}')))

    supports = await _supports(plugin, 'llama2')

    assert supports['tools'] is True
    assert supports['media'] is True
    assert supports['multiturn'] is True


@pytest.mark.asyncio
async def test_a_slow_probe_times_out_to_the_fallback() -> None:
    """A probe is bounded by the plugin timeout when that is under five seconds."""

    async def hang(name: str) -> Any:
        await asyncio.sleep(10)

    plugin, _ = _plugin_with_show(AsyncMock(side_effect=hang), timeout=0.01)

    supports = await _supports(plugin, 'llava')

    assert supports['tools'] is True


@pytest.mark.asyncio
async def test_a_raising_header_callable_does_not_break_resolve() -> None:
    """The probe resolves headers too; a failure there is just a failed probe."""

    def headers() -> dict[str, str]:
        raise RuntimeError('token service down')

    plugin = Ollama(request_headers=headers)

    action = await plugin.resolve(ActionKind.MODEL, 'llava')

    assert action is not None


@pytest.mark.asyncio
async def test_a_model_is_probed_once() -> None:
    """A successful probe is cached; ``:latest`` names the same model."""
    show = AsyncMock(return_value=_show(['completion', 'tools']))
    plugin, _ = _plugin_with_show(show)

    await plugin.resolve(ActionKind.MODEL, 'llama3.2')
    await plugin.resolve(ActionKind.MODEL, 'llama3.2:latest')

    show.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failed_probe_is_retried_after_its_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that was down gets asked again once the 30s failure entry expires."""
    now = [1000.0]
    monkeypatch.setattr(plugin_module.time, 'monotonic', lambda: now[0])
    show = AsyncMock(side_effect=[ConnectionError('down'), _show(['completion'])])
    plugin, _ = _plugin_with_show(show)

    assert (await _supports(plugin, 'gemma2'))['tools'] is True
    now[0] += 10
    assert (await _supports(plugin, 'gemma2'))['tools'] is True
    assert show.await_count == 1

    now[0] += 30
    assert (await _supports(plugin, 'gemma2'))['tools'] is False
    assert show.await_count == 2


class _Tag(BaseModel):
    model: str
    digest: str = ''


class _Tags(BaseModel):
    models: list[_Tag]


@pytest.mark.asyncio
async def test_list_actions_probes_each_model_and_skips_embedders() -> None:
    """The Dev UI rows carry probed capabilities; embedders are not probed."""
    answers = {
        'llava': _show(['completion', 'tools', 'vision']),
        'gemma2': _show(['completion']),
    }

    async def show(name: str) -> Any:
        if name not in answers:
            raise ollama_api.ResponseError(f'model "{name}" not found', 404)
        return answers[name]

    show_mock = AsyncMock(side_effect=show)
    plugin, client_mock = _plugin_with_show(show_mock)
    client_mock.list = AsyncMock(
        return_value=_Tags(models=[_Tag(model='llava'), _Tag(model='gemma2'), _Tag(model='nomic-embed-text')])
    )

    actions = await plugin.list_actions()

    rows = {a.name: a for a in actions}
    assert list(rows) == ['ollama/llava', 'ollama/gemma2', 'ollama/nomic-embed-text']

    def supports(name: str) -> dict[str, Any]:
        metadata = rows[name].metadata
        assert metadata is not None
        return cast(dict[str, Any], metadata['model']['supports'])

    assert supports('ollama/llava')['media'] is True
    assert supports('ollama/gemma2')['tools'] is False
    assert sorted(c.args[0] for c in show_mock.await_args_list) == ['gemma2', 'llava']


@pytest.mark.asyncio
async def test_list_actions_bounds_concurrent_probes() -> None:
    """At most four probes are in flight at once."""
    in_flight = 0
    peak = 0

    async def show(name: str) -> Any:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return _show(['completion'])

    plugin, client_mock = _plugin_with_show(AsyncMock(side_effect=show))
    client_mock.list = AsyncMock(return_value=_Tags(models=[_Tag(model=f'menu-model-{i}') for i in range(10)]))

    actions = await plugin.list_actions()

    assert len(actions) == 10
    assert peak == 4


@pytest.mark.asyncio
async def test_list_actions_reprobes_a_repulled_model() -> None:
    """A new digest from /api/tags invalidates the cached capabilities."""
    show = AsyncMock(side_effect=[_show(['completion']), _show(['completion', 'tools'])])
    plugin, client_mock = _plugin_with_show(show)

    client_mock.list = AsyncMock(return_value=_Tags(models=[_Tag(model='qwen3', digest='a')]))
    await plugin.list_actions()
    await plugin.list_actions()
    assert show.await_count == 1

    client_mock.list = AsyncMock(return_value=_Tags(models=[_Tag(model='qwen3', digest='b')]))
    actions = await plugin.list_actions()

    assert show.await_count == 2
    assert actions[0].metadata is not None
    assert cast(dict[str, Any], actions[0].metadata['model']['supports'])['tools'] is True


@pytest.mark.asyncio
async def test_a_cached_probe_keeps_the_requested_name() -> None:
    """``llama3.2:latest`` and ``llama3.2`` share a probe, but each action keeps its own name."""
    show = AsyncMock(return_value=_show(['completion', 'tools']))
    plugin, client_mock = _plugin_with_show(show)
    client_mock.chat = AsyncMock(
        return_value=ollama_api.ChatResponse(message=ollama_api.Message(role='assistant', content='Tartine'))
    )

    await plugin.resolve(ActionKind.MODEL, 'llama3.2:latest')
    action = await plugin.resolve(ActionKind.MODEL, 'llama3.2')
    assert action is not None
    await action._fn(ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])]), None)

    show.assert_awaited_once()
    assert cast(dict[str, Any], action.metadata)['model']['label'] == 'Ollama - llama3.2'
    assert client_mock.chat.await_args is not None
    assert client_mock.chat.await_args.kwargs['model'] == 'llama3.2'


@pytest.mark.asyncio
async def test_concurrent_first_resolves_share_one_probe() -> None:
    """Five parallel first uses of one model send one /api/show."""

    async def slow_show(name: str) -> Any:
        await asyncio.sleep(0.01)
        return _show(['completion', 'tools'])

    show = AsyncMock(side_effect=slow_show)
    plugin, _ = _plugin_with_show(show)

    actions = await asyncio.gather(*(plugin.resolve(ActionKind.MODEL, 'qwen3') for _ in range(5)))

    assert all(a is not None for a in actions)
    show.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_actions_and_its_probes_share_one_client() -> None:
    """A Dev UI refresh opens one pool; the header callable runs once per HTTP request."""
    minted: list[int] = []

    def mint() -> dict[str, str]:
        minted.append(1)
        return {'Authorization': 'Bearer tok'}

    plugin = Ollama(request_headers=mint)
    seen: list[httpx.Request] = []

    with _real_client_on(_fake_server(seen, tags=[f'menu-model-{i}' for i in range(3)])) as factory:
        actions = await plugin.list_actions()

    assert len(actions) == 3
    assert [r.url.path for r in seen].count('/api/show') == 3
    assert len(minted) == len(seen) == 4
    assert factory.call_count == 1


@pytest.mark.asyncio
async def test_the_probe_timeout_covers_a_slow_header_callable() -> None:
    """A header mint that hangs is cut off by the probe timeout, not just the HTTP call."""

    async def headers() -> dict[str, str]:
        await asyncio.sleep(10)
        return {}

    plugin = Ollama(request_headers=headers, timeout=0.01)

    supports = await asyncio.wait_for(_supports(plugin, 'llava'), timeout=1)

    assert supports['tools'] is True
