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


"""Tests for the OpenAI compatible plugin."""

import asyncio
import queue
import threading
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genkit_openai.models.model_info import SUPPORTED_OPENAI_MODELS
from genkit_openai.openai_plugin import OpenAI, openai_model
from genkit_openai.typing import SupportedOutputFormat
from openai import APIStatusError, APITimeoutError
from openai.types import Model

from genkit import Document, EmbedRequest, GenkitError, Supports
from genkit.plugin_api import ActionKind, ActionMetadata, loop_local_client


@pytest.mark.asyncio
async def test_openai_plugin_init() -> None:
    """Test OpenAI plugin init method."""
    plugin = OpenAI(api_key='test-key')

    # init() should return known models and embedders
    result = await plugin.init()
    assert len(result) > 0, 'Should initialize with known models and embedders'
    assert all(hasattr(action, 'kind') for action in result), 'All actions should have a kind'
    assert all(hasattr(action, 'name') for action in result), 'All actions should have a name'
    assert all(action.name.startswith('openai/') for action in result), (
        "All actions should be namespaced with 'openai/'"
    )

    gpt_image = next(a for a in result if a.name == 'openai/gpt-image-1')
    assert gpt_image.metadata is not None
    gpt_image_model = cast(dict[str, Any], gpt_image.metadata['model'])
    assert gpt_image_model['customOptions']['properties']['quality']['enum'] == ['low', 'medium', 'high']
    assert 'configSchema' not in gpt_image_model

    # Verify we have both models and embedders
    model_actions = [a for a in result if a.kind == ActionKind.MODEL]
    embedder_actions = [a for a in result if a.kind == ActionKind.EMBEDDER]
    assert len(model_actions) > 0, 'Should have at least one model'
    assert len(embedder_actions) > 0, 'Should have at least one embedder'


@pytest.mark.parametrize(
    'kind, name',
    [
        (ActionKind.MODEL, 'gpt-3.5-turbo'),
        # Unlisted ids still resolve when asked for by name.
        (ActionKind.MODEL, 'codex-mini-latest'),
    ],
)
@pytest.mark.asyncio
async def test_openai_plugin_resolve_action(kind: ActionKind, name: str) -> None:
    """Unit Tests for resolve method."""
    plugin = OpenAI(api_key='test-key')

    action = await plugin.resolve(kind, f'openai/{name}')

    assert action is not None
    assert action.name == f'openai/{name}'
    assert action.kind == ActionKind.MODEL


GPT_6_ASTRA_SUPPORTS = {
    'multiturn': True,
    'media': True,
    'tools': False,
    'systemRole': True,
    'output': [SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
}


def test_gpt_6_astra_catalog_entry() -> None:
    """gpt-6-astra is a catalog entry with media, system role, and json, but no tools."""
    info = SUPPORTED_OPENAI_MODELS['gpt-6-astra']

    assert info.label == 'OpenAI - gpt-6-astra'
    assert info.supports == Supports(
        multiturn=True,
        media=True,
        tools=False,
        system_role=True,
        output=[SupportedOutputFormat.JSON_MODE, SupportedOutputFormat.TEXT],
    )


@pytest.mark.asyncio
async def test_gpt_6_astra_registered_without_tools() -> None:
    """init() and resolve() both register gpt-6-astra with the catalog supports."""
    plugin = OpenAI(api_key='test-key')

    init_action = next((a for a in await plugin.init() if a.name == 'openai/gpt-6-astra'), None)
    resolved = await plugin.resolve(ActionKind.MODEL, OpenAI.gpt_model('gpt-6-astra').name)

    assert init_action is not None
    assert resolved is not None
    for action in (init_action, resolved):
        assert action.metadata is not None
        model_meta = cast(dict[str, Any], action.metadata['model'])
        assert model_meta['label'] == 'OpenAI - gpt-6-astra'
        assert model_meta['supports'] == GPT_6_ASTRA_SUPPORTS


@pytest.mark.asyncio
async def test_unlisted_chat_model_resolves_with_default_supports() -> None:
    """An id outside the catalog is registered with multiturn only, so tools, media, and json stay hidden."""
    plugin = OpenAI(api_key='test-key')

    action = await plugin.resolve(ActionKind.MODEL, 'openai/gpt-6-nova')

    assert action is not None
    assert action.metadata is not None
    model_meta = cast(dict[str, Any], action.metadata['model'])
    assert model_meta['supports'] == {'multiturn': True}


@pytest.mark.asyncio
async def test_openai_plugin_list_actions() -> None:
    """Test OpenAI plugin list_actions method."""
    entries = [
        Model(id='gpt-4-0613', created=1686588896, object='model', owned_by='openai'),
        Model(id='gpt-4', created=1687882411, object='model', owned_by='openai'),
        Model(id='gpt-3.5-turbo', created=1677610602, object='model', owned_by='openai'),
        Model(id='gpt-image-1', created=1744060800, object='model', owned_by='openai'),
        Model(id='o4-mini-deep-research-2025-06-26', created=1750866121, object='model', owned_by='system'),
        Model(id='codex-mini-latest', created=1746673257, object='model', owned_by='system'),
        Model(id='babbage-002', created=1692634615, object='model', owned_by='system'),
        Model(id='davinci-002', created=1692634301, object='model', owned_by='system'),
        Model(id='o3-pro', created=1748475948, object='model', owned_by='system'),
        Model(id='text-embedding-ada-002', created=1671217299, object='model', owned_by='openai-internal'),
    ]
    plugin = OpenAI(api_key='test-key')
    mock_client = MagicMock()

    mock_result_ = MagicMock()
    mock_result_.data = entries
    mock_client.models.list = AsyncMock(return_value=mock_result_)

    plugin._runtime_client = lambda: mock_client

    actions: list[ActionMetadata] = await plugin.list_actions()
    mock_client.models.list.assert_called_once()
    _ = await plugin.list_actions()
    # list_actions is cached after the first API fetch.
    assert mock_client.models.list.call_count == 1

    # Ids Chat Completions does not serve are dropped from the listing.
    listed = {action.name for action in actions}
    assert 'openai/codex-mini-latest' not in listed
    assert 'openai/babbage-002' not in listed
    assert 'openai/davinci-002' not in listed
    assert 'openai/o3-pro' not in listed
    assert len(actions) == len(entries) - 4
    assert actions[0].name == 'openai/gpt-4-0613'
    assert actions[-1].name == 'openai/text-embedding-ada-002'

    chat = next(a for a in actions if a.name == 'openai/gpt-4')
    assert chat.metadata is not None
    chat_props = chat.metadata['model']['customOptions']['properties']
    assert 'frequencyPenalty' in chat_props
    assert 'maxTokens' in chat_props

    image = next(a for a in actions if a.name == 'openai/gpt-image-1')
    assert image.metadata is not None
    image_model = cast(dict[str, Any], image.metadata['model'])
    assert image_model['customOptions']['properties']['quality']['enum'] == ['low', 'medium', 'high']
    assert 'configSchema' not in image_model

    embed = next(a for a in actions if a.name == 'openai/text-embedding-ada-002')
    assert embed.metadata is not None
    embed_options = embed.metadata.get('embedder', {}).get('customOptions')
    assert embed_options is None or 'frequencyPenalty' not in (embed_options.get('properties') or {})


@pytest.mark.asyncio
async def test_openai_runtime_clients_are_loop_local() -> None:
    """Runtime OpenAI clients are cached per event loop."""
    plugin = OpenAI(api_key='test-key')
    plugin._runtime_client = loop_local_client(lambda: object())

    first = plugin._runtime_client()
    second = plugin._runtime_client()
    assert first is second

    q: queue.Queue[object] = queue.Queue()

    def _other_thread() -> None:
        async def _get_client() -> object:
            return plugin._runtime_client()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            q.put(loop.run_until_complete(_get_client()))
        finally:
            loop.close()

    t = threading.Thread(target=_other_thread, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()

    other_loop_client = q.get_nowait()
    assert other_loop_client is not first


@pytest.mark.parametrize(
    'kind, name',
    [(ActionKind.MODEL, 'model_doesnt_exist')],
)
@pytest.mark.asyncio
async def test_openai_plugin_resolve_action_not_found(kind: ActionKind, name: str) -> None:
    """Unit Tests for resolve method with non-existent model."""
    plugin = OpenAI(api_key='test-key')
    action = await plugin.resolve(kind, f'openai/{name}')

    # Should still return an action even for unknown models
    assert action is not None
    assert action.name == f'openai/{name}'


def test_openai_model_function() -> None:
    """Test openai_model function."""
    assert openai_model('gpt-4') == 'openai/gpt-4'


_ERROR_MESSAGE = 'OpenAI request failed'


def _http_request() -> httpx.Request:
    """Create the request required by OpenAI SDK errors."""
    return httpx.Request('POST', 'https://api.openai.com/v1/embeddings')


def _status_error(status_code: int, retry_after: str | None = None) -> APIStatusError:
    """Create a real OpenAI status error."""
    headers = {'retry-after': retry_after} if retry_after is not None else None
    response = httpx.Response(status_code, request=_http_request(), headers=headers)
    return APIStatusError(_ERROR_MESSAGE, response=response, body={'error': {'message': _ERROR_MESSAGE}})


def _plugin_with(client: MagicMock) -> OpenAI:
    """Create a plugin bound to a stub client."""
    plugin = OpenAI(api_key='test-key')
    plugin._runtime_client = lambda: client
    return plugin


def _embedder_client(error: Exception) -> MagicMock:
    """Create a stub client whose embeddings call raises an error."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(side_effect=error)
    return client


def _embedding_client() -> MagicMock:
    """Create a stub client whose embeddings call returns one vector."""
    client = MagicMock()
    item = MagicMock()
    item.embedding = [0.1, 0.2]
    result = MagicMock()
    result.data = [item]
    client.embeddings.create = AsyncMock(return_value=result)
    return client


async def _run_embedder(client: MagicMock, options: dict[str, Any] | None = None) -> None:
    """Run the embedder action function against a stub client."""
    action = _plugin_with(client)._create_embedder_action('openai/text-embedding-3-small')
    await action._fn(EmbedRequest(input=[Document.from_text('hello')], options=options))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'options',
    [
        None,
        {'dimensions': 256},
        {'encodingFormat': 'float'},
        {'dimensions': 256, 'encodingFormat': 'base64'},
    ],
    ids=['no-options', 'dimensions', 'encoding-format', 'both'],
)
async def test_embedder_maps_status_errors_for_every_option_shape(options: dict[str, Any] | None) -> None:
    """Each embeddings call variant maps its failures."""
    api_error = _status_error(401)

    with pytest.raises(GenkitError) as exc_info:
        await _run_embedder(_embedder_client(api_error), options=options)

    assert exc_info.value.status == 'UNAUTHENTICATED'
    assert exc_info.value.__cause__ is api_error


@pytest.mark.asyncio
async def test_embedder_carries_retry_after_metadata() -> None:
    """A rate-limited embed reports RESOURCE_EXHAUSTED with the parsed delay."""
    api_error = _status_error(429, retry_after='2.5')

    with pytest.raises(GenkitError) as exc_info:
        await _run_embedder(_embedder_client(api_error))

    assert exc_info.value.status == 'RESOURCE_EXHAUSTED'
    assert exc_info.value.__cause__ is api_error
    assert exc_info.value.response_metadata == {'retry_after_ms': 2500.0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'dimensions',
    ['lots', '256', 256.9, True, [256]],
    ids=['non-numeric-str', 'numeric-str', 'float', 'bool', 'list'],
)
async def test_embedder_classifies_bad_dimensions_option(dimensions: Any) -> None:
    """A dimensions option that is not an int is INVALID_ARGUMENT, before any API call."""
    client = MagicMock()
    client.embeddings.create = AsyncMock()

    with pytest.raises(GenkitError) as exc_info:
        await _run_embedder(client, options={'dimensions': dimensions})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert 'dimensions' in str(exc_info.value)
    client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
async def test_embedder_forwards_zero_dimensions() -> None:
    """A dimensions of 0 reaches the API rather than being dropped as falsy."""
    client = _embedding_client()

    await _run_embedder(client, options={'dimensions': 0})

    assert client.embeddings.create.await_args.kwargs['dimensions'] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [
        APITimeoutError(request=_http_request()),
        ValueError('No embedding data received'),
        RuntimeError('unexpected failure'),
    ],
    ids=['timeout', 'sdk-value-error', 'runtime-error'],
)
async def test_embedder_propagates_unclassified_errors(error: Exception) -> None:
    """An error without a failing HTTP status escapes the embedder unchanged.

    A ValueError from the SDK is a provider-side failure, not caller input,
    so it must not be reported as INVALID_ARGUMENT."""
    with pytest.raises(Exception) as exc_info:
        await _run_embedder(_embedder_client(error))

    assert exc_info.value is error
    assert not isinstance(exc_info.value, GenkitError)


def _list_client(*results: Any) -> MagicMock:
    """Create a stub client whose model listing yields the given results in order."""
    client = MagicMock()
    client.models.list = AsyncMock(side_effect=list(results))
    return client


@pytest.mark.asyncio
async def test_list_actions_maps_status_errors() -> None:
    """A failed model listing reports the status the HTTP response carried."""
    api_error = _status_error(503)

    with pytest.raises(GenkitError) as exc_info:
        await _plugin_with(_list_client(api_error)).list_actions()

    assert exc_info.value.status == 'UNAVAILABLE'
    assert exc_info.value.__cause__ is api_error


@pytest.mark.asyncio
async def test_list_actions_error_is_not_cached() -> None:
    """A failed model listing is retried on the next call, not cached."""
    ok_result = MagicMock()
    ok_result.data = [Model(id='gpt-4', created=1687882411, object='model', owned_by='openai')]
    client = _list_client(_status_error(503), ok_result)
    plugin = _plugin_with(client)

    with pytest.raises(GenkitError):
        await plugin.list_actions()

    actions = await plugin.list_actions()
    assert [a.name for a in actions] == ['openai/gpt-4']
    assert client.models.list.call_count == 2


@pytest.mark.asyncio
async def test_list_actions_propagates_unclassified_errors() -> None:
    """An error without a failing HTTP status escapes list_actions unchanged."""
    error = APITimeoutError(request=_http_request())

    with pytest.raises(APITimeoutError) as exc_info:
        await _plugin_with(_list_client(error)).list_actions()

    assert exc_info.value is error
    assert not isinstance(exc_info.value, GenkitError)
