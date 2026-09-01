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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genkit_openai.openai_plugin import OpenAI, openai_model
from openai import APIStatusError, APITimeoutError
from openai.types import Model

from genkit import Document, EmbedRequest, GenkitError
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

    # Verify we have both models and embedders
    model_actions = [a for a in result if a.kind == ActionKind.MODEL]
    embedder_actions = [a for a in result if a.kind == ActionKind.EMBEDDER]
    assert len(model_actions) > 0, 'Should have at least one model'
    assert len(embedder_actions) > 0, 'Should have at least one embedder'


@pytest.mark.parametrize(
    'kind, name',
    [(ActionKind.MODEL, 'gpt-3.5-turbo')],
)
@pytest.mark.asyncio
async def test_openai_plugin_resolve_action(kind: ActionKind, name: str) -> None:
    """Unit Tests for resolve method."""
    plugin = OpenAI(api_key='test-key')

    action = await plugin.resolve(kind, f'openai/{name}')

    assert action is not None
    assert action.name == f'openai/{name}'
    assert action.kind == ActionKind.MODEL


@pytest.mark.asyncio
async def test_openai_plugin_list_actions() -> None:
    """Test OpenAI plugin list_actions method."""
    entries = [
        Model(id='gpt-4-0613', created=1686588896, object='model', owned_by='openai'),
        Model(id='gpt-4', created=1687882411, object='model', owned_by='openai'),
        Model(id='gpt-3.5-turbo', created=1677610602, object='model', owned_by='openai'),
        Model(id='o4-mini-deep-research-2025-06-26', created=1750866121, object='model', owned_by='system'),
        Model(id='codex-mini-latest', created=1746673257, object='model', owned_by='system'),
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

    assert len(actions) == len(entries)
    assert actions[0].name == 'openai/gpt-4-0613'
    assert actions[-1].name == 'openai/text-embedding-ada-002'

    chat = next(a for a in actions if a.name == 'openai/gpt-4')
    assert chat.metadata is not None
    chat_props = chat.metadata['model']['customOptions']['properties']
    assert 'frequencyPenalty' in chat_props
    assert 'maxTokens' in chat_props

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
async def test_embedder_classifies_bad_dimensions_option() -> None:
    """A non-numeric dimensions option is INVALID_ARGUMENT, before any API call."""
    client = MagicMock()
    client.embeddings.create = AsyncMock()

    with pytest.raises(GenkitError) as exc_info:
        await _run_embedder(client, options={'dimensions': 'lots'})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    client.embeddings.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [
        APITimeoutError(request=_http_request()),
        RuntimeError('unexpected failure'),
    ],
    ids=['timeout', 'runtime-error'],
)
async def test_embedder_propagates_unclassified_errors(error: Exception) -> None:
    """An error without a failing HTTP status escapes the embedder unchanged."""
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
