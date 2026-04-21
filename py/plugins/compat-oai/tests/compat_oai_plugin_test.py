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
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types import Model

from genkit.plugin_api import ActionKind, ActionMetadata, loop_local_client
from genkit.plugins.compat_oai.openai_plugin import OpenAI, openai_model


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
