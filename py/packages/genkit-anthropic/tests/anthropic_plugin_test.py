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

"""Tests for Anthropic plugin."""

import asyncio
import queue
import threading
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from genkit_anthropic import Anthropic, anthropic_name
from genkit_anthropic.model_info import (
    SUPPORTED_ANTHROPIC_MODELS as SUPPORTED_MODELS,
    get_model_info,
)

from genkit import (
    ActionKind,
    Constrained,
    Message,
    ModelConfig,
    ModelRequest,
    Part,
    Role,
    TextPart,
    ToolDefinition,
)


def test_anthropic_name() -> None:
    """Test anthropic_name helper function."""
    assert anthropic_name('claude-sonnet-4') == 'anthropic/claude-sonnet-4'


def test_init_with_api_key() -> None:
    """Test plugin initialization with API key."""
    plugin = Anthropic(api_key='test-key')

    async def _get_api_key() -> str | None:
        return plugin._runtime_client().api_key

    assert asyncio.run(_get_api_key()) == 'test-key'
    assert plugin.models == list(SUPPORTED_MODELS.keys())


def test_init_without_api_key_raises() -> None:
    """Test plugin initialization without API key uses default behavior."""
    with patch.dict('os.environ', {}, clear=True):
        # AsyncAnthropic allows initialization without API key
        # Error only occurs when making actual API calls
        plugin = Anthropic()

        async def _has_client() -> bool:
            return plugin._runtime_client() is not None

        assert asyncio.run(_has_client())


def test_init_with_env_var() -> None:
    """Test plugin initialization with environment variable."""
    with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'env-key'}):
        plugin = Anthropic()

        async def _get_api_key() -> str | None:
            return plugin._runtime_client().api_key

        assert asyncio.run(_get_api_key()) == 'env-key'


def test_custom_models() -> None:
    """Test plugin initialization with custom models."""
    plugin = Anthropic(api_key='test-key', models=['claude-sonnet-4'])
    assert plugin.models == ['claude-sonnet-4']


@pytest.mark.asyncio
async def test_plugin_init() -> None:
    """Test plugin init method."""
    plugin = Anthropic(api_key='test-key', models=['claude-sonnet-4'])

    # init() should return an empty list (using lazy loading)
    result = await plugin.init()
    assert result == []


@pytest.mark.asyncio
async def test_resolve_action_model() -> None:
    """Test resolve method for model."""
    plugin = Anthropic(api_key='test-key')

    # Test resolving with unprefixed name
    action = await plugin.resolve(ActionKind.MODEL, 'anthropic/claude-sonnet-4')

    assert action is not None
    assert action.name == 'anthropic/claude-sonnet-4'
    assert action.kind == ActionKind.MODEL


@patch('genkit_anthropic.plugin.AsyncAnthropic')
@pytest.mark.asyncio
async def test_anthropic_runtime_clients_are_loop_local(mock_client_ctor: MagicMock) -> None:
    """Runtime Anthropic clients are cached per event loop."""
    created: list[object] = []

    def _new_client(**kwargs: object) -> object:  # noqa: ANN003
        _ = kwargs
        client = object()
        created.append(client)
        return client

    mock_client_ctor.side_effect = _new_client
    plugin = Anthropic(api_key='test-key')

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


def test_supported_models() -> None:
    """Test that all supported models have proper metadata."""
    assert len(SUPPORTED_MODELS) == 12
    assert 'claude-3-haiku' not in SUPPORTED_MODELS
    for _name, info in SUPPORTED_MODELS.items():
        assert info.label is not None
        assert info.label.startswith('Anthropic - ')
        assert info.versions is not None
        assert len(info.versions) > 0
        assert info.supports is not None
        assert info.supports.multiturn is True
        assert info.supports.tools is True
        assert info.supports.media is True
        assert info.supports.system_role is True

    for model_name, expected_label in (
        ('claude-opus-4-7', 'Anthropic - Claude Opus 4.7'),
        ('claude-opus-4-8', 'Anthropic - Claude Opus 4.8'),
        ('claude-sonnet-4-6', 'Anthropic - Claude Sonnet 4.6'),
        ('claude-sonnet-5', 'Anthropic - Claude Sonnet 5'),
        ('claude-fable-5', 'Anthropic - Claude Fable 5'),
    ):
        info = SUPPORTED_MODELS[model_name]
        assert info.label == expected_label
        assert info.versions == [model_name]
        assert info.supports is not None
        assert info.supports.output == ['text', 'json']
        assert info.supports.constrained == Constrained.ALL


def test_mythos_excluded_but_resolvable() -> None:
    """Test claude-mythos-5 is not advertised but resolves via the generic fallback."""
    assert 'claude-mythos-5' not in SUPPORTED_MODELS
    info = get_model_info('claude-mythos-5')
    assert info.label == 'Anthropic - claude-mythos-5'
    assert info.supports is not None
    assert info.supports.output == ['text']


def test_get_model_info_known() -> None:
    """Test get_model_info returns correct info for known model."""
    info = get_model_info('claude-sonnet-4')
    assert info.label == 'Anthropic - Claude Sonnet 4'
    assert info.supports is not None
    assert info.supports.multiturn is True
    assert info.supports.tools is True


def test_get_model_info_unknown() -> None:
    """Test get_model_info returns default info for unknown model."""
    info = get_model_info('unknown-model')
    assert info.label == 'Anthropic - unknown-model'
    assert info.supports is not None
    assert info.supports.multiturn is True
    assert info.supports.tools is True


class _FakeModelPage:
    """Minimal async-iterable stand-in for the SDK's AsyncPaginator[BetaModelInfo].

    ``client.beta.models.list()`` in the real SDK is a synchronous call that
    returns an object iterated over with ``async for`` (auto-paginating). A
    plain ``MagicMock`` does not reliably support the async-iterator protocol,
    so this tiny fake implements it directly.
    """

    def __init__(self, items: list[SimpleNamespace]) -> None:
        self._items = items

    def __aiter__(self) -> '_FakeModelPage':
        self._iter = iter(self._items)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_list_actions_dynamic_union_dedup_and_fallback_info() -> None:
    """Dynamic models union with statics, dedup by id, unknown ids get generic info."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    api_items = [
        SimpleNamespace(id='claude-mythos-5'),  # unknown id -> generic fallback info
        SimpleNamespace(id='claude-sonnet-4'),  # known static id -> must appear once, curated info
        SimpleNamespace(id='claude-unknown-xyz'),  # unknown id -> generic fallback info
    ]
    mock_client.beta.models.list = MagicMock(return_value=_FakeModelPage(api_items))
    plugin._runtime_client = lambda: mock_client

    actions = await plugin.list_actions()
    names = [a.name for a in actions]

    # No duplicates.
    assert len(names) == len(set(names))

    # API ids present, in API order, first.
    assert names[0] == 'anthropic/claude-mythos-5'
    assert names[1] == 'anthropic/claude-sonnet-4'
    assert names[2] == 'anthropic/claude-unknown-xyz'

    # Static-only ids the mock didn't return are still present (appended after API ids).
    for model_id in SUPPORTED_MODELS:
        if model_id != 'claude-sonnet-4':
            assert anthropic_name(model_id) in names
    assert len(names) == 3 + len(SUPPORTED_MODELS) - 1

    # Curated info preserved for a known id returned by the API.
    sonnet_action = next(a for a in actions if a.name == 'anthropic/claude-sonnet-4')
    assert sonnet_action.metadata is not None
    assert sonnet_action.metadata['model']['label'] == 'Anthropic - Claude Sonnet 4'

    # Unknown ids get the generic fallback info, not curated.
    mythos_action = next(a for a in actions if a.name == 'anthropic/claude-mythos-5')
    assert mythos_action.metadata is not None
    assert mythos_action.metadata['model']['label'] == 'Anthropic - claude-mythos-5'
    assert mythos_action.metadata['model']['supports']['output'] == ['text']


@pytest.mark.asyncio
async def test_list_actions_falls_back_to_static_on_error_uncached() -> None:
    """API errors fall back to the static list without caching the failure."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    mock_client.beta.models.list = MagicMock(side_effect=RuntimeError('boom'))
    plugin._runtime_client = lambda: mock_client

    actions = await plugin.list_actions()
    names = {a.name for a in actions}
    assert names == {anthropic_name(model_id) for model_id in SUPPORTED_MODELS}

    _ = await plugin.list_actions()
    assert mock_client.beta.models.list.call_count == 2  # not cached on failure


@pytest.mark.asyncio
async def test_list_actions_caches_on_success() -> None:
    """A successful API fetch is memoized for the plugin's lifetime."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    mock_client.beta.models.list = MagicMock(return_value=_FakeModelPage([SimpleNamespace(id='claude-sonnet-4')]))
    plugin._runtime_client = lambda: mock_client

    first = await plugin.list_actions()
    second = await plugin.list_actions()
    assert mock_client.beta.models.list.call_count == 1
    assert first is second


@pytest.mark.asyncio
async def test_list_actions_skips_empty_model_ids() -> None:
    """Models with an empty or missing id are dropped, not turned into bogus actions."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    api_items = [
        SimpleNamespace(id='claude-sonnet-4'),
        SimpleNamespace(id=''),  # empty id -> dropped
        SimpleNamespace(id=None),  # missing id -> dropped
    ]
    mock_client.beta.models.list = MagicMock(return_value=_FakeModelPage(api_items))
    plugin._runtime_client = lambda: mock_client

    actions = await plugin.list_actions()
    names = [a.name for a in actions]

    # The valid id is present; the empty/missing ones produce no action.
    assert 'anthropic/claude-sonnet-4' in names
    assert 'anthropic/' not in names
    assert 'anthropic/None' not in names
    # Only the static set is advertised (the one API id overlaps it).
    assert len(names) == len(SUPPORTED_MODELS)


@pytest.mark.asyncio
async def test_list_actions_empty_api_response_returns_and_caches_statics() -> None:
    """A successful but empty API response still advertises (and caches) the static set."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    mock_client.beta.models.list = MagicMock(return_value=_FakeModelPage([]))
    plugin._runtime_client = lambda: mock_client

    actions = await plugin.list_actions()
    names = {a.name for a in actions}
    assert names == {anthropic_name(model_id) for model_id in SUPPORTED_MODELS}

    # Unlike the error path, an empty-but-successful response is cached.
    _ = await plugin.list_actions()
    assert mock_client.beta.models.list.call_count == 1


_ANTHROPIC_CONFIG_KEYS = {
    'apiKey',
    'apiVersion',
    'betas',
    'maxOutputTokens',
    'tool_choice',
    'metadata',
    'thinking',
    'output_config',
}


def _custom_options(action_metadata: object) -> dict[str, Any]:
    metadata = cast(dict[str, Any], action_metadata)
    model_metadata = cast(dict[str, Any], metadata['model'])
    return cast(dict[str, Any], model_metadata['customOptions'])


@pytest.mark.asyncio
async def test_resolve_advertises_anthropic_config() -> None:
    """resolve() metadata customOptions reflects the typed AnthropicConfig."""
    plugin = Anthropic(api_key='test-key')

    action = await plugin.resolve(ActionKind.MODEL, 'anthropic/claude-sonnet-4')

    assert action is not None
    custom_options = _custom_options(action.metadata)
    properties = set(custom_options['properties'].keys())
    assert _ANTHROPIC_CONFIG_KEYS <= properties


@pytest.mark.asyncio
async def test_list_actions_advertises_anthropic_config() -> None:
    """list_actions() customOptions reflects the typed AnthropicConfig."""
    plugin = Anthropic(api_key='test-key')
    mock_client = MagicMock()
    mock_client.beta.models.list = MagicMock(side_effect=RuntimeError('offline'))
    plugin._runtime_client = lambda: mock_client

    actions = await plugin.list_actions()

    assert actions
    for action in actions:
        custom_options = _custom_options(action.metadata)
        properties = set(custom_options['properties'].keys())
        assert _ANTHROPIC_CONFIG_KEYS <= properties


def _create_sample_request() -> ModelRequest:
    """Create a sample generation request for testing."""
    return ModelRequest(
        messages=[
            Message(
                role=Role.USER,
                content=[Part(root=TextPart(text='Hello, how are you?'))],
            )
        ],
        config=ModelConfig(),
        tools=[
            ToolDefinition(
                name='get_weather',
                description='Get weather for a location',
                input_schema={
                    'type': 'object',
                    'properties': {'location': {'type': 'string', 'description': 'Location name'}},
                    'required': ['location'],
                },
            )
        ],
    )
