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

import asyncio
import importlib
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from genkit import GenkitError, Message, ModelRequest, Part, ReasoningPart, Role, TextPart
from genkit.plugin_api import ActionKind


def _deepseek_module() -> ModuleType:
    return importlib.import_module('genkit_openai.deepseek')


def _request() -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text='Hello'))])],
    )


@pytest.mark.asyncio
async def test_init_registers_exactly_the_two_canonical_models() -> None:
    deepseek = _deepseek_module()
    plugin = deepseek.DeepSeek(api_key='test-key')

    actions = await plugin.init()

    assert [action.name for action in actions] == [
        'deepseek/deepseek-chat',
        'deepseek/deepseek-reasoner',
    ]
    assert all(action.kind == ActionKind.MODEL for action in actions)
    for action in actions:
        model_metadata = action.metadata['model']
        assert model_metadata['supports'] == {
            'multiturn': True,
            'media': False,
            'tools': True,
            'systemRole': True,
            'output': ['text', 'json'],
        }
        max_tokens = model_metadata['customOptions']['properties']['maxTokens']
        integer_schema = next(schema for schema in max_tokens['anyOf'] if schema.get('type') == 'integer')
        assert integer_schema['minimum'] == 1
        assert 'maximum' not in integer_schema


@pytest.mark.asyncio
async def test_list_actions_is_static_and_offline() -> None:
    deepseek = _deepseek_module()
    plugin = deepseek.DeepSeek(api_key='test-key')
    plugin._runtime_client = MagicMock(side_effect=AssertionError('list_actions must not create a network client'))

    actions = await plugin.list_actions()

    assert [action.name for action in actions] == [
        'deepseek/deepseek-chat',
        'deepseek/deepseek-reasoner',
    ]


@pytest.mark.asyncio
async def test_resolve_only_handles_model_actions() -> None:
    deepseek = _deepseek_module()
    plugin = deepseek.DeepSeek(api_key='test-key')

    action = await plugin.resolve(ActionKind.MODEL, 'deepseek/deepseek-reasoner')

    assert action is not None
    assert action.name == 'deepseek/deepseek-reasoner'
    assert await plugin.resolve(ActionKind.EMBEDDER, 'deepseek/deepseek-reasoner') is None
    assert await plugin.resolve(ActionKind.MODEL, 'openai/gpt-4o') is None

    dynamic = await plugin.resolve(ActionKind.MODEL, 'deepseek/custom-model')
    assert dynamic is not None
    assert dynamic.name == 'deepseek/custom-model'


def test_client_uses_canonical_base_url_and_explicit_key() -> None:
    deepseek = _deepseek_module()
    with patch.object(deepseek, 'AsyncOpenAI') as client_constructor:
        plugin = deepseek.DeepSeek(api_key='explicit-key', timeout=30)

        asyncio.run(_get_runtime_client(plugin))

    client_constructor.assert_called_once_with(
        api_key='explicit-key',
        base_url='https://api.deepseek.com',
        timeout=30,
    )


def test_client_uses_deepseek_api_key_environment_variable() -> None:
    deepseek = _deepseek_module()
    with (
        patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'env-key'}, clear=True),
        patch.object(deepseek, 'AsyncOpenAI') as client_constructor,
    ):
        plugin = deepseek.DeepSeek()

        asyncio.run(_get_runtime_client(plugin))

    client_constructor.assert_called_once_with(
        api_key='env-key',
        base_url='https://api.deepseek.com',
    )


def test_missing_api_key_raises() -> None:
    deepseek = _deepseek_module()
    with patch.dict('os.environ', {}, clear=True), pytest.raises(ValueError, match='DEEPSEEK_API_KEY'):
        deepseek.DeepSeek()


async def _get_runtime_client(plugin: object) -> object:
    return plugin._runtime_client()  # type: ignore[attr-defined]


def test_deepseek_config_constrains_max_tokens() -> None:
    deepseek = _deepseek_module()

    assert deepseek.DeepSeekConfig(max_tokens=1).max_tokens == 1
    assert deepseek.DeepSeekConfig(max_tokens=8192).max_tokens == 8192
    assert deepseek.DeepSeekConfig(max_tokens=8193).max_tokens == 8193
    with pytest.raises(ValidationError):
        deepseek.DeepSeekConfig(max_tokens=0)


@pytest.mark.asyncio
async def test_action_normalizes_camel_case_deepseek_config() -> None:
    deepseek = _deepseek_module()
    mock_message = MagicMock(
        content='Configured answer',
        reasoning_content=None,
        role='assistant',
        tool_calls=None,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=mock_message)]),
    )

    plugin = deepseek.DeepSeek(api_key='test-key')
    plugin._runtime_client = lambda: mock_client
    action = await plugin.resolve(ActionKind.MODEL, 'deepseek/deepseek-chat')
    assert action is not None
    request = ModelRequest.model_validate({
        **_request().model_dump(by_alias=True),
        'config': {'maxTokens': 256, 'model': 'deepseek-chat-version'},
    })

    await action.run(request)

    await_args = mock_client.chat.completions.create.await_args
    assert await_args is not None
    assert await_args.kwargs['model'] == 'deepseek-chat-version'
    assert await_args.kwargs['max_tokens'] == 256
    assert 'maxTokens' not in await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('config', 'expected_max_tokens'),
    [
        ({'max_completion_tokens': 512}, 512),
        ({'maxCompletionTokens': 768}, 768),
        ({'max_tokens': 256, 'maxCompletionTokens': 512}, 256),
        ({'maxTokens': 384, 'max_completion_tokens': 768}, 384),
    ],
)
async def test_action_maps_max_completion_tokens_to_max_tokens(
    config: dict[str, int],
    expected_max_tokens: int,
) -> None:
    deepseek = _deepseek_module()
    mock_message = MagicMock(
        content='Configured answer',
        reasoning_content=None,
        role='assistant',
        tool_calls=None,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=mock_message)]),
    )

    plugin = deepseek.DeepSeek(api_key='test-key')
    plugin._runtime_client = lambda: mock_client
    action = await plugin.resolve(ActionKind.MODEL, 'deepseek/deepseek-chat')
    assert action is not None
    request = ModelRequest.model_validate({
        **_request().model_dump(by_alias=True),
        'config': config,
    })

    await action.run(request)

    await_args = mock_client.chat.completions.create.await_args
    assert await_args is not None
    assert await_args.kwargs['max_tokens'] == expected_max_tokens
    assert 'max_completion_tokens' not in await_args.kwargs
    assert 'maxCompletionTokens' not in await_args.kwargs


@pytest.mark.asyncio
async def test_action_validates_mapped_max_completion_tokens() -> None:
    deepseek = _deepseek_module()
    plugin = deepseek.DeepSeek(api_key='test-key')
    plugin._runtime_client = MagicMock(side_effect=AssertionError('invalid config must be rejected before client use'))
    action = await plugin.resolve(ActionKind.MODEL, 'deepseek/deepseek-chat')
    assert action is not None
    request = ModelRequest.model_validate({
        **_request().model_dump(by_alias=True),
        'config': {'maxCompletionTokens': 0},
    })

    with pytest.raises(GenkitError) as exc_info:
        await action.run(request)

    assert isinstance(exc_info.value.cause, ValidationError)


@pytest.mark.asyncio
async def test_reasoner_action_uses_shared_runtime_reasoning_conversion() -> None:
    deepseek = _deepseek_module()
    mock_message = MagicMock(
        content='Final answer',
        reasoning_content='Step by step',
        role='assistant',
        tool_calls=None,
    )
    mock_response = MagicMock(choices=[MagicMock(message=mock_message)])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    plugin = deepseek.DeepSeek(api_key='test-key')
    plugin._runtime_client = lambda: mock_client
    action = await plugin.resolve(ActionKind.MODEL, 'deepseek/deepseek-reasoner')
    assert action is not None

    request = _request()
    request.config = deepseek.DeepSeekConfig(
        max_tokens=256,
        model='deepseek-reasoner-version',
    )
    response = (await action.run(request)).response

    assert response.message is not None
    reasoning = response.message.content[0].root
    assert isinstance(reasoning, ReasoningPart)
    assert reasoning.reasoning == 'Step by step'
    assert response.message.content[1].root.text == 'Final answer'
    await_args = mock_client.chat.completions.create.await_args
    assert await_args is not None
    assert await_args.kwargs['model'] == 'deepseek-reasoner-version'
    assert await_args.kwargs['max_tokens'] == 256
    assert 'maxTokens' not in await_args.kwargs


def test_public_exports_and_model_helper() -> None:
    package = importlib.import_module('genkit_openai')

    assert package.deepseek_model('deepseek-chat') == 'deepseek/deepseek-chat'
    assert package.DeepSeek.name == 'deepseek'
    assert package.DeepSeekConfig is not None
    assert {'DeepSeek', 'DeepSeekConfig', 'deepseek_model'} <= set(package.__all__)
