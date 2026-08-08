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

from genkit import MediaPart, Message, ModelConfig, ModelRequest, Part, Role, TextPart
from genkit.plugin_api import ActionKind


def _xai_module() -> ModuleType:
    return importlib.import_module('genkit_openai.xai')


def _request(prompt: str = 'Hello') -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part(root=TextPart(text=prompt))])],
    )


async def _get_runtime_client(plugin: object) -> object:
    return plugin._runtime_client()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_init_registers_exact_canonical_models_with_provider_metadata() -> None:
    xai = _xai_module()
    actions = await xai.XAI(api_key='test-key').init()

    assert [action.name for action in actions] == [
        'xai/grok-3',
        'xai/grok-3-fast',
        'xai/grok-3-mini',
        'xai/grok-3-mini-fast',
        'xai/grok-2-vision-1212',
        'xai/grok-2-image-1212',
    ]
    assert all(action.kind == ActionKind.MODEL for action in actions)

    standard_supports = {
        'multiturn': True,
        'media': False,
        'tools': True,
        'systemRole': True,
        'output': ['text', 'json'],
    }
    for action in actions[:4]:
        assert action.metadata['model']['supports'] == standard_supports

    assert actions[4].metadata['model']['supports'] == {
        'multiturn': False,
        'media': True,
        'tools': True,
        'systemRole': False,
        'output': ['text', 'json'],
    }
    assert actions[5].metadata['model']['supports'] == {
        'multiturn': False,
        'media': False,
        'tools': False,
        'systemRole': False,
        'output': ['media'],
    }


@pytest.mark.asyncio
async def test_model_metadata_exposes_xai_specific_config_schemas() -> None:
    actions = await _xai_module().XAI(api_key='test-key').init()

    chat_properties = actions[0].metadata['model']['customOptions']['properties']
    deferred_schema = next(schema for schema in chat_properties['deferred']['anyOf'] if schema.get('type') == 'boolean')
    reasoning_schema = next(schema for schema in chat_properties['reasoningEffort']['anyOf'] if 'enum' in schema)
    web_search_schema = next(
        schema for schema in chat_properties['webSearchOptions']['anyOf'] if schema.get('type') == 'object'
    )
    assert deferred_schema['type'] == 'boolean'
    assert set(reasoning_schema['enum']) == {'low', 'medium', 'high'}
    assert web_search_schema['type'] == 'object'

    image_properties = actions[-1].metadata['model']['customOptions']['properties']
    assert {'size', 'style', 'user', 'n', 'quality', 'responseFormat'} <= set(image_properties)
    assert image_properties['n']['default'] == 1
    assert image_properties['n']['minimum'] == 1
    assert image_properties['n']['maximum'] == 10
    assert image_properties['responseFormat']['default'] == 'b64_json'


@pytest.mark.asyncio
async def test_list_actions_is_static_and_offline() -> None:
    xai = _xai_module()
    plugin = xai.XAI(api_key='test-key')
    plugin._runtime_client = MagicMock(side_effect=AssertionError('list_actions must not create a network client'))

    actions = await plugin.list_actions()

    assert [action.name for action in actions] == [
        'xai/grok-3',
        'xai/grok-3-fast',
        'xai/grok-3-mini',
        'xai/grok-3-mini-fast',
        'xai/grok-2-vision-1212',
        'xai/grok-2-image-1212',
    ]


@pytest.mark.asyncio
async def test_resolve_routes_chat_and_image_models_only_as_model_actions() -> None:
    plugin = _xai_module().XAI(api_key='test-key')

    chat = await plugin.resolve(ActionKind.MODEL, 'xai/grok-custom')
    image = await plugin.resolve(ActionKind.MODEL, 'xai/grok-custom-image')
    vision = await plugin.resolve(ActionKind.MODEL, 'xai/grok-custom-vision')

    assert chat is not None
    assert chat.name == 'xai/grok-custom'
    assert image is not None
    assert image.name == 'xai/grok-custom-image'
    assert vision is not None
    assert vision.metadata['model']['supports']['media'] is True
    assert await plugin.resolve(ActionKind.EMBEDDER, 'xai/grok-3') is None
    assert await plugin.resolve(ActionKind.MODEL, 'openai/gpt-4o') is None


def test_client_uses_canonical_base_url_and_explicit_key() -> None:
    xai = _xai_module()
    with patch.object(xai, 'AsyncOpenAI') as client_constructor:
        plugin = xai.XAI(api_key='explicit-key', timeout=30)
        asyncio.run(_get_runtime_client(plugin))

    client_constructor.assert_called_once_with(
        api_key='explicit-key',
        base_url='https://api.x.ai/v1',
        timeout=30,
    )


def test_client_uses_xai_api_key_environment_variable() -> None:
    xai = _xai_module()
    with (
        patch.dict('os.environ', {'XAI_API_KEY': 'env-key'}, clear=True),
        patch.object(xai, 'AsyncOpenAI') as client_constructor,
    ):
        plugin = xai.XAI()
        asyncio.run(_get_runtime_client(plugin))

    client_constructor.assert_called_once_with(
        api_key='env-key',
        base_url='https://api.x.ai/v1',
    )


def test_missing_api_key_raises() -> None:
    xai = _xai_module()
    with patch.dict('os.environ', {}, clear=True), pytest.raises(ValueError, match='XAI_API_KEY'):
        xai.XAI()


def test_xai_config_rejects_unsupported_reasoning_effort_and_image_count() -> None:
    xai = _xai_module()

    assert xai.XAIConfig(reasoning_effort='high').reasoning_effort == 'high'
    with pytest.raises(ValidationError):
        xai.XAIConfig(reasoning_effort='minimal')
    assert xai.XAIImageConfig(n=1).n == 1
    assert xai.XAIImageConfig(n=10).n == 10
    assert isinstance(xai.XAIImageConfig(), ModelConfig)
    assert xai.XAIImageConfig(max_output_tokens=100).max_output_tokens == 100
    with pytest.raises(ValidationError):
        xai.XAIImageConfig(n=0)
    with pytest.raises(ValidationError):
        xai.XAIImageConfig(n=11)
    with pytest.raises(ValidationError, match='extra_forbidden'):
        xai.XAIImageConfig.model_validate({'unknownImageOption': True})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('extra_body_config', 'expected_extra_body'),
    [
        ({'extra_body': {'source': 'snake'}}, {'source': 'snake', 'deferred': True}),
        ({'extraBody': {'source': 'camel'}}, {'source': 'camel', 'deferred': True}),
        (
            {'extra_body': {'source': 'snake'}, 'extraBody': {'source': 'camel'}},
            {'source': 'snake', 'deferred': True},
        ),
    ],
)
async def test_chat_action_maps_xai_request_extensions_to_api_fields(
    extra_body_config: dict[str, object],
    expected_extra_body: dict[str, object],
) -> None:
    xai = _xai_module()
    mock_message = MagicMock(content='Hello back', reasoning_content=None, role='assistant', tool_calls=None)
    mock_response = MagicMock(choices=[MagicMock(message=mock_message)])
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    plugin = xai.XAI(api_key='test-key')
    plugin._runtime_client = lambda: mock_client
    action = await plugin.resolve(ActionKind.MODEL, 'xai/grok-3')
    assert action is not None

    request = _request()
    request.config = xai.XAIConfig.model_validate({
        'deferred': True,
        'reasoningEffort': 'high',
        'webSearchOptions': {'search_context_size': 'high'},
        **extra_body_config,
    })
    response = (await action.run(request)).response

    assert response.text == 'Hello back'
    await_args = mock_client.chat.completions.create.await_args
    assert await_args is not None
    assert await_args.kwargs['model'] == 'grok-3'
    assert await_args.kwargs['reasoning_effort'] == 'high'
    assert await_args.kwargs['web_search_options'] == {'search_context_size': 'high'}
    assert await_args.kwargs['extra_body'] == expected_extra_body
    assert 'extraBody' not in await_args.kwargs
    assert 'deferred' not in await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize('extra_body_key', ['extra_body', 'extraBody'])
async def test_chat_model_rejects_non_dictionary_extra_body(extra_body_key: str) -> None:
    xai = _xai_module()
    request = _request()
    request.config = xai.XAIConfig.model_validate({extra_body_key: 'invalid'})
    model = xai._XAIChatModel('grok-3', MagicMock())

    with pytest.raises(ValueError, match='extra_body must be a dictionary'):
        await model._get_openai_request_config(request)


def test_chat_model_normalizes_dictionary_config_as_xai_config() -> None:
    xai = _xai_module()
    config = xai._XAIChatModel.normalize_config({'reasoningEffort': 'high', 'deferred': True})

    assert isinstance(config, xai.XAIConfig)
    assert config.reasoning_effort == 'high'
    assert config.deferred is True

    with pytest.raises(ValidationError, match='reasoningEffort'):
        xai._XAIChatModel.normalize_config({'reasoningEffort': 'minimal'})


@pytest.mark.asyncio
@pytest.mark.parametrize('response_format_key', ['responseFormat', 'response_format'])
async def test_image_action_uses_shared_image_runtime(response_format_key: str) -> None:
    xai = _xai_module()
    mock_image = MagicMock(url='https://example.com/grok.png', b64_json=None)
    mock_client = MagicMock()
    mock_client.images.generate = AsyncMock(return_value=MagicMock(data=[mock_image]))

    plugin = xai.XAI(api_key='test-key')
    plugin._runtime_client = lambda: mock_client
    action = await plugin.resolve(ActionKind.MODEL, 'xai/grok-2-image-1212')
    assert action is not None

    request = _request('Draw a rocket')
    request.config = xai.XAIImageConfig.model_validate(
        {'size': '1024x1024', 'n': 1, response_format_key: 'url'},
    )
    response = (await action.run(request)).response

    await_args = mock_client.images.generate.await_args
    assert await_args is not None
    assert await_args.kwargs == {
        'model': 'grok-2-image-1212',
        'prompt': 'Draw a rocket',
        'response_format': 'url',
        'size': '1024x1024',
        'n': 1,
    }
    assert response.message is not None
    media = response.message.content[0].root
    assert isinstance(media, MediaPart)
    assert str(media.media.url) == 'https://example.com/grok.png'


def test_public_exports_and_model_helper() -> None:
    package = importlib.import_module('genkit_openai')

    assert package.xai_model('grok-3') == 'xai/grok-3'
    assert package.XAI.name == 'xai'
    assert package.XAIConfig is not None
    assert package.XAIImageConfig is not None
    assert {'XAI', 'XAIConfig', 'XAIImageConfig', 'xai_model'} <= set(package.__all__)
