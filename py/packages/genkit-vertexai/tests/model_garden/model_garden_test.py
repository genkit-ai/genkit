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

"""Unittests for VertexAI Model Garden Models."""

import warnings
from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from genkit_anthropic import AnthropicConfig
from genkit_openai import OpenAIConfig
from genkit_vertexai.model_garden import ModelGarden, ModelGardenPlugin
from genkit_vertexai.model_garden._model_info import DEFAULT_SUPPORTS, SUPPORTED_OPENAI_COMPAT_MODELS
from genkit_vertexai.model_garden.anthropic import AnthropicModelGarden
from genkit_vertexai.model_garden.model_garden import ModelGardenModel
from openai.types.chat import ChatCompletion

from genkit import ActionRunContext, Genkit, Message, Part, Role
from genkit._ai._formats import built_in_formats
from genkit.model import ModelRequest, OutputConfig


def test_catalog_output_names_are_known_formats() -> None:
    """supports.output lists Genkit output formats, not OpenAI request options like json_mode."""
    known = {f.name for f in built_in_formats}
    entries = {name: info.supports for name, info in SUPPORTED_OPENAI_COMPAT_MODELS.items()}
    entries['<default>'] = DEFAULT_SUPPORTS
    for name, supports in entries.items():
        unknown = set((supports.output if supports else None) or []) - known
        assert not unknown, f'{name}: {sorted(unknown)}'


@pytest.fixture
def model_garden_instance() -> ModelGardenModel:
    """Model Garden fixture."""
    return ModelGardenModel(model='test', location='us-central1', project_id='project')


def _chat_completion(text: str) -> ChatCompletion:
    return ChatCompletion.model_validate({
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'meta/llama-3.1-405b-instruct-maas',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': text},
            }
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    })


@pytest.mark.parametrize(
    'model_name, expected',
    [
        (
            'meta/llama-3.1-405b-instruct-maas',
            {
                'name': 'ModelGarden - Meta - llama-3.1',
                'supports': {
                    'constrained': None,
                    'content_type': None,
                    'long_running': False,
                    'multiturn': True,
                    'media': False,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json',
                        'text',
                    ],
                    'tool_choice': None,
                },
            },
        ),
        (
            'meta/lazaro-model-pro-max',
            {
                'name': 'ModelGarden - meta/lazaro-model-pro-max',
                'supports': {
                    'constrained': None,
                    'content_type': None,
                    'long_running': None,
                    'multiturn': True,
                    'media': True,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json',
                        'text',
                    ],
                    'tool_choice': None,
                },
            },
        ),
    ],
)
def test_get_model_info(model_name: str, expected: dict[str, Any], model_garden_instance: ModelGardenModel) -> None:
    """Unittest for get_model_info."""
    model_garden_instance.name = model_name

    result = model_garden_instance.get_model_info()

    assert result == expected


def test_model_garden_plugin_deprecated_alias() -> None:
    """ModelGardenPlugin warns and delegates to ModelGarden."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', DeprecationWarning)
        plugin = ModelGardenPlugin(project_id='my-project', location='us-central1')

    assert len(caught) == 1
    assert 'ModelGardenPlugin is deprecated' in str(caught[0].message)
    assert isinstance(plugin, ModelGarden)


def test_anthropic_model_garden_uses_anthropic_config_schema() -> None:
    """Anthropic Model Garden advertises the schema enforced by its handler."""
    schema = AnthropicModelGarden.get_config_schema()
    assert issubclass(schema, AnthropicConfig)


def test_anthropic_model_garden_does_not_advertise_api_key() -> None:
    """Vertex authenticates with Google credentials, so apiKey is not offered."""
    properties = AnthropicModelGarden.get_config_schema().model_json_schema()['properties']
    assert 'apiKey' not in properties
    assert 'apiVersion' in properties


@pytest.mark.asyncio
async def test_model_garden_llama_json_request_sends_json_object() -> None:
    """ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', output_format='json') sends json_object."""
    captured: dict[str, Any] = {}
    client = MagicMock()

    async def create(**kwargs: Any) -> ChatCompletion:
        captured.update(kwargs)
        return ChatCompletion.construct(
            id='1',
            object='chat.completion',
            created=1,
            model='llama',
            choices=[
                {
                    'index': 0,
                    'message': {'role': 'assistant', 'content': '{"a": 1}'},
                    'finish_reason': 'stop',
                }
            ],
        )

    client.chat.completions.create = AsyncMock(side_effect=create)
    garden = ModelGardenModel(
        model='meta/llama-3.1-405b-instruct-maas',
        location='us-central1',
        project_id='p',
    )
    ctx = MagicMock(spec=ActionRunContext)
    type(ctx).is_streaming = PropertyMock(return_value=False)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('give me json')])],
        output=OutputConfig(format='json'),
        config=OpenAIConfig(),
    )

    with patch.object(garden, 'create_client', AsyncMock(return_value=client)):
        await garden.to_openai_compatible_model()(request, ctx)

    assert captured['response_format'] == {'type': 'json_object'}


@pytest.mark.asyncio
async def test_model_garden_openai_compatible_model_resolves_and_generates() -> None:
    """ai.generate on a catalog Llama model sends the prompt to the OpenAI client and returns its reply."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_chat_completion('hello from llama'))

    with patch(
        'genkit_vertexai.model_garden.model_garden.ModelGardenModel.create_client',
        new=AsyncMock(return_value=client),
    ):
        ai = Genkit(plugins=[ModelGarden(project_id='my-project', location='us-central1')])
        response = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')

    assert response.text == 'hello from llama'
    client.chat.completions.create.assert_awaited_once()
    sent = client.chat.completions.create.call_args.kwargs
    assert sent['model'] == 'meta/llama-3.1-405b-instruct-maas'
    assert sent['messages'] == [{'role': 'user', 'content': 'hi'}]


class _FakeCredentials:
    """Google credentials stand-in with a token that can expire."""

    def __init__(self) -> None:
        self.token = 'tok-1'
        self.valid = True
        self.refresh_count = 0

    def refresh(self, _request: object) -> None:
        self.refresh_count += 1
        if self.refresh_count > 1:
            self.token = 'tok-2'
        self.valid = True


@pytest.mark.asyncio
async def test_model_garden_repeated_generate_refreshes_credentials_once() -> None:
    """Three ai.generate calls on a Llama model refresh Google credentials once and reuse one OpenAI client."""
    creds = _FakeCredentials()
    clients: list[MagicMock] = []

    def make_client(**kwargs: object) -> MagicMock:
        client = MagicMock()
        client.api_key = kwargs.get('api_key')
        client.chat.completions.create = AsyncMock(return_value=_chat_completion('ok'))
        clients.append(client)
        return client

    with (
        patch('genkit_vertexai.model_garden.client.auth.default', return_value=(creds, 'my-project')),
        patch('genkit_vertexai.model_garden.client.google.auth.transport.requests.Request'),
        patch('genkit_vertexai.model_garden.client._AsyncOpenAI', side_effect=make_client),
    ):
        ai = Genkit(plugins=[ModelGarden(project_id='my-project', location='us-central1')])
        for _ in range(3):
            response = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
            assert response.text == 'ok'

    assert creds.refresh_count == 1
    assert len(clients) == 1


@pytest.mark.asyncio
async def test_model_garden_generate_after_token_expiry_sends_fresh_token() -> None:
    """After cached credentials expire, the next generate refreshes and sends the new token."""
    creds = _FakeCredentials()
    clients: list[MagicMock] = []

    def make_client(**kwargs: object) -> MagicMock:
        client = MagicMock()
        client.api_key = kwargs.get('api_key')
        client.chat.completions.create = AsyncMock(return_value=_chat_completion('ok'))
        clients.append(client)
        return client

    with (
        patch('genkit_vertexai.model_garden.client.auth.default', return_value=(creds, 'my-project')),
        patch('genkit_vertexai.model_garden.client.google.auth.transport.requests.Request'),
        patch('genkit_vertexai.model_garden.client._AsyncOpenAI', side_effect=make_client),
    ):
        ai = Genkit(plugins=[ModelGarden(project_id='my-project', location='us-central1')])
        first = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
        assert first.text == 'ok'
        assert clients[0].api_key == 'tok-1'

        creds.valid = False
        second = await ai.generate(model='modelgarden/meta/llama-3.1-405b-instruct-maas', prompt='hi')
        assert second.text == 'ok'

    assert creds.refresh_count == 2
    assert len(clients) == 1
    assert clients[0].api_key == 'tok-2'
