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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_anthropic import AnthropicConfig
from genkit_vertexai.model_garden import ModelGarden, ModelGardenPlugin
from genkit_vertexai.model_garden.anthropic import AnthropicModelGarden
from genkit_vertexai.model_garden.model_garden import ModelGardenModel

from genkit import Genkit, Message, Part, Role


@pytest.fixture
@patch('genkit_vertexai.model_garden.model_garden.OpenAIClient')
def model_garden_instance(client: MagicMock) -> ModelGardenModel:
    """Model Garden fixture."""
    return ModelGardenModel(model='test', location='us-central1', project_id='project')


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
                    'context': None,
                    'long_running': False,
                    'multiturn': True,
                    'media': False,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json_mode',
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
                    'context': None,
                    'long_running': None,
                    'multiturn': True,
                    'media': True,
                    'tools': True,
                    'system_role': True,
                    'output': [
                        'json_mode',
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


@pytest.mark.asyncio
async def test_generate_model_garden_claude_drops_part_with_only_signature_key() -> None:
    """A Model Garden Claude thinking part carrying only ``signature`` is left out of the request."""
    ai = Genkit(plugins=[ModelGarden(project_id='project', location='us-east5')])
    history = [
        Message(role=Role.USER, content=[Part.from_text('what is 17 * 23?')]),
        Message(
            role=Role.MODEL,
            content=[Part.from_reasoning('because', metadata={'signature': 'sig'}), Part.from_text('391')],
        ),
    ]
    reply = MagicMock()
    reply.content = [MagicMock(type='text', text='ok')]
    reply.usage = MagicMock(input_tokens=1, output_tokens=1)
    reply.stop_reason = 'end_turn'

    with patch('genkit_vertexai.model_garden.anthropic.AsyncAnthropicVertex') as client_ctor:
        client = client_ctor.return_value
        client.messages.create = AsyncMock(return_value=reply)
        client.beta.messages.create = AsyncMock(return_value=reply)
        response = await ai.generate(
            model='modelgarden/anthropic/claude-sonnet-4@20250514', messages=history, prompt='now add 100'
        )

    assert response.text == 'ok'
    create = client.messages.create if client.messages.create.called else client.beta.messages.create
    assert create.call_args.kwargs['messages'][1] == {'role': 'assistant', 'content': [{'type': 'text', 'text': '391'}]}


def test_anthropic_model_garden_does_not_advertise_api_key() -> None:
    """Vertex authenticates with Google credentials, so apiKey is not offered."""
    properties = AnthropicModelGarden.get_config_schema().model_json_schema()['properties']
    assert 'apiKey' not in properties
    assert 'apiVersion' in properties
