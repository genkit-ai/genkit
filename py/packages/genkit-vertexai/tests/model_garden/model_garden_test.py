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

from genkit import ActionRunContext, GenkitError, Message, Part, Role
from genkit._ai._formats import built_in_formats
from genkit.model import ModelRequest, OutputConfig
from genkit.plugin_api import ActionKind


def test_catalog_output_names_are_known_formats() -> None:
    """supports.output lists Genkit output formats, not OpenAI request options like json_mode."""
    known = {f.name for f in built_in_formats}
    entries = {name: info.supports for name, info in SUPPORTED_OPENAI_COMPAT_MODELS.items()}
    entries['<default>'] = DEFAULT_SUPPORTS
    for name, supports in entries.items():
        unknown = set((supports.output if supports else None) or []) - known
        assert not unknown, f'{name}: {sorted(unknown)}'


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
@pytest.mark.parametrize(
    'model_name', ['modelgarden/meta/llama-3.2-90b-vision-instruct-maas', 'modelgarden/anthropic/claude-sonnet-4']
)
async def test_resolve_without_project_is_failed_precondition(model_name: str) -> None:
    """No project configured is a local setup problem, not a bad request."""
    with patch.dict('os.environ', {}, clear=True):
        plugin = ModelGarden(location='us-central1')

    with pytest.raises(GenkitError, match='project_id must be provided') as raised:
        await plugin.resolve(ActionKind.MODEL, model_name)

    assert raised.value.status == 'FAILED_PRECONDITION'


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
