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

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropicVertex
from genkit_anthropic.config import AnthropicConfig
from genkit_vertexai.model_garden import ModelGarden
from genkit_vertexai.model_garden.anthropic import AnthropicModelGarden
from genkit_vertexai.model_garden.model_garden import ModelGardenModel

from genkit import Genkit


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


_CLAUDE = 'modelgarden/anthropic/claude-sonnet-4-5'


async def _project_sent_to_vertex(plugin: ModelGarden) -> object:
    """Run one Model Garden generate and return the project its Vertex client was built for."""
    reply = MagicMock()
    reply.content = [MagicMock(type='text', text='hello')]
    reply.usage = MagicMock(input_tokens=1, output_tokens=1)
    reply.stop_reason = 'end_turn'
    client = MagicMock(spec=AsyncAnthropicVertex)
    client.messages = MagicMock()
    client.beta = MagicMock()
    client.messages.create = AsyncMock(return_value=reply)
    client.beta.messages.create = AsyncMock(return_value=reply)
    with patch('genkit_vertexai.model_garden.anthropic.AsyncAnthropicVertex', return_value=client) as vertex:
        ai = Genkit(plugins=[plugin])
        response = await ai.generate(model=_CLAUDE, prompt='hi')
    assert response.text == 'hello'
    return vertex.call_args.kwargs['project_id']


@pytest.mark.asyncio
async def test_generate_model_garden_with_project_sends_it_to_vertex() -> None:
    """ModelGarden(project='p') sends generate calls to project p."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': 'env-proj'}):
        assert await _project_sent_to_vertex(ModelGarden(project='p', location='us-central1')) == 'p'


@pytest.mark.asyncio
async def test_generate_model_garden_with_project_id_still_sends_it_to_vertex() -> None:
    """ModelGarden(project_id='p') keeps sending generate calls to project p."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': 'env-proj'}):
        assert await _project_sent_to_vertex(ModelGarden(project_id='p', location='us-central1')) == 'p'


@pytest.mark.asyncio
async def test_generate_model_garden_with_project_and_project_id_same_value_works() -> None:
    """ModelGarden(project='p', project_id='p') is fine and uses p."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': 'env-proj'}):
        plugin = ModelGarden(project='p', project_id='p', location='us-central1')
        assert await _project_sent_to_vertex(plugin) == 'p'


def test_model_garden_with_project_and_project_id_different_values_raises() -> None:
    """ModelGarden(project='a', project_id='b') raises ValueError naming both."""
    with pytest.raises(ValueError) as exc_info:
        ModelGarden(project='a', project_id='b')
    assert "project='a'" in str(exc_info.value)
    assert "project_id='b'" in str(exc_info.value)
    assert 'same setting' in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_model_garden_with_no_project_uses_environment() -> None:
    """ModelGarden() with neither set still sends generate calls to GCLOUD_PROJECT."""
    with patch.dict(os.environ, {'GCLOUD_PROJECT': 'env-proj', 'GOOGLE_CLOUD_PROJECT': ''}):
        assert await _project_sent_to_vertex(ModelGarden(location='us-central1')) == 'env-proj'


def test_anthropic_model_garden_uses_anthropic_config_schema() -> None:
    """Anthropic Model Garden advertises the schema enforced by its handler."""
    schema = AnthropicModelGarden.get_config_schema()
    assert issubclass(schema, AnthropicConfig)


def test_anthropic_model_garden_does_not_advertise_api_key() -> None:
    """Vertex authenticates with Google credentials, so apiKey is not offered."""
    properties = AnthropicModelGarden.get_config_schema().model_json_schema()['properties']
    assert 'apiKey' not in properties
    assert 'apiVersion' in properties
