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
from unittest.mock import MagicMock, patch

import pytest
from genkit_vertexai.model_garden import ModelGarden, ModelGardenPlugin
from genkit_vertexai.model_garden.model_garden import ModelGardenModel


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
