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

"""VertexAI client construction without a resolvable project."""

from unittest.mock import patch

import pytest
from genkit_google_genai._google import _plugin_client
from google.genai import types as genai_types

from genkit import GenkitError


def test_vertex_without_project_is_failed_precondition() -> None:
    """google-genai's bare ValueError becomes FAILED_PRECONDITION naming VertexAI(project=...)."""
    kwargs = {
        'vertexai': True,
        'api_key': None,
        'credentials': None,
        'project': None,
        'location': 'us-central1',
        'http_options': genai_types.HttpOptions(),
    }
    with (
        patch('genkit_google_genai._google.genai.client.Client', side_effect=ValueError('Project or API key must be set')),
        pytest.raises(GenkitError) as raised,
    ):
        _plugin_client(kwargs)

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'VertexAI(project=...)' in raised.value.message
    assert 'GOOGLE_CLOUD_PROJECT' in raised.value.message


def test_other_client_value_errors_are_not_rewritten() -> None:
    """A ValueError unrelated to a missing Vertex project passes through unchanged."""
    kwargs = {'vertexai': True, 'api_key': None, 'project': 'menu-prod', 'location': 'us-central1'}
    with (
        patch('genkit_google_genai._google.genai.client.Client', side_effect=ValueError('bad location')),
        pytest.raises(ValueError, match='bad location'),
    ):
        _plugin_client(kwargs)
