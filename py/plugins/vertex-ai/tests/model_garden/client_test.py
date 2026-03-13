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

"""Unittests for VertexAI Model Garden OpenAI Client."""

from unittest.mock import MagicMock, patch

import pytest

from genkit.plugins.vertex_ai.model_garden.client import OpenAIClient


@pytest.mark.asyncio
@patch('google.auth.default')
@patch('google.auth.transport.requests.Request')
@patch('genkit.plugins.vertex_ai.model_garden.client._AsyncOpenAI')
async def test_client_initialization_with_explicit_project_id(
    mock_openai_cls: MagicMock, mock_request_cls: MagicMock, mock_default_auth: MagicMock
) -> None:
    """Unittests for init client."""
    mock_location = 'location'
    mock_project_id = 'project_id'
    mock_token = 'token'

    mock_credentials = MagicMock()
    mock_credentials.token = mock_token

    mock_default_auth.return_value = (mock_credentials, 'project_id')

    client_instance = await OpenAIClient.create(location=mock_location, project_id=mock_project_id)

    mock_default_auth.assert_called_once()
    mock_credentials.refresh.assert_called_once()
    mock_request_cls.assert_called_once()

    assert client_instance is not None


@pytest.mark.asyncio
@patch('google.auth.default')
@patch('google.auth.transport.requests.Request')
@patch('genkit.plugins.vertex_ai.model_garden.client._AsyncOpenAI')
async def test_client_initialization_without_explicit_project_id(
    mock_openai_cls: MagicMock, mock_request_cls: MagicMock, mock_default_auth: MagicMock
) -> None:
    """Unittests for init client."""
    mock_location = 'location'
    mock_token = 'token'

    mock_credentials = MagicMock()
    mock_credentials.token = mock_token

    mock_default_auth.return_value = (mock_credentials, 'project_id')

    client_instance = await OpenAIClient.create(
        location=mock_location,
    )

    mock_default_auth.assert_called_once()
    mock_credentials.refresh.assert_called_once()
    mock_request_cls.assert_called_once()

    assert client_instance is not None
