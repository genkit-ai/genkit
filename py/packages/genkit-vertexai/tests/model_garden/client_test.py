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
from genkit_vertexai.model_garden.client import OpenAIClient
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError

from genkit import GenkitError


@pytest.mark.asyncio
@patch('google.auth.default')
@patch('google.auth.transport.requests.Request')
@patch('genkit_vertexai.model_garden.client._AsyncOpenAI')
async def test_client_initialization_with_explicit_project(
    mock_openai_cls: MagicMock, mock_request_cls: MagicMock, mock_default_auth: MagicMock
) -> None:
    """Unittests for init client."""
    mock_location = 'location'
    mock_project = 'project'
    mock_token = 'token'

    mock_credentials = MagicMock()
    mock_credentials.token = mock_token

    mock_default_auth.return_value = (mock_credentials, 'project')

    client_instance = await OpenAIClient.create(location=mock_location, project=mock_project)

    mock_default_auth.assert_called_once()
    mock_credentials.refresh.assert_called_once()
    mock_request_cls.assert_called_once()

    assert client_instance is not None


@pytest.mark.asyncio
@patch('google.auth.default')
@patch('google.auth.transport.requests.Request')
@patch('genkit_vertexai.model_garden.client._AsyncOpenAI')
async def test_client_initialization_without_explicit_project(
    mock_openai_cls: MagicMock, mock_request_cls: MagicMock, mock_default_auth: MagicMock
) -> None:
    """Unittests for init client."""
    mock_location = 'location'
    mock_token = 'token'

    mock_credentials = MagicMock()
    mock_credentials.token = mock_token

    mock_default_auth.return_value = (mock_credentials, 'project')

    client_instance = await OpenAIClient.create(
        location=mock_location,
    )

    mock_default_auth.assert_called_once()
    mock_credentials.refresh.assert_called_once()
    mock_request_cls.assert_called_once()

    assert client_instance is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'auth_error',
    [
        DefaultCredentialsError('Your default credentials were not found.'),
        RefreshError('invalid_grant: Token has been expired or revoked.'),
    ],
)
async def test_credential_failure_is_unauthenticated(auth_error: Exception) -> None:
    """Missing or revoked ADC is UNAUTHENTICATED, so retry doesn't keep calling with it."""
    with patch('google.auth.default', side_effect=auth_error), pytest.raises(GenkitError) as raised:
        await OpenAIClient.create(location='us-central1', project='menu-prod')

    assert raised.value.status == 'UNAUTHENTICATED'
    assert raised.value.cause is auth_error


def _metadata_server_blip() -> RefreshError:
    """What compute_engine.Credentials.refresh raises when the metadata server drops: retryable=False."""
    try:
        try:
            raise TransportError('metadata server unreachable')
        except TransportError as e:
            raise RefreshError(e) from e
    except RefreshError as error:
        return error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'flaky',
    [
        RefreshError('token endpoint returned 503', retryable=True),
        TransportError('metadata server unreachable'),
        _metadata_server_blip(),
    ],
    ids=['retryable_refresh', 'transport', 'metadata_server_blip'],
)
async def test_transient_auth_failure_stays_raw(flaky: Exception) -> None:
    with patch('google.auth.default', side_effect=flaky), pytest.raises(type(flaky)) as raised:
        await OpenAIClient.create(location='us-central1', project='menu-prod')

    assert raised.value is flaky


@pytest.mark.asyncio
@patch('google.auth.transport.requests.Request')
async def test_missing_project_is_failed_precondition(mock_request_cls: MagicMock) -> None:
    credentials = MagicMock()
    credentials.token = 'token'
    with patch('google.auth.default', return_value=(credentials, None)), pytest.raises(GenkitError) as raised:
        await OpenAIClient.create(location='us-central1')

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'project' in str(raised.value)
