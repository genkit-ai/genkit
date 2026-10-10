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
from genkit_vertexai._model_garden._client import CachedOpenAI
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError

from genkit import GenkitError


def _adc(project: str | None) -> MagicMock:
    """Stand-in for google.auth.default that returns a token and `project`."""
    credentials = MagicMock()
    credentials.token = 'token'
    return MagicMock(return_value=(credentials, project))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('explicit', 'expected'),
    [
        pytest.param('menu-prod', 'menu-prod', id='explicit-beats-adc'),
        pytest.param(None, 'adc-proj', id='adc-fallback'),
    ],
)
@patch('google.auth.transport.requests.Request')
async def test_client_targets_resolved_project(
    mock_request_cls: MagicMock, explicit: str | None, expected: str
) -> None:
    """The OpenAI base_url points at project=, or the ADC project when none is passed."""
    with (
        patch('google.auth.default', _adc('adc-proj')),
        patch('genkit_vertexai._model_garden._client._AsyncOpenAI') as openai_cls,
    ):
        await CachedOpenAI(location='us-central1', project=explicit).get()

    assert openai_cls.call_args.kwargs['base_url'] == (
        f'https://us-central1-aiplatform.googleapis.com/v1beta1'
        f'/projects/{expected}/locations/us-central1/endpoints/openapi'
    )
    assert openai_cls.call_args.kwargs['api_key'] == 'token'


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
        await CachedOpenAI(location='us-central1', project='menu-prod').get()

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
@patch('google.auth.transport.requests.Request')
async def test_transient_credential_refresh_failure_is_unavailable(
    mock_request_cls: MagicMock, flaky: Exception
) -> None:
    """A credential refresh that fails transiently raises UNAVAILABLE, so retry tries again."""
    credentials = MagicMock()
    credentials.refresh.side_effect = flaky
    with patch('google.auth.default', return_value=(credentials, 'menu-prod')), pytest.raises(GenkitError) as raised:
        await CachedOpenAI(location='us-central1', project='menu-prod').get()

    assert raised.value.status == 'UNAVAILABLE'
    assert raised.value.__cause__ is flaky


@pytest.mark.asyncio
@patch('google.auth.transport.requests.Request')
async def test_missing_project_is_failed_precondition(mock_request_cls: MagicMock) -> None:
    with patch('google.auth.default', _adc(None)), pytest.raises(GenkitError) as raised:
        await CachedOpenAI(location='us-central1', project=None).get()

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'project' in str(raised.value)
