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


"""Vertex AI client.

Provides an async factory for creating AsyncOpenAI clients authenticated
with Google Cloud credentials. Credential refresh is performed off the
event loop using ``asyncio.to_thread`` to avoid blocking.
"""

import asyncio

import google.auth.credentials
import google.auth.transport.requests
from google import auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError, TransportError
from openai import AsyncOpenAI as _AsyncOpenAI

from genkit import GenkitError
from genkit.plugin_api import mark_provider_error


def _refresh_credentials(
    project: str | None,
) -> tuple[google.auth.credentials.Credentials, str]:
    """Resolve and refresh Google Cloud credentials (blocking I/O).

    This is intentionally synchronous — it is called via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Args:
        project: Explicit project ID, or None to auto-detect.

    Returns:
        A (credentials, project) tuple with a refreshed token.

    Raises:
        GenkitError: UNAUTHENTICATED when ADC is missing or the refresh is
            rejected; FAILED_PRECONDITION when no project can be resolved.
            A transient refresh failure (marked retryable, or raised from a
            TransportError such as a metadata-server blip) and a bare
            TransportError propagate unchanged.
    """
    credentials: google.auth.credentials.Credentials
    resolved_project: str | None = project
    try:
        if project:
            credentials, _ = auth.default()
        else:
            credentials, resolved_project = auth.default()

        credentials.refresh(google.auth.transport.requests.Request())
    except (DefaultCredentialsError, RefreshError) as e:
        if e.retryable or isinstance(e.__cause__, TransportError):
            raise
        raise mark_provider_error(
            error=GenkitError(
                status='UNAUTHENTICATED',
                message='Google Cloud credentials are missing or were rejected',
                cause=e,
            )
        ) from e

    if not resolved_project:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message='Could not determine project from credentials or arguments.',
        )

    return credentials, resolved_project


def _openai_base_url(*, location: str, project: str) -> str:
    return (
        f'https://{location}-aiplatform.googleapis.com/v1beta1'
        f'/projects/{project}/locations/{location}/endpoints/openapi'
    )


class CachedOpenAI:
    """One AsyncOpenAI plus its Google credentials, per event loop.

    The bearer token expires about hourly. Refresh only then so a long-lived
    server keeps one HTTP pool and doesn't 401 after the token dies.
    """

    def __init__(self, *, location: str, project: str | None) -> None:
        """Hold location and project until the first generate builds the client."""
        self._location = location
        self._project = project
        self._credentials: google.auth.credentials.Credentials | None = None
        self._resolved_project: str | None = None
        self._client: _AsyncOpenAI | None = None

    async def get(self) -> _AsyncOpenAI:
        """Return the cached client, refreshing the token only when it has expired."""
        if self._credentials is None or not self._credentials.valid:
            self._credentials, self._resolved_project = await asyncio.to_thread(_refresh_credentials, self._project)
            token = self._credentials.token
            if not token:
                raise ValueError('Google credentials did not return an access token.')
            if self._resolved_project is None:
                raise ValueError('Could not determine project from credentials or arguments.')
            if self._client is None:
                self._client = _AsyncOpenAI(
                    api_key=token,
                    base_url=_openai_base_url(location=self._location, project=self._resolved_project),
                )
            else:
                self._client.api_key = token
        if self._client is None:
            raise RuntimeError('Model Garden OpenAI client was not built after credential refresh.')
        return self._client


class OpenAIClient:
    """Factory for AsyncOpenAI clients authenticated via Google Cloud.

    Use the async ``create()`` classmethod instead of direct instantiation
    to avoid blocking the event loop during credential refresh.
    """

    @classmethod
    async def create(cls, **openai_params: object) -> _AsyncOpenAI:
        """Create an AsyncOpenAI client with refreshed Google credentials.

        Runs the blocking ``credentials.refresh()`` call in a thread so
        the event loop is never blocked.

        Args:
            **openai_params: Must include ``location`` and optionally
                ``project``.

        Returns:
            A configured AsyncOpenAI client.
        """
        location = str(openai_params.get('location') or '')
        project_str = str(val) if (val := openai_params.get('project')) is not None else None

        # Offload blocking credential refresh to a thread.
        credentials, resolved_project = await asyncio.to_thread(_refresh_credentials, project_str)

        return _AsyncOpenAI(
            api_key=credentials.token,
            base_url=_openai_base_url(location=location, project=resolved_project),
        )
