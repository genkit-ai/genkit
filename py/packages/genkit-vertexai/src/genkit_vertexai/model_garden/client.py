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
from openai import AsyncOpenAI as _AsyncOpenAI


def _refresh_credentials(
    project_id: str | None,
) -> tuple[google.auth.credentials.Credentials, str]:
    """Resolve and refresh Google Cloud credentials (blocking I/O).

    This is intentionally synchronous — it is called via
    ``asyncio.to_thread`` so the event loop is never blocked.

    Args:
        project_id: Explicit project ID, or None to auto-detect.

    Returns:
        A (credentials, project_id) tuple with a refreshed token.
    """
    credentials: google.auth.credentials.Credentials
    resolved_project_id: str | None = project_id
    if project_id:
        credentials, _ = auth.default()
    else:
        credentials, resolved_project_id = auth.default()

    credentials.refresh(google.auth.transport.requests.Request())

    if not resolved_project_id:
        raise ValueError('Could not determine project_id from credentials or arguments.')

    return credentials, resolved_project_id


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
                ``project_id``.

        Returns:
            A configured AsyncOpenAI client.
        """
        location = openai_params.get('location')
        project_id_str = str(val) if (val := openai_params.get('project_id')) is not None else None

        # Offload blocking credential refresh to a thread.
        credentials, resolved_project_id = await asyncio.to_thread(_refresh_credentials, project_id_str)

        base_url = (
            f'https://{location}-aiplatform.googleapis.com/v1beta1'
            f'/projects/{resolved_project_id}/locations/{location}/endpoints/openapi'
        )
        return _AsyncOpenAI(api_key=credentials.token, base_url=base_url)
