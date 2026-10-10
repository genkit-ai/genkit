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

"""Raw HTTP helpers for the Google AI Interactions API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.parse import quote

import httpx
from genkit_google_genai._interactions._options import ClientOptions
from genkit_google_genai._provider_errors import TRANSPORT_ERRORS, transport_error
from google.genai.interactions import Interaction

from genkit import GenkitError
from genkit.plugin_api import GENKIT_CLIENT_HEADER, loop_local_client, provider_error

DEFAULT_API_VERSION = 'v1beta'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
API_REVISION = '2026-05-20'
# Creates can run for many minutes. No read timeout, but keep a connect
# budget so a hung handshake doesn't sit forever.
NO_TIMEOUT = httpx.Timeout(None, connect=10.0)
RESERVED_HEADERS = ('x-goog-api-key', 'x-goog-api-client')


@loop_local_client
def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=NO_TIMEOUT)


def google_ai_url(
    resource_path: str,
    *,
    client_options: ClientOptions | None = None,
) -> str:
    """Build a Google AI REST URL for the given resource path."""
    opts = client_options or ClientOptions()
    api_version = opts.api_version or DEFAULT_API_VERSION
    base_url = (opts.base_url or DEFAULT_BASE_URL).rstrip('/')
    return f'{base_url}/{api_version}/{resource_path}'


def headers(*, api_key: str, client_options: ClientOptions | None) -> dict[str, str]:
    """Build request headers; api key and Genkit client attribution win over custom.

    Header names are matched case-insensitively so ``X-Goog-Api-Key`` cannot
    sneak a second key onto the request.
    """
    custom = httpx.Headers((client_options or ClientOptions()).custom_headers or {})
    for name in RESERVED_HEADERS:
        custom.pop(name, None)
    custom['Content-Type'] = 'application/json'
    custom['x-goog-api-client'] = GENKIT_CLIENT_HEADER
    custom['Api-Revision'] = API_REVISION
    custom['x-goog-api-key'] = api_key
    return dict(custom)


def timeout_seconds(client_options: ClientOptions | None) -> float | None:
    """Convert ClientOptions.timeout (milliseconds) to httpx seconds."""
    opts = client_options or ClientOptions()
    if opts.timeout is not None and opts.timeout >= 0:
        return opts.timeout / 1000.0
    return None


async def create_interaction(
    api_key: str,
    body: dict[str, object],
    client_options: ClientOptions | None = None,
) -> Interaction:
    """POST /interactions and return the parsed Interaction."""
    url = google_ai_url('interactions', client_options=client_options)
    created = await request(
        method='POST',
        url=url,
        api_key=api_key,
        client_options=client_options,
        json_body=body,
    )
    assert created is not None
    return created


async def get_interaction(
    api_key: str,
    interaction_id: str,
    client_options: ClientOptions | None = None,
) -> Interaction:
    """GET /interactions/{id} and return the parsed Interaction."""
    url = google_ai_url(f'interactions/{quote(interaction_id, safe="")}', client_options=client_options)
    found = await request(
        method='GET',
        url=url,
        api_key=api_key,
        client_options=client_options,
    )
    assert found is not None
    return found


async def cancel_interaction(
    api_key: str,
    interaction_id: str,
    client_options: ClientOptions | None = None,
) -> Interaction:
    """POST /interactions/{id}/cancel and return a cancelled Interaction."""
    url = google_ai_url(f'interactions/{quote(interaction_id, safe="")}/cancel', client_options=client_options)
    try:
        interaction = await request(
            method='POST',
            url=url,
            api_key=api_key,
            client_options=client_options,
            allow_empty=True,
        )
    except GenkitError as error:
        if error.status == 'CANCELLED':
            return Interaction.model_validate({'id': interaction_id, 'status': 'cancelled'})
        raise
    if interaction is None:
        return Interaction.model_validate({'id': interaction_id, 'status': 'cancelled'})
    return interaction.model_copy(update={'status': 'cancelled'})


async def request(
    *,
    method: str,
    url: str,
    api_key: str,
    client_options: ClientOptions | None,
    json_body: dict[str, object] | None = None,
    allow_empty: bool = False,
) -> Interaction | None:
    """Issue one Interactions HTTP call and parse the Interaction body."""
    # Auth/key headers are per-request; the loop-local client is just the transport.
    client = _http_client()
    request_headers = headers(api_key=api_key, client_options=client_options)
    timeout = timeout_seconds(client_options)

    try:
        if timeout is not None:
            response = await client.request(
                method,
                url,
                headers=request_headers,
                json=json_body,
                timeout=timeout,
            )
        else:
            response = await client.request(
                method,
                url,
                headers=request_headers,
                json=json_body,
            )
    except TRANSPORT_ERRORS as error:
        raise transport_error(error) from error

    if response.is_success:
        if not response.content:
            if allow_empty:
                return None
            raise GenkitError(
                status='INTERNAL',
                message=f'Received an empty response from {url}',
            )
        try:
            return Interaction.model_validate(response.json())
        except Exception as error:
            raise GenkitError(
                status='INTERNAL',
                message=f'Unable to parse Interaction response from {url}: {error}',
            ) from error

    error_message = response.text
    error_detail: object | None = None
    try:
        payload: object = response.json()
        error_detail = payload
        if isinstance(payload, Mapping):
            api_error = payload.get('error')
            if isinstance(api_error, Mapping) and api_error.get('message') is not None:
                error_message = str(api_error['message'])
    except json.JSONDecodeError:
        pass

    message = f'Request to {url} failed with HTTP {response.status_code} {response.reason_phrase}: {error_message}'
    cause = httpx.HTTPStatusError(message, request=httpx.Request(method, url), response=response)
    error = provider_error(cause, http_status=response.status_code, headers=response.headers, message=message)
    if isinstance(error_detail, Mapping):
        error.details.update(error_detail)
    raise error from cause
