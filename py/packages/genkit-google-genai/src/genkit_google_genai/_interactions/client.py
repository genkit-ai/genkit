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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import cast
from urllib.parse import quote

import httpx
from genkit_google_genai._interactions.options import ClientOptions
from google.genai.interactions import Interaction

from genkit import GenkitError
from genkit._core._error import ErrorResponseMetadata, StatusName
from genkit._core._logger import get_logger
from genkit.plugin_api import GENKIT_CLIENT_HEADER, get_cached_client

logger = get_logger(__name__)

DEFAULT_API_VERSION = 'v1beta'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
API_REVISION = '2026-05-20'
# Creates can run far longer than the shared client's 60s default. No read
# timeout, but keep a connect budget so a hung handshake doesn't sit forever.
CACHE_KEY = 'googleai-interactions'
NO_TIMEOUT = httpx.Timeout(None, connect=10.0)
RESERVED_HEADERS = ('x-goog-api-key', 'x-goog-api-client')


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
    return {key: value for key, value in custom.items()}


def timeout_seconds(client_options: ClientOptions | None) -> float | None:
    """Convert ClientOptions.timeout (milliseconds) to httpx seconds."""
    opts = client_options or ClientOptions()
    if opts.timeout is not None and opts.timeout >= 0:
        return opts.timeout / 1000.0
    return None


def parse_retry_after_ms(value: str | None) -> float | None:
    """Read a Retry-After header, which may be a delay in seconds or an HTTP date."""
    if not value or not value.strip():
        return None
    try:
        seconds = float(value)
    except ValueError:
        seconds = -1.0
    if seconds >= 0:
        return seconds * 1000
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds() * 1000)


def status_for_http_code(status_code: int) -> StatusName:
    """Map an HTTP status onto the Genkit error status callers switch on."""
    match status_code:
        case 429:
            return 'RESOURCE_EXHAUSTED'
        case 400:
            return 'INVALID_ARGUMENT'
        case 401:
            return 'UNAUTHENTICATED'
        case 403:
            return 'PERMISSION_DENIED'
        case 404:
            return 'NOT_FOUND'
        case 499:
            return 'CANCELLED'
        case 500:
            return 'INTERNAL'
        case 503:
            return 'UNAVAILABLE'
        case _:
            return 'UNKNOWN'


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
    client = get_cached_client(cache_key=CACHE_KEY, timeout=NO_TIMEOUT)
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
    except httpx.TimeoutException as error:
        raise GenkitError(
            status='DEADLINE_EXCEEDED',
            message=f'Request to {url} exceeded the configured timeout: {error}',
        ) from error
    except GenkitError:
        raise
    except Exception as error:
        logger.exception('Interactions request failed')
        raise GenkitError(
            status='UNKNOWN',
            message=f'Unable to complete request to {url}: {error}',
        ) from error

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

    retry_after_ms = parse_retry_after_ms(response.headers.get('retry-after'))
    response_metadata: ErrorResponseMetadata | None = None
    if retry_after_ms is not None:
        response_metadata = cast(ErrorResponseMetadata, {'retry_after_ms': retry_after_ms})

    raise GenkitError(
        status=status_for_http_code(response.status_code),
        message=(f'Request to {url} failed with HTTP {response.status_code} {response.reason_phrase}: {error_message}'),
        details=error_detail,
        response_metadata=response_metadata,
    )
