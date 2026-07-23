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

"""Raw HTTP client for the Google AI Interactions API."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from genkit import GenkitError
from genkit._core._error import ErrorResponseMetadata, StatusName
from genkit.plugin_api import GENKIT_CLIENT_HEADER
from genkit_google_genai._interactions.types import (
    API_REVISION,
    ClientOptions,
    CreateInteractionRequest,
    GeminiInteraction,
)

logger = logging.getLogger(__name__)

DEFAULT_API_VERSION = 'v1beta'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'


def get_google_ai_url(
    *,
    resource_path: str,
    resource_method: str | None = None,
    query_params: str | None = None,
    client_options: ClientOptions | None = None,
) -> str:
    """Build a Google AI REST URL for the given resource path."""
    api_version = (client_options or {}).get('api_version') or DEFAULT_API_VERSION
    base_url = (client_options or {}).get('base_url') or DEFAULT_BASE_URL
    url = f'{base_url}/{api_version}/{resource_path}'
    if resource_method:
        url += f':{resource_method}'
    if query_params:
        url += f'?{query_params}'
    return url


def _parse_retry_after_ms(value: str) -> float | None:
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


def _status_for_http_code(status_code: int) -> StatusName:
    match status_code:
        case 429:
            return 'RESOURCE_EXHAUSTED'
        case 400:
            return 'INVALID_ARGUMENT'
        case 499:
            return 'CANCELLED'
        case 500:
            return 'INTERNAL'
        case 503:
            return 'UNAVAILABLE'
        case _:
            return 'UNKNOWN'


def _build_headers(api_key: str | None, client_options: ClientOptions | None) -> dict[str, str]:
    custom_headers = dict((client_options or {}).get('custom_headers') or {})
    custom_headers.pop('x-goog-api-key', None)
    custom_headers.pop('x-goog-api-client', None)
    headers = {
        **custom_headers,
        'Content-Type': 'application/json',
        'x-goog-api-client': GENKIT_CLIENT_HEADER,
        'Api-Revision': API_REVISION,
    }
    if api_key:
        headers['x-goog-api-key'] = api_key
    return headers


class InteractionsClient:
    """HTTP client for Google AI Interactions endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        client_options: ClientOptions | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._client_options: ClientOptions = client_options or {}
        self._http_client = http_client
        self._owns_client = http_client is None

    async def __aenter__(self) -> InteractionsClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned by this instance."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            timeout = self._client_options.get('timeout')
            timeout_config = timeout if timeout is not None and timeout >= 0 else None
            self._http_client = httpx.AsyncClient(timeout=timeout_config)
        return self._http_client

    async def create_interaction(self, request: CreateInteractionRequest) -> GeminiInteraction:
        """Create a new interaction."""
        url = get_google_ai_url(resource_path='interactions', client_options=self._client_options)
        return await self._request('POST', url, json_body=request)

    async def get_interaction(self, interaction_id: str) -> GeminiInteraction:
        """Fetch an interaction by ID."""
        url = get_google_ai_url(
            resource_path=f'interactions/{interaction_id}',
            client_options=self._client_options,
        )
        return await self._request('GET', url)

    async def cancel_interaction(self, interaction_id: str) -> GeminiInteraction:
        """Cancel an in-progress interaction."""
        url = get_google_ai_url(
            resource_path=f'interactions/{interaction_id}/cancel',
            client_options=self._client_options,
        )
        try:
            await self._request('POST', url)
            raise GenkitError(status='CANCELLED', message='successfully cancelled')
        except GenkitError as error:
            if error.status == 'CANCELLED':
                return {'id': interaction_id, 'status': 'cancelled'}
            raise

    async def _request(
        self,
        method: str,
        url: str,
        json_body: CreateInteractionRequest | None = None,
    ) -> GeminiInteraction:
        headers = _build_headers(self._api_key, self._client_options)
        client = self._get_client()
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
            )
        except httpx.HTTPError as error:
            logger.exception('Interactions request failed')
            raise GenkitError(status='UNKNOWN', message=f'Failed to fetch from {url}: {error}') from error

        if response.is_success:
            if not response.content:
                return {}
            return response.json()

        error_text = response.text
        error_message = error_text
        error_detail: Any | None = None
        try:
            payload = response.json()
            error_detail = payload
            api_error = payload.get('error') if isinstance(payload, dict) else None
            if isinstance(api_error, dict) and api_error.get('message'):
                error_message = api_error['message']
                details = api_error.get('details')
                if isinstance(details, list):
                    detail_lines: list[str] = []
                    for detail in details:
                        if isinstance(detail, dict) and isinstance(detail.get('detail'), str):
                            detail_text = detail['detail']
                            if '[ORIGINAL ERROR]' in detail_text:
                                detail_text = detail_text.split('[ORIGINAL ERROR]', 1)[1].split('[', 1)[0].strip()
                            detail_lines.append(f'{detail_text}\nRaw: {json.dumps(detail, indent=2)}')
                        else:
                            detail_lines.append(json.dumps(detail, indent=2))
                    if detail_lines:
                        error_message += '\nDetails:\n' + '\n'.join(detail_lines)
        except json.JSONDecodeError:
            pass

        retry_after_ms = _parse_retry_after_ms(response.headers.get('retry-after', ''))
        response_metadata: ErrorResponseMetadata | None = (
            {'retry_after_ms': retry_after_ms} if retry_after_ms is not None else None
        )
        raise GenkitError(
            status=_status_for_http_code(response.status_code),
            message=(f'Error fetching from {url}: [{response.status_code} {response.reason_phrase}] {error_message}'),
            details=error_detail,
            response_metadata=response_metadata,
        )
