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

"""Shared helpers for Google AI Interactions-backed models."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast

from google import genai
from google.genai.errors import APIError
from google.genai.types import HttpOptions
from pydantic import BaseModel

from genkit import GenkitError, Message
from genkit._core._error import ErrorResponseMetadata, StatusName
from genkit.plugin_api import GENKIT_CLIENT_HEADER
from genkit_google_genai._interactions.options import ClientOptions, ResponseModality

_CLIENT_OPTION_KEYS = frozenset({
    'api_key',
    'base_url',
    'api_version',
    'custom_headers',
    'timeout',
    'experimental_debug_traces',
})


def get_api_key_from_env() -> str | None:
    """Read a Gemini API key from common environment variables."""
    return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY') or os.getenv('GOOGLE_GENAI_API_KEY')


def calculate_api_key(
    plugin_api_key: str | None,
    request_api_key: str | None,
) -> str:
    """Resolve the effective API key for an Interactions call.

    Fallback Hierarchy:
        1. request_api_key: Override passed in request config.
        2. plugin_api_key: Plugin initialization key (from self._client_kwargs).
        3. Environment variables: GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY.
    """
    api_key = request_api_key or plugin_api_key or get_api_key_from_env()
    if not api_key:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=(
                'Please pass in the API key or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable.\n'
                'For more details see https://genkit.dev/docs/plugins/google-genai/'
            ),
        )
    return api_key


def config_as_dict(config: Any) -> dict[str, Any]:  # noqa: ANN401
    """Normalize model config to a plain snake_case dict."""
    if config is None:
        return {}
    if isinstance(config, BaseModel):
        return config.model_dump(exclude_none=True)
    if isinstance(config, dict):
        return dict(config)
    return {}


def extract_version(model_name: str) -> str:
    """Return the bare model version from a namespaced model name."""
    if '/' in model_name:
        return model_name.split('/', 1)[1]
    return model_name


def downgrade_system_messages(messages: list[Message]) -> list[Message]:
    """Map system turns to user turns for agents that reject system instructions."""
    downgraded = [message.model_copy(deep=True) for message in messages]
    for message in downgraded:
        if message.role == 'system':
            message.role = 'user'
    return downgraded


def merge_client_options(
    base: ClientOptions,
    config: dict[str, Any],
) -> ClientOptions:
    """Apply per-request client overrides from model config onto base plugin options."""
    merged = cast(ClientOptions, dict(base))
    if base_url := config.get('base_url'):
        merged['base_url'] = str(base_url)
    if api_version := config.get('api_version'):
        merged['api_version'] = str(api_version)
    if custom_headers := config.get('custom_headers'):
        merged['custom_headers'] = dict(custom_headers)
    if isinstance(config.get('timeout'), (int, float)):
        merged['timeout'] = float(config['timeout'])
    return merged


def remove_client_option_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Drop client-only config keys before passthrough to the wire payload."""
    return {key: value for key, value in config.items() if key not in _CLIENT_OPTION_KEYS}


def client_options_for_operation(
    client_options: ClientOptions,
    *,
    api_key: str | None = None,
) -> ClientOptions:
    """Persist client settings on Operation.metadata['clientOptions'] for check/cancel calls."""
    persisted = cast(ClientOptions, dict(client_options))
    if api_key:
        persisted['api_key'] = api_key
    return persisted


def response_modalities_from_config(
    config: dict[str, Any],
    *,
    default: list[ResponseModality] | None = None,
) -> list[ResponseModality] | None:
    """Read response_modalities from config (already lowercase at the Python surface)."""
    raw = config.get('response_modalities')
    if raw is None:
        return default
    return [str(item).lower() for item in raw]  # type: ignore[misc]


def _http_options_from_client_options(client_options: ClientOptions | None) -> HttpOptions | None:
    options = client_options or {}
    headers = dict(options.get('custom_headers') or {})
    # Keep Genkit visible in traces even when callers override other headers.
    headers.setdefault('x-goog-api-client', GENKIT_CLIENT_HEADER)
    headers.setdefault('user-agent', GENKIT_CLIENT_HEADER)

    http_kwargs: dict[str, Any] = {'headers': headers}
    if api_version := options.get('api_version'):
        http_kwargs['api_version'] = api_version
    if base_url := options.get('base_url'):
        http_kwargs['base_url'] = base_url
    if (timeout := options.get('timeout')) is not None and timeout >= 0:
        # google-genai HttpOptions.timeout is milliseconds.
        http_kwargs['timeout'] = int(timeout * 1000)

    return HttpOptions(**http_kwargs)


def make_genai_client(
    *,
    api_key: str,
    client_options: ClientOptions | None = None,
) -> genai.Client:
    """Build a google-genai Client for Interactions calls."""
    return genai.Client(
        api_key=api_key,
        http_options=_http_options_from_client_options(client_options),
    )


def _options_need_ephemeral_client(
    plugin_client_options: ClientOptions,
    client_options: ClientOptions,
) -> bool:
    for key in ('base_url', 'api_version', 'timeout'):
        if client_options.get(key) != plugin_client_options.get(key):
            return True
    return (client_options.get('custom_headers') or {}) != (plugin_client_options.get('custom_headers') or {})


@asynccontextmanager
async def resolve_interactions_client(
    *,
    client_getter: Callable[[], genai.Client] | None,
    plugin_api_key: str | None,
    api_key: str,
    request_api_key: str | None,
    plugin_client_options: ClientOptions,
    client_options: ClientOptions,
) -> AsyncIterator[genai.Client]:
    """Yield a shared plugin client when safe, otherwise an ephemeral one."""
    reuse_shared = (
        client_getter is not None
        and request_api_key is None
        and (plugin_api_key is None or api_key == plugin_api_key)
        and not _options_need_ephemeral_client(plugin_client_options, client_options)
    )
    if reuse_shared:
        assert client_getter is not None
        yield client_getter()
        return

    client = make_genai_client(api_key=api_key, client_options=client_options)
    try:
        yield client
    finally:
        await client.aio.aclose()


def _parse_retry_after_ms(value: str | None) -> float | None:
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


def _status_from_api_error(error: APIError) -> StatusName:
    raw_status = error.status
    if isinstance(raw_status, str):
        candidate = raw_status.upper()
        # API sometimes returns gRPC status names directly.
        valid: set[str] = {
            'OK',
            'CANCELLED',
            'UNKNOWN',
            'INVALID_ARGUMENT',
            'DEADLINE_EXCEEDED',
            'NOT_FOUND',
            'ALREADY_EXISTS',
            'PERMISSION_DENIED',
            'UNAUTHENTICATED',
            'RESOURCE_EXHAUSTED',
            'FAILED_PRECONDITION',
            'ABORTED',
            'OUT_OF_RANGE',
            'UNIMPLEMENTED',
            'INTERNAL',
            'UNAVAILABLE',
            'DATA_LOSS',
        }
        if candidate in valid:
            return cast(StatusName, candidate)
    return _status_for_http_code(int(error.code or 0))


def map_genai_error(exc: BaseException) -> GenkitError:
    """Map a google-genai SDK error onto GenkitError for callers/poll backoff."""
    if isinstance(exc, GenkitError):
        return exc
    if isinstance(exc, APIError):
        retry_after_ms: float | None = None
        headers: dict[str, str] = {}
        response = getattr(exc, 'response', None)
        raw_headers = getattr(response, 'headers', None)
        if raw_headers is not None:
            try:
                headers = {str(key).lower(): str(value) for key, value in raw_headers.items()}
            except Exception:  # noqa: BLE001 - headers shape varies by transport
                headers = {}
            retry_after_ms = _parse_retry_after_ms(headers.get('retry-after'))
        response_metadata: ErrorResponseMetadata | None = None
        if retry_after_ms is not None or headers:
            meta: ErrorResponseMetadata = {}
            if retry_after_ms is not None:
                meta['retry_after_ms'] = retry_after_ms
            if headers:
                meta['headers'] = headers
            response_metadata = meta
        return GenkitError(
            status=_status_from_api_error(exc),
            message=exc.message or str(exc),
            details=getattr(exc, 'details', None),
            response_metadata=response_metadata,
        )
    return GenkitError(status='UNKNOWN', message=str(exc))
