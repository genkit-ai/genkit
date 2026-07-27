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
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast

from google import genai
from google.genai.errors import APIError
from google.genai.types import HttpOptions

from genkit import GenkitError
from genkit._core._error import ErrorResponseMetadata, StatusName
from genkit.plugin_api import GENKIT_CLIENT_HEADER
from genkit_google_genai._interactions.options import ClientOptions

# Snake_case: callers dump config with by_alias=False before we strip these.
CLIENT_OPTION_KEYS = frozenset({
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
    """Resolve the effective API key for an Interactions call."""
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


def extract_version(model_name: str) -> str:
    """Return the bare model version from a namespaced model name."""
    if '/' in model_name:
        return model_name.split('/', 1)[1]
    return model_name


def remove_client_option_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Drop client-only config keys before passthrough to the wire payload."""
    return {key: value for key, value in config.items() if key not in CLIENT_OPTION_KEYS}


def take_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Pull the given keys out of payload (mutating) and return them as a new dict.

    Used to peel a dumped model config into the buckets interactions.create
    expects — top-level create fields, agent_config, tools — so the leftovers
    can be passed through as extras without colliding.
    """
    return {key: payload.pop(key) for key in keys if key in payload}


def require_interaction_steps(steps: list[Any]) -> list[Any]:
    """Reject an empty Interactions input before we pay for a round trip."""
    if not steps:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='Missing input.',
        )
    return steps


def http_status_code_from_exception(exc: BaseException) -> int | None:
    """Read an HTTP status from SDK errors that aren't google.genai.errors.APIError.

    Interactions (gaos) raises its own BadRequestError hierarchy with
    status_code, which doesn't subclass google.genai.errors.APIError.
    """
    for attr in ('status_code', 'code'):
        raw = getattr(exc, attr, None)
        if raw is None:
            continue
        try:
            code = int(raw)
        except (TypeError, ValueError):
            continue
        if code > 0:
            return code
    return None


def client_options_for_operation(
    client_options: ClientOptions,
    *,
    api_key: str | None = None,
) -> ClientOptions:
    """Persist client settings on an Operation for later check/cancel calls."""
    if api_key:
        return client_options.model_copy(update={'api_key': api_key})
    return client_options


def http_options_from_client_options(client_options: ClientOptions) -> HttpOptions:
    """Translate plugin client options into google-genai transport options."""
    headers = dict(client_options.custom_headers or {})
    # Keep Genkit visible in traces even when callers override other headers.
    headers.setdefault('x-goog-api-client', GENKIT_CLIENT_HEADER)
    headers.setdefault('user-agent', GENKIT_CLIENT_HEADER)

    http_kwargs: dict[str, Any] = {'headers': headers}
    if client_options.api_version:
        http_kwargs['api_version'] = client_options.api_version
    if client_options.base_url:
        http_kwargs['base_url'] = client_options.base_url
    if client_options.timeout is not None and client_options.timeout >= 0:
        # google-genai HttpOptions.timeout is milliseconds.
        http_kwargs['timeout'] = int(client_options.timeout * 1000)

    return HttpOptions(**http_kwargs)


def make_genai_client(
    *,
    api_key: str,
    client_options: ClientOptions | None = None,
) -> genai.Client:
    """Build a google-genai Client for Interactions calls."""
    return genai.Client(
        api_key=api_key,
        http_options=http_options_from_client_options(client_options or ClientOptions()),
    )


def options_need_ephemeral_client(
    plugin_client_options: ClientOptions,
    client_options: ClientOptions,
) -> bool:
    """Report whether a request overrides transport settings the shared client pinned."""
    return (
        client_options.base_url != plugin_client_options.base_url
        or client_options.api_version != plugin_client_options.api_version
        or client_options.timeout != plugin_client_options.timeout
        or (client_options.custom_headers or {}) != (plugin_client_options.custom_headers or {})
    )


@asynccontextmanager
async def resolve_interactions_client(
    *,
    client_getter: Callable[[], genai.Client] | None,
    plugin_api_key: str | None,
    api_key: str,
    request_api_key: str | None,
    plugin_client_options: ClientOptions,
    client_options: ClientOptions,
) -> AsyncGenerator[genai.Client]:
    """Yield a shared plugin client when safe, otherwise an ephemeral one."""
    reuse_shared = (
        client_getter is not None
        and request_api_key is None
        and (plugin_api_key is None or api_key == plugin_api_key)
        and not options_need_ephemeral_client(plugin_client_options, client_options)
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


def status_from_api_error(error: APIError) -> StatusName:
    """Pick the Genkit error status for an SDK error, preferring the status it names."""
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
    return status_for_http_code(int(error.code or 0))


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
            retry_after_ms = parse_retry_after_ms(headers.get('retry-after'))
        response_metadata: ErrorResponseMetadata | None = None
        if retry_after_ms is not None or headers:
            meta: ErrorResponseMetadata = {}
            if retry_after_ms is not None:
                meta['retry_after_ms'] = retry_after_ms
            if headers:
                meta['headers'] = headers
            response_metadata = meta
        return GenkitError(
            status=status_from_api_error(exc),
            message=exc.message or str(exc),
            details=getattr(exc, 'details', None),
            response_metadata=response_metadata,
        )
    # Interactions path: gaos BadRequestError etc. carry status_code but aren't APIError.
    status_code = http_status_code_from_exception(exc)
    if status_code is not None:
        message = getattr(exc, 'message', None) or str(exc)
        return GenkitError(status=status_for_http_code(status_code), message=message)
    return GenkitError(status='UNKNOWN', message=str(exc))
