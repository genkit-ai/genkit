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
from typing import Any, cast

from google.genai.errors import APIError

from genkit import GenkitError
from genkit._core._error import ErrorResponseMetadata, StatusName
from genkit_google_genai._interactions.client import parse_retry_after_ms, status_for_http_code
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


def client_overrides_from_config(
    *,
    base_url: str | None = None,
    api_version: str | None = None,
    timeout: float | None = None,
    custom_headers: dict[str, str] | None = None,
) -> ClientOptions:
    """Lift per-request transport knobs out of a model config into ClientOptions."""
    return ClientOptions(
        base_url=base_url,
        api_version=api_version,
        timeout=timeout,
        custom_headers=custom_headers,
    )


def partition_keys(
    payload: dict[str, Any],
    *groups: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    """Split payload into one dict per key group, then a remainder of leftovers.

    Does not mutate payload. Keys listed in any group are claimed for that group
    (or dropped from the remainder even when absent), so callers can peel a
    dumped model config into agent_config / create options / tool fields /
    passthrough extras without colliding.
    """
    claimed: set[str] = set()
    for keys in groups:
        claimed.update(keys)
    parts = tuple({key: payload[key] for key in keys if key in payload} for keys in groups)
    remainder = {key: value for key, value in payload.items() if key not in claimed}
    return (*parts, remainder)


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
