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

"""Turn google-genai SDK failures into GenkitErrors that Retry and Fallback can act on."""

import asyncio
import importlib
import math
import re
from collections.abc import Mapping
from typing import cast

import httpx
from google.genai.errors import APIError

from genkit import GenkitError
from genkit.plugin_api import StatusName, provider_error

_RETRY_INFO_TYPE = 'type.googleapis.com/google.rpc.RetryInfo'
_ERROR_INFO_TYPE = 'type.googleapis.com/google.rpc.ErrorInfo'
_DURATION = re.compile(r'(\d+(?:\.\d+)?)s')
# HTTP codes that several statuses share, and the statuses an error body can
# name to say which one it is.
_SHARED_CODE_STATUSES: dict[int, frozenset[str]] = {
    400: frozenset({'INVALID_ARGUMENT', 'FAILED_PRECONDITION', 'OUT_OF_RANGE'}),
    409: frozenset({'ALREADY_EXISTS', 'ABORTED'}),
    500: frozenset({'INTERNAL', 'DATA_LOSS'}),
}

# On Python 3.10, asyncio.TimeoutError is not the builtin TimeoutError, so list both.
_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError)
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (httpx.NetworkError, httpx.RemoteProtocolError, ConnectionError)
try:
    # The SDK sends over aiohttp when it's installed.
    _CONNECTION_ERRORS = (*_CONNECTION_ERRORS, importlib.import_module('aiohttp').ClientConnectionError)
except ImportError:
    pass

# Network failures the SDK lets through after its own retries: the request
# never got an answer, so there's no HTTP status to map.
TRANSPORT_ERRORS: tuple[type[BaseException], ...] = (*_TIMEOUT_ERRORS, *_CONNECTION_ERRORS)


def api_error(error: APIError) -> GenkitError:
    """GenkitError for an HTTP error response the SDK raised.

    When several statuses share the HTTP code, the status the error body names
    picks one, so a 400 that says FAILED_PRECONDITION keeps that status. A key
    the Gemini API rejects is UNAUTHENTICATED.
    """
    response = getattr(error, 'response', None)
    return provider_error(
        error,
        http_status=error.code,
        status=_body_status(error),
        headers=getattr(response, 'headers', None),
        retry_after_ms=retry_info_ms(error.details),
        message=error.message or str(error),
    )


def transport_error(error: BaseException) -> GenkitError:
    """DEADLINE_EXCEEDED for a timeout, UNAVAILABLE for a refused or dropped connection."""
    # Timeouts first: aiohttp's read timeout is also a connection error.
    if isinstance(error, _TIMEOUT_ERRORS):
        return provider_error(error, status='DEADLINE_EXCEEDED')
    return provider_error(error, status='UNAVAILABLE')


def retry_info_ms(body: object) -> float | None:
    """The google.rpc.RetryInfo delay in an error body, in milliseconds.

    A Gemini 429 usually says how long to back off in the body rather than a
    Retry-After header, so retry reads it here to avoid hammering the quota.
    A missing or malformed delay is None.
    """
    delay = _field(_detail(body, _RETRY_INFO_TYPE), 'retryDelay')
    match = _DURATION.fullmatch(delay.strip()) if isinstance(delay, str) else None
    if match is None:
        return None
    delay_ms = float(match.group(1)) * 1000
    return delay_ms if math.isfinite(delay_ms) else None


def _body_status(error: APIError) -> StatusName | None:
    """The status the error body names for a shared HTTP code, or None to map the code."""
    # The Gemini API answers a bad key with INVALID_ARGUMENT and names it only in ErrorInfo.
    if _field(_detail(error.details, _ERROR_INFO_TYPE), 'reason') == 'API_KEY_INVALID':
        return 'UNAUTHENTICATED'
    status = error.status
    if isinstance(status, str) and status in _SHARED_CODE_STATUSES.get(error.code, frozenset()):
        return cast(StatusName, status)
    return None


def _detail(body: object, type_url: str) -> object:
    """The first google.rpc detail of the given type in an error body, if any."""
    details = _field(_field(body, 'error') or body, 'details')
    if not isinstance(details, list):
        return None
    return next((d for d in cast(list[object], details) if _field(d, '@type') == type_url), None)


def _field(value: object, key: str) -> object:
    return cast(Mapping[str, object], value).get(key) if isinstance(value, Mapping) else None
