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
from genkit.plugin_api import provider_error

_RETRY_INFO_TYPE = 'type.googleapis.com/google.rpc.RetryInfo'
_DURATION = re.compile(r'(\d+(?:\.\d+)?)s')

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
    """GenkitError for an HTTP error response the SDK raised."""
    response = getattr(error, 'response', None)
    return provider_error(
        error,
        http_status=error.code,
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
    details = _field(_field(body, 'error') or body, 'details')
    if not isinstance(details, list):
        return None
    for detail in cast(list[object], details):
        if _field(detail, '@type') == _RETRY_INFO_TYPE:
            delay = _field(detail, 'retryDelay')
            match = _DURATION.fullmatch(delay.strip()) if isinstance(delay, str) else None
            if match is None:
                return None
            delay_ms = float(match.group(1)) * 1000
            return delay_ms if math.isfinite(delay_ms) else None
    return None


def _field(value: object, key: str) -> object:
    return cast(Mapping[str, object], value).get(key) if isinstance(value, Mapping) else None
