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

"""Translate google-genai SDK errors into GenkitErrors."""

from __future__ import annotations

import math
from typing import Any, cast, get_args

from google.genai.errors import APIError

from genkit import ErrorResponseMetadata, GenkitError
from genkit.plugin_api import StatusName, from_http_code, http_code, retry_after_ms_from_error

_STATUS_NAMES: frozenset[str] = frozenset(get_args(StatusName)) - {'OK'}

_RETRY_INFO_TYPE = 'google.rpc.RetryInfo'


def status_for_api_error(error: APIError) -> StatusName | None:
    """Status for an SDK error, or None when it carries neither a status name nor a failing HTTP code.

    A canonical status name the service reported takes precedence over the
    HTTP code. A non-JSON body leaves the HTTP reason phrase in ``status``,
    which is not a status name, so the code decides.
    """
    status = error.status
    if isinstance(status, str) and status in _STATUS_NAMES:
        return cast(StatusName, status)
    code = http_code(error.code)
    if code is None or code < 400:
        return None
    return from_http_code(code)


def retry_delay_ms(error: APIError) -> float | None:
    """Delay the service asked for before retrying, in milliseconds.

    Reads the ``google.rpc.RetryInfo`` detail from the error body first and
    the ``Retry-After`` response header second. None when neither is present.
    """
    ms = _retry_info_ms(error.details)
    if ms is None:
        ms = retry_after_ms_from_error(error)
    return ms


def from_api_error(error: APIError) -> GenkitError:
    """GenkitError carrying the SDK error's status, retry delay, and the error itself as cause.

    Raises the original error when it carries neither a status name nor a
    failing HTTP code, so it stays unclassified rather than claiming a status.
    """
    status = status_for_api_error(error)
    if status is None:
        raise error
    response_metadata: ErrorResponseMetadata | None = None
    ms = retry_delay_ms(error)
    if ms is not None:
        response_metadata = {'retry_after_ms': ms}
    return GenkitError(
        status=status,
        message=error.message or str(error),
        cause=error,
        response_metadata=response_metadata,
    )


def _retry_info_ms(details: Any) -> float | None:  # noqa: ANN401
    """RetryInfo delay from an error body whose details sit under ``error`` or at the top level."""
    if not isinstance(details, dict):
        return None
    error = details.get('error')
    items = (error if isinstance(error, dict) else details).get('details')
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        type_name = item.get('@type')
        if isinstance(type_name, str) and type_name.endswith(_RETRY_INFO_TYPE):
            ms = _duration_ms(item.get('retryDelay'))
            if ms is not None:
                return ms
    return None


def _duration_ms(value: Any) -> float | None:  # noqa: ANN401
    """Milliseconds for a JSON proto Duration: ``'58s'``, ``'0.5s'``, or ``{'seconds': n, 'nanos': n}``."""
    ms: float
    if isinstance(value, str) and value.endswith('s'):
        try:
            ms = float(value[:-1]) * 1000
        except ValueError:
            return None
    elif isinstance(value, dict) and ('seconds' in value or 'nanos' in value):
        try:
            ms = float(value.get('seconds') or 0) * 1000 + float(value.get('nanos') or 0) / 1e6
        except (TypeError, ValueError):
            return None
    else:
        return None
    return ms if math.isfinite(ms) and ms >= 0 else None
