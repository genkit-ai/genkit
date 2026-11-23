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

"""Unit tests for status_types module."""

import pytest

from genkit._core._error import (
    StatusCodes,
    from_http_code,
    http_status_code,
)


def test_status_codes_values() -> None:
    """Tests that StatusCodes has correct values and can be used as ints."""
    assert StatusCodes.OK == 0
    assert StatusCodes.CANCELLED == 1
    assert StatusCodes.UNKNOWN == 2
    assert StatusCodes.INVALID_ARGUMENT == 3
    assert StatusCodes.DEADLINE_EXCEEDED == 4
    assert StatusCodes.NOT_FOUND == 5
    assert StatusCodes.ALREADY_EXISTS == 6
    assert StatusCodes.PERMISSION_DENIED == 7
    assert StatusCodes.UNAUTHENTICATED == 16
    assert StatusCodes.RESOURCE_EXHAUSTED == 8
    assert StatusCodes.FAILED_PRECONDITION == 9
    assert StatusCodes.ABORTED == 10
    assert StatusCodes.OUT_OF_RANGE == 11
    assert StatusCodes.UNIMPLEMENTED == 12
    assert StatusCodes.INTERNAL == 13
    assert StatusCodes.UNAVAILABLE == 14
    assert StatusCodes.DATA_LOSS == 15


def test_http_status_code_mapping() -> None:
    """Tests http_status_code function returns correct HTTP status codes."""
    assert http_status_code('OK') == 200
    assert http_status_code('CANCELLED') == 499
    assert http_status_code('UNKNOWN') == 500
    assert http_status_code('INVALID_ARGUMENT') == 400
    assert http_status_code('DEADLINE_EXCEEDED') == 504
    assert http_status_code('NOT_FOUND') == 404
    assert http_status_code('ALREADY_EXISTS') == 409
    assert http_status_code('PERMISSION_DENIED') == 403
    assert http_status_code('UNAUTHENTICATED') == 401
    assert http_status_code('RESOURCE_EXHAUSTED') == 429
    assert http_status_code('FAILED_PRECONDITION') == 400
    assert http_status_code('ABORTED') == 409
    assert http_status_code('OUT_OF_RANGE') == 400
    assert http_status_code('UNIMPLEMENTED') == 501
    assert http_status_code('INTERNAL') == 500
    assert http_status_code('UNAVAILABLE') == 503
    assert http_status_code('DATA_LOSS') == 500


def test_from_http_code() -> None:
    """Reverse map: shared codes pick the retry default; unmapped 5xx are INTERNAL."""
    assert from_http_code(200) == 'OK'
    assert from_http_code(400) == 'INVALID_ARGUMENT'
    assert from_http_code(408) == 'DEADLINE_EXCEEDED'
    assert from_http_code(401) == 'UNAUTHENTICATED'
    assert from_http_code(403) == 'PERMISSION_DENIED'
    assert from_http_code(404) == 'NOT_FOUND'
    assert from_http_code(409) == 'ABORTED'
    assert from_http_code(429) == 'RESOURCE_EXHAUSTED'
    assert from_http_code(499) == 'CANCELLED'
    assert from_http_code(500) == 'INTERNAL'
    assert from_http_code(501) == 'UNIMPLEMENTED'
    assert from_http_code(503) == 'UNAVAILABLE'
    assert from_http_code(504) == 'DEADLINE_EXCEEDED'
    assert from_http_code(418) == 'UNKNOWN'
    assert from_http_code(502) == 'INTERNAL'
    # 529 is Anthropic-only; the shared map treats it as an unmapped 5xx.
    assert from_http_code(529) == 'INTERNAL'


def test_http_status_code_invalid_input() -> None:
    """Tests http_status_code function with invalid input."""
    with pytest.raises(KeyError):
        http_status_code('INVALID_STATUS')  # type: ignore[arg-type]
