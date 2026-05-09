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
from pydantic import ValidationError

from genkit._core._error import Status, StatusCodes, http_status_code


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


def test_status_immutability() -> None:
    """Tests that Status objects are immutable."""
    status = Status(name='OK')

    with pytest.raises(ValidationError):
        # pyrefly: ignore[read-only] - Intentionally testing immutability
        status.name = 'NOT_FOUND'

    with pytest.raises(ValidationError):
        # pyrefly: ignore[read-only] - Intentionally testing immutability
        status.message = 'New message'


def test_status_validation() -> None:
    """Tests that Status validates inputs correctly."""
    # Test invalid status name
    with pytest.raises(ValidationError):
        Status(name='INVALID_STATUS')  # type: ignore[arg-type]

    # Test with invalid type for name
    with pytest.raises(ValidationError):
        Status(name=123)  # type: ignore[arg-type]

    # Test with invalid type for message
    with pytest.raises(ValidationError):
        Status(name='OK', message=123)  # type: ignore[arg-type]

    # Test with extra fields
    with pytest.raises(ValidationError):
        Status(name='OK', extra_field='value')  # type: ignore[call-arg]


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


def test_http_status_code_invalid_input() -> None:
    """Tests http_status_code function with invalid input."""
    with pytest.raises(KeyError):
        http_status_code('INVALID_STATUS')  # type: ignore[arg-type]


def test_status_json_serialization() -> None:
    """Tests that Status objects can be serialized to JSON."""
    status = Status(name='NOT_FOUND', message='Resource not found')
    json_data = status.model_dump_json()
    assert '"name":"NOT_FOUND"' in json_data
    assert '"message":"Resource not found"' in json_data


def test_status_json_deserialization() -> None:
    """Tests that Status objects can be deserialized from JSON."""
    json_data = '{"name": "NOT_FOUND", "message": "Resource not found"}'
    status = Status.model_validate_json(json_data)
    assert status.name == 'NOT_FOUND'
    assert status.message == 'Resource not found'


def test_status_equality() -> None:
    """Tests Status equality comparison."""
    status1 = Status(name='OK')
    status2 = Status(name='OK')
    status3 = Status(name='NOT_FOUND')

    assert status1 == status2
    assert status1 != status3
