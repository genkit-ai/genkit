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

"""Tests for google-genai API error translation."""

from types import SimpleNamespace
from typing import Any

import pytest
from genkit_google_genai.models._errors import from_api_error, retry_delay_ms, status_for_api_error
from google.genai.errors import APIError, ClientError, ServerError

from genkit import GenkitError

RETRY_INFO_TYPE = 'type.googleapis.com/google.rpc.RetryInfo'


def _api_error(
    code: int,
    *,
    status: str | None = None,
    message: str | None = 'boom',
    details: list[Any] | None = None,
    response: Any = None,  # noqa: ANN401
) -> APIError:
    """Build an SDK error with the standard ``{'error': {...}}`` body shape."""
    body: dict[str, Any] = {'code': code}
    if message is not None:
        body['message'] = message
    if status is not None:
        body['status'] = status
    if details is not None:
        body['details'] = details
    cls: type[APIError] = ClientError if 400 <= code < 500 else ServerError if code >= 500 else APIError
    return cls(code, {'error': body}, response)


def _retry_info(delay: Any) -> dict[str, Any]:  # noqa: ANN401
    return {'@type': RETRY_INFO_TYPE, 'retryDelay': delay}


def _response(retry_after: str) -> SimpleNamespace:
    """Stand-in for an HTTP response exposing only a Retry-After header."""
    return SimpleNamespace(headers={'retry-after': retry_after})


# --- status_for_api_error ---------------------------------------------------


def test_status_name_wins_over_http_code() -> None:
    """A recognised status name in the body takes precedence over the HTTP code."""
    assert status_for_api_error(_api_error(400, status='FAILED_PRECONDITION')) == 'FAILED_PRECONDITION'
    assert status_for_api_error(_api_error(429, status='UNAVAILABLE')) == 'UNAVAILABLE'


@pytest.mark.parametrize(
    ('code', 'expected'),
    [
        (400, 'INVALID_ARGUMENT'),
        (401, 'UNAUTHENTICATED'),
        (403, 'PERMISSION_DENIED'),
        (404, 'NOT_FOUND'),
        (408, 'DEADLINE_EXCEEDED'),
        (409, 'ABORTED'),
        (418, 'UNKNOWN'),
        (429, 'RESOURCE_EXHAUSTED'),
        (499, 'CANCELLED'),
        (500, 'INTERNAL'),
        (501, 'UNIMPLEMENTED'),
        (502, 'INTERNAL'),
        (503, 'UNAVAILABLE'),
        (504, 'DEADLINE_EXCEEDED'),
        (599, 'INTERNAL'),
    ],
)
def test_http_code_fallback(code: int, expected: str) -> None:
    """Without a status name the HTTP code decides."""
    assert status_for_api_error(_api_error(code)) == expected


def test_reason_phrase_status_falls_back_to_code() -> None:
    """A non-JSON body leaves the HTTP reason phrase in status; it is not a status name."""
    error = ServerError(503, {'message': '<html>overloaded</html>', 'status': 'Service Unavailable'})
    assert error.status == 'Service Unavailable'
    assert status_for_api_error(error) == 'UNAVAILABLE'


def test_ok_status_is_not_used_for_an_error() -> None:
    """A body claiming status OK on an error response falls back to the HTTP code."""
    assert status_for_api_error(_api_error(500, status='OK')) == 'INTERNAL'


def test_status_name_without_http_code() -> None:
    """An error with a status name but no HTTP code is classified by the name."""
    error = APIError(0, {'error': {'status': 'UNAVAILABLE', 'message': 'stream closed'}})
    assert error.code is None
    assert status_for_api_error(error) == 'UNAVAILABLE'


def test_neither_status_nor_code_is_unclassified() -> None:
    """An error with neither a status name nor an HTTP code has no status."""
    error = APIError(0, {'error': {'message': 'stream closed'}})
    assert error.code is None
    assert status_for_api_error(error) is None


def test_non_dict_body() -> None:
    """A body the SDK could not parse still maps by code and carries no retry delay."""
    error = ClientError(400, 'not json')
    assert status_for_api_error(error) == 'INVALID_ARGUMENT'
    assert retry_delay_ms(error) is None


# --- retry_delay_ms ---------------------------------------------------------


@pytest.mark.parametrize(
    ('delay', 'expected_ms'),
    [
        ('58s', 58_000.0),
        ('0.5s', 500.0),
        ({'seconds': 2, 'nanos': 500_000_000}, 2_500.0),
        ({'nanos': 250_000_000}, 250.0),
        ({'seconds': '3'}, 3_000.0),
    ],
)
def test_retry_info_delay_forms(delay: Any, expected_ms: float) -> None:  # noqa: ANN401
    """RetryInfo delays are read in both the string and the seconds/nanos form."""
    assert retry_delay_ms(_api_error(429, details=[_retry_info(delay)])) == expected_ms


def test_retry_info_in_flat_body() -> None:
    """Details are found when the body is the error object itself rather than wrapped in ``error``."""
    error = APIError(429, {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': [_retry_info('2s')]})
    assert retry_delay_ms(error) == 2_000.0


def test_retry_info_skips_other_details() -> None:
    """Details of other types are ignored; a RetryInfo after them is still found."""
    details = [
        {'@type': 'type.googleapis.com/google.rpc.ErrorInfo', 'reason': 'RATE_LIMIT_EXCEEDED'},
        _retry_info('1s'),
    ]
    assert retry_delay_ms(_api_error(429, details=details)) == 1_000.0


@pytest.mark.parametrize(
    'delay',
    ['abc', 'nans', 'infs', '-1s', '', 12, None, {'seconds': 'x'}, {}, {'seconds': float('inf')}],
)
def test_retry_info_malformed_delay_is_ignored(delay: Any) -> None:  # noqa: ANN401
    """Malformed delays never raise and produce no delay."""
    assert retry_delay_ms(_api_error(429, details=[_retry_info(delay)])) is None


def test_retry_after_header_is_the_fallback() -> None:
    """A Retry-After header is read when the body has no RetryInfo."""
    assert retry_delay_ms(_api_error(503, response=_response('30'))) == 30_000.0


def test_retry_info_wins_over_header() -> None:
    """When both are present the RetryInfo detail is used."""
    error = _api_error(429, details=[_retry_info('58s')], response=_response('30'))
    assert retry_delay_ms(error) == 58_000.0


def test_no_response_and_no_details() -> None:
    """An error without a response object or details has no delay."""
    assert retry_delay_ms(_api_error(503)) is None


def test_response_without_headers() -> None:
    """A response object with no headers attribute is tolerated."""
    assert retry_delay_ms(_api_error(503, response=object())) is None


# --- from_api_error ---------------------------------------------------------


def test_from_api_error_carries_status_message_and_cause() -> None:
    """The GenkitError reports the mapped status, the service message, and the original error."""
    error = _api_error(503, message='overloaded')
    wrapped = from_api_error(error)
    assert isinstance(wrapped, GenkitError)
    assert wrapped.status == 'UNAVAILABLE'
    assert wrapped.original_message == 'overloaded'
    assert wrapped.cause is error
    assert wrapped.response_metadata is None


def test_from_api_error_message_falls_back_to_the_error_text() -> None:
    """Without a service message the SDK error's own text is used."""
    error = ClientError(400, 'not json')
    assert error.message is None
    assert from_api_error(error).original_message == str(error)


def test_from_api_error_attaches_retry_delay() -> None:
    """Retry information is exposed through response_metadata."""
    error = _api_error(429, details=[_retry_info('58s')])
    wrapped = from_api_error(error)
    assert wrapped.status == 'RESOURCE_EXHAUSTED'
    assert wrapped.response_metadata == {'retry_after_ms': 58_000.0}


def test_from_api_error_reraises_unclassifiable_error() -> None:
    """An error with neither a status name nor an HTTP code is raised as-is."""
    error = APIError(0, {'error': {'message': 'stream closed'}})
    with pytest.raises(APIError) as raised:
        from_api_error(error)
    assert raised.value is error
