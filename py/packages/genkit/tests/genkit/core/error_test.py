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

"""Unit tests for the error module."""

from unittest.mock import MagicMock

import pytest

from genkit import ErrorResponseMetadata
from genkit._core import _error as error_mod
from genkit._core._error import (
    GenkitError,
    PublicError,
    ReflectionError,
    get_callable_json,
    get_error_stack,
    get_http_status,
    parse_retry_after_ms,
    wrap_http_error,
)


def test_genkit_error() -> None:
    error = GenkitError(
        status='INVALID_ARGUMENT',
        message='Test message',
        details={'extra_msg': 'Test detail'},
        source='test_source',
    )
    assert error.original_message == 'Test message'
    assert error.http_code == 400
    assert error.status == 'INVALID_ARGUMENT'
    assert error.details['extra_msg'] == 'Test detail'
    assert error.source == 'test_source'
    assert str(error) == 'test_source: INVALID_ARGUMENT: Test message'

    error_no_source = GenkitError(status='INTERNAL', message='Test message 2')
    assert str(error_no_source) == 'INTERNAL: Test message 2'

    # When wrapping another exception the cause should appear in str(...) too,
    # so the model and any plain ``f"{e}"`` log line see the real reason.
    wrapped = GenkitError(
        status='INTERNAL',
        message='Error while running action read_file',
        cause=ValueError("File not found: 'workspace/foo.py'"),
    )
    assert str(wrapped) == ("INTERNAL: Error while running action read_file: File not found: 'workspace/foo.py'")
    assert wrapped.original_message == 'Error while running action read_file'


def test_genkit_error_to_json() -> None:
    # NOT_FOUND is a valid gRPC-style status (maps to HTTP 404).
    error = GenkitError(status='NOT_FOUND', message='Resource not found', details={'id': 123})
    serializable = error.to_serializable()
    assert isinstance(serializable, ReflectionError)
    assert serializable.code == 5
    assert serializable.message == 'Resource not found'
    assert serializable.details is not None
    assert serializable.details.model_dump()['id'] == 123


def test_genkit_error_response_metadata_is_in_process_only() -> None:
    response_metadata: ErrorResponseMetadata = {
        'retry_after_ms': 1500.5,
        'headers': {'retry-after': '1.5005'},
    }
    error = GenkitError(
        status='RESOURCE_EXHAUSTED',
        message='Rate limited',
        response_metadata=response_metadata,
    )

    assert error.response_metadata == response_metadata
    assert 'response_metadata' not in error.to_callable_serializable().model_dump()
    assert 'response_metadata' not in error.to_serializable().model_dump()


def test_public_error() -> None:
    error = PublicError(
        status='UNAUTHENTICATED',
        message='Please log in',
        details={'extra_msg': 'Session expired'},
    )
    assert error.status == 'UNAUTHENTICATED'
    assert error.original_message == 'Please log in'
    assert error.details['extra_msg'] == 'Session expired'


def test_get_http_status() -> None:
    genkit_error = GenkitError(status='PERMISSION_DENIED', message='No access')
    assert get_http_status(genkit_error) == 403

    non_genkit_error = ValueError('Some other error')
    assert get_http_status(non_genkit_error) == 500


def test_get_callable_json() -> None:
    genkit_error = GenkitError(status='INVALID_ARGUMENT', message='Oops')
    json_data = get_callable_json(genkit_error)
    assert isinstance(json_data, dict)
    assert json_data['status'] == 'INVALID_ARGUMENT'
    assert json_data['message'] == 'Oops'
    assert 'details' in json_data

    non_genkit_error = TypeError('Type error')
    json_data = get_callable_json(non_genkit_error)
    assert isinstance(json_data, dict)
    assert json_data['status'] == 'INTERNAL'
    assert json_data['message'] == 'Type error'
    assert 'details' in json_data


def test_get_error_stack() -> None:
    try:
        raise ValueError('Example Error')
    except ValueError as e:
        tb = get_error_stack(e)
        assert tb == ''


def test_wrap_http_error_classifies_status() -> None:
    cause = RuntimeError('bad request')
    error = wrap_http_error(cause, status_code=400)
    assert error.status == 'INVALID_ARGUMENT'
    assert error.cause is cause
    assert error.original_message == 'bad request'


def test_wrap_http_error_marks_503_unavailable() -> None:
    """A 503 must stay retryable — not collapse to INTERNAL."""
    cause = RuntimeError('overloaded')
    error = wrap_http_error(cause, status_code=503)
    assert error.status == 'UNAVAILABLE'
    assert error.cause is cause


def test_wrap_http_error_coerces_string_status_code() -> None:
    """Some SDKs leave the code as a string; still classify a real 503."""
    cause = RuntimeError('overloaded')
    error = wrap_http_error(cause, status_code='503')
    assert error.status == 'UNAVAILABLE'


@pytest.mark.parametrize('status_code', [None, 'nope', 0, -1, 200, 301])
def test_wrap_http_error_leaves_missing_status_unclassified(status_code: object) -> None:
    """No HTTP failure status means retry still sees the raw error."""
    cause = RuntimeError('model failed')
    with pytest.raises(RuntimeError) as raised:
        wrap_http_error(cause, status_code=status_code)
    assert raised.value is cause


def test_wrap_http_error_marks_408_deadline_exceeded() -> None:
    """A request timeout is transient — retry should wait and try again."""
    cause = RuntimeError('request timeout')
    error = wrap_http_error(cause, status_code=408)
    assert error.status == 'DEADLINE_EXCEEDED'
    assert error.cause is cause


def test_wrap_http_error_reads_retry_after() -> None:
    """Retry should wait the provider delay, not come back in a second."""

    class FakeResponse:
        headers = {'retry-after': '60'}

    class FakeError(RuntimeError):
        def __init__(self) -> None:
            super().__init__('rate limited')
            self.response = FakeResponse()

    error = wrap_http_error(FakeError(), status_code=429, message='rate limited')
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.response_metadata == {'retry_after_ms': 60000.0}
    assert error.to_callable_serializable().message == 'rate limited'


def test_callable_wire_uses_original_message_when_cause_is_set() -> None:
    """The callable wire shows the provider text, not the SDK repr."""
    error = GenkitError(
        status='UNAVAILABLE',
        message='overloaded',
        cause=RuntimeError('APIError(503 UNAVAILABLE)'),
    )
    assert get_callable_json(error)['message'] == 'overloaded'


@pytest.mark.parametrize(
    ('value', 'expected_ms'),
    [
        ('2', 2000.0),
        (' 1.5 ', 1500.0),
        ('0', 0.0),
    ],
)
def test_parse_retry_after_delay_seconds(value: str, expected_ms: float) -> None:
    """Parse whole, fractional, and zero delay-seconds values."""
    assert parse_retry_after_ms(value) == expected_ms


@pytest.mark.parametrize('value', ['', '   ', 'not-a-delay'])
def test_parse_retry_after_rejects_blank_and_malformed_values(value: str) -> None:
    """Do not attach metadata for blank or malformed header values."""
    assert parse_retry_after_ms(value) is None


@pytest.mark.parametrize('value', ['inf', 'Infinity', 'nan', '1e999', '1e307'])
def test_parse_retry_after_rejects_non_finite_delays(value: str) -> None:
    """Reject delays that are, or scale to, non-finite milliseconds."""
    assert parse_retry_after_ms(value) is None


def test_parse_retry_after_future_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert a future HTTP-date to a relative millisecond delay."""
    monkeypatch.setattr(error_mod.time, 'time', lambda: 1_700_000_000.0)

    assert parse_retry_after_ms('Tue, 14 Nov 2023 22:13:25 GMT') == 5000.0


def test_parse_retry_after_past_http_date(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clamp a past HTTP-date delay to zero."""
    monkeypatch.setattr(error_mod.time, 'time', lambda: 1_700_000_000.0)

    assert parse_retry_after_ms('Tue, 14 Nov 2023 22:13:15 GMT') == 0.0


def test_parse_retry_after_returns_none_on_timestamp_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore platform timestamp failures for parseable dates."""
    retry_at = MagicMock()
    retry_at.timestamp.side_effect = OSError
    monkeypatch.setattr(error_mod, 'parsedate_to_datetime', lambda _: retry_at)

    assert parse_retry_after_ms('Thu, 01 Jan 1601 00:00:00') is None
