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

"""What a plugin author gets back from plugin_api.provider_error."""

import pytest

from genkit import GenkitError
from genkit._core import _error as error_mod
from genkit._core._error import PublicError, get_callable_json, get_http_status
from genkit.plugin_api import StatusName, provider_error

# 2023-11-14 22:13:20 GMT
_NOW = 1_700_000_000.0


@pytest.mark.parametrize(
    ('http_status', 'status'),
    [
        (400, 'INVALID_ARGUMENT'),
        (404, 'NOT_FOUND'),
        (408, 'DEADLINE_EXCEEDED'),
        (429, 'RESOURCE_EXHAUSTED'),
        (503, 'UNAVAILABLE'),
        (599, 'INTERNAL'),
    ],
)
def test_provider_error_maps_http_status_to_genkit_status(http_status: int, status: StatusName) -> None:
    """`provider_error(e, http_status=...)` returns a GenkitError with the matching Genkit status."""
    error = provider_error(RuntimeError('boom'), http_status=http_status)

    assert isinstance(error, GenkitError)
    assert error.status == status
    assert error.original_message == 'boom'
    assert error.response_metadata is None


@pytest.mark.parametrize('http_status', [413, 418, 422])
def test_provider_error_unmapped_4xx_is_unknown(http_status: int) -> None:
    """`provider_error(e, http_status=413)` (or 418, 422) returns a GenkitError with status UNKNOWN."""
    error = provider_error(RuntimeError('too large'), http_status=http_status)

    assert isinstance(error, GenkitError)
    assert error.status == 'UNKNOWN'
    assert error.original_message == 'too large'
    assert error.response_metadata is None


def test_provider_error_explicit_status_wins_over_http_status() -> None:
    """`provider_error(e, http_status=500, status='UNAVAILABLE')` returns UNAVAILABLE, not INTERNAL."""
    error = provider_error(RuntimeError('overloaded'), http_status=500, status='UNAVAILABLE')

    assert isinstance(error, GenkitError)
    assert error.status == 'UNAVAILABLE'
    assert error.original_message == 'overloaded'
    assert error.response_metadata is None


@pytest.mark.parametrize('http_status', [None, -1])
def test_provider_error_with_no_usable_status_is_unknown(http_status: int | None) -> None:
    """`provider_error(e)` with no status, or `http_status=-1`, returns a GenkitError with status UNKNOWN."""
    error = provider_error(ConnectionError('reset'), http_status=http_status)

    assert isinstance(error, GenkitError)
    assert error.status == 'UNKNOWN'
    assert error.original_message == 'reset'
    assert error.response_metadata is None


@pytest.mark.parametrize('header_name', ['Retry-After', 'retry-after'])
def test_provider_error_reads_retry_after_seconds_from_headers(header_name: str) -> None:
    """`provider_error(e, http_status=429, headers={'Retry-After': '30'})` sets retry_after_ms to 30000."""
    error = provider_error(RuntimeError('slow down'), http_status=429, headers={header_name: '30'})

    assert isinstance(error, GenkitError)
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.original_message == 'slow down'
    assert error.response_metadata == {'retry_after_ms': 30000.0}


@pytest.mark.parametrize(
    ('retry_after', 'expected_ms'),
    [
        ('Tue, 14 Nov 2023 22:13:25 GMT', 5000.0),
        ('Tue, 14 Nov 2023 22:13:15 GMT', 0.0),
    ],
    ids=['future_date', 'past_date'],
)
def test_provider_error_reads_retry_after_http_date(
    monkeypatch: pytest.MonkeyPatch, retry_after: str, expected_ms: float
) -> None:
    """A `Retry-After` HTTP date sets retry_after_ms to the time until then, or 0 if it already passed."""
    monkeypatch.setattr(error_mod.time, 'time', lambda: _NOW)

    error = provider_error(RuntimeError('slow down'), http_status=429, headers={'Retry-After': retry_after})

    assert isinstance(error, GenkitError)
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.original_message == 'slow down'
    assert error.response_metadata == {'retry_after_ms': expected_ms}


def test_provider_error_explicit_retry_after_ms_wins_over_header() -> None:
    """`provider_error(e, headers={'Retry-After': '30'}, retry_after_ms=1500)` sets retry_after_ms to 1500."""
    error = provider_error(
        RuntimeError('slow down'),
        http_status=429,
        headers={'Retry-After': '30'},
        retry_after_ms=1500,
    )

    assert isinstance(error, GenkitError)
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.original_message == 'slow down'
    assert error.response_metadata == {'retry_after_ms': 1500}


@pytest.mark.parametrize(
    'headers',
    [None, {}, {'Retry-After': ''}, {'Retry-After': 'soon'}, {'Retry-After': '-5'}, {'Retry-After': 'inf'}],
    ids=['no_headers', 'no_retry_after', 'blank', 'words', 'negative', 'infinite'],
)
def test_provider_error_ignores_missing_or_malformed_retry_after(headers: dict[str, str] | None) -> None:
    """A missing or unreadable `Retry-After` still returns the GenkitError, just without retry_after_ms."""
    error = provider_error(RuntimeError('slow down'), http_status=429, headers=headers)

    assert isinstance(error, GenkitError)
    assert error.status == 'RESOURCE_EXHAUSTED'
    assert error.original_message == 'slow down'
    assert error.response_metadata is None


def test_provider_error_keeps_provider_message_and_cause() -> None:
    """The GenkitError keeps the provider's text and error as its cause; `message=` replaces only the text."""
    cause = RuntimeError('API key not valid')

    error = provider_error(cause, http_status=401)
    renamed = provider_error(cause, http_status=401, message='gemini rejected the key')

    assert isinstance(error, GenkitError)
    assert error.status == 'UNAUTHENTICATED'
    assert error.original_message == 'API key not valid'
    assert error.cause is cause
    assert error.__cause__ is cause
    assert str(error) == 'UNAUTHENTICATED: API key not valid'
    assert renamed.status == 'UNAUTHENTICATED'
    assert renamed.original_message == 'gemini rejected the key'
    assert renamed.cause is cause
    assert renamed.__cause__ is cause
    assert str(renamed) == 'UNAUTHENTICATED: gemini rejected the key: API key not valid'


def test_provider_error_is_internal_error_over_http() -> None:
    """A served flow raising `provider_error(e, http_status=401)` answers 500 Internal Error, not 401."""
    error = provider_error(RuntimeError('API key not valid'), http_status=401)

    assert isinstance(error, GenkitError)
    assert not isinstance(error, PublicError)
    assert error.status == 'UNAUTHENTICATED'
    assert get_http_status(error) == 500
    assert get_callable_json(error) == {'message': 'Internal Error', 'status': 'INTERNAL'}


def test_provider_error_names_the_type_when_the_error_has_no_text() -> None:
    """A bare TimeoutError() has an empty str(); the message falls back to its type name."""
    error = provider_error(TimeoutError(), status='DEADLINE_EXCEEDED')

    assert str(error) == 'DEADLINE_EXCEEDED: TimeoutError'
    assert error.original_message == 'TimeoutError'


def test_genkit_error_does_not_repeat_a_cause_already_in_the_message() -> None:
    """A message that embeds the cause's text does not get it appended a second time."""
    cause = RuntimeError('Rate exceeded')

    error = GenkitError(status='RESOURCE_EXHAUSTED', message='bedrock converse failed: Rate exceeded', cause=cause)

    assert str(error) == 'RESOURCE_EXHAUSTED: bedrock converse failed: Rate exceeded'


def test_genkit_error_skips_a_cause_with_no_text() -> None:
    """A cause whose str() is empty adds no dangling ': '."""
    error = GenkitError(status='DEADLINE_EXCEEDED', message='bedrock converse failed', cause=TimeoutError())

    assert str(error) == 'DEADLINE_EXCEEDED: bedrock converse failed'
