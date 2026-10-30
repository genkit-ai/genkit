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

"""Tests for the caller-callback check shared by Retry and Fallback."""

from genkit_middleware._errors import caused_by_caller_callback

from genkit import GenkitError
from genkit.model import StreamingCallbackError


def _raised_from(exc: BaseException, cause: BaseException) -> BaseException:
    try:
        raise exc from cause
    except BaseException as raised:
        return raised


def test_caused_by_caller_callback_matches_the_bare_error() -> None:
    assert caused_by_caller_callback(StreamingCallbackError(ConnectionError('websocket closed')))


def test_caused_by_caller_callback_matches_through_a_plugin_rewrap() -> None:
    sink_failed = StreamingCallbackError(ConnectionError('websocket closed'))
    rewrapped = _raised_from(GenkitError(status='UNAVAILABLE', message='stream broke'), sink_failed)

    assert caused_by_caller_callback(rewrapped)


def test_caused_by_caller_callback_ignores_provider_failures() -> None:
    provider_failed = _raised_from(GenkitError(status='UNAVAILABLE', message='503'), ConnectionError('reset'))

    assert not caused_by_caller_callback(provider_failed)


def test_caused_by_caller_callback_ignores_implicit_context() -> None:
    try:
        try:
            raise StreamingCallbackError(ConnectionError('websocket closed'))
        except StreamingCallbackError:
            raise GenkitError(status='UNAVAILABLE', message='cleanup failed')  # noqa: B904
    except GenkitError as exc:
        assert exc.__context__ is not None
        assert not caused_by_caller_callback(exc)


def test_caused_by_caller_callback_stops_on_a_cause_cycle() -> None:
    first = GenkitError(status='UNAVAILABLE', message='a')
    second = GenkitError(status='UNAVAILABLE', message='b')
    first.__cause__ = second
    second.__cause__ = first

    assert not caused_by_caller_callback(first)
