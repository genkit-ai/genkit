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

"""Tests for the status check Retry and Fallback share."""

import pytest
from genkit_middleware._fallback import FallbackConfig
from genkit_middleware._retry import RetryConfig
from genkit_middleware._statuses import TRANSIENT_STATUSES, status_matches

from genkit import GenkitError


@pytest.mark.parametrize('unclassified', [True, False])
def test_genkit_error_matches_by_status_whatever_unclassified_says(unclassified: bool) -> None:
    """A GenkitError is decided by its status alone."""
    unavailable = GenkitError(status='UNAVAILABLE', message='provider is down')
    invalid = GenkitError(status='INVALID_ARGUMENT', message='bad temperature')

    assert status_matches(unavailable, ['UNAVAILABLE'], unclassified=unclassified) is True
    assert status_matches(invalid, ['UNAVAILABLE'], unclassified=unclassified) is False


@pytest.mark.parametrize('unclassified', [True, False])
def test_raw_exception_returns_unclassified(unclassified: bool) -> None:
    """A raw exception has no status, so the caller's `unclassified` decides."""
    assert status_matches(ConnectionError('reset'), ['INTERNAL'], unclassified=unclassified) is unclassified


def test_default_statuses_share_the_transient_list() -> None:
    """Retry defaults to the transient list; Fallback adds NOT_FOUND and UNIMPLEMENTED on top."""
    assert RetryConfig().statuses == list(TRANSIENT_STATUSES)
    assert FallbackConfig().statuses == [*TRANSIENT_STATUSES, 'NOT_FOUND', 'UNIMPLEMENTED']


def test_default_statuses_are_fresh_lists() -> None:
    """Editing one config's statuses doesn't change another config's defaults."""
    first = RetryConfig()
    first.statuses.append('NOT_FOUND')

    assert RetryConfig().statuses == list(TRANSIENT_STATUSES)
