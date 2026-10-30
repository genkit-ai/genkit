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

"""Tests for the status each MCP client error reports."""

import pytest
from genkit_mcp._errors import (
    _ClientError,
    _ConnectionClosedError,
    _ConnectionFailedError,
    _ProtocolError,
    _RequestTimeoutError,
    _ServerError,
)

from genkit import GenkitError


@pytest.mark.parametrize(
    ('error_type', 'status'),
    [
        (_ConnectionClosedError, 'FAILED_PRECONDITION'),
        (_ConnectionFailedError, 'UNAVAILABLE'),
        (_RequestTimeoutError, 'DEADLINE_EXCEEDED'),
        (_ServerError, 'UNKNOWN'),
        (_ProtocolError, 'INTERNAL'),
    ],
)
def test_each_error_is_a_genkit_error_with_its_status(error_type: type[_ClientError], status: str) -> None:
    """Callers tell the failures apart by status, as they do for any GenkitError."""
    error = error_type("MCP connection to 'npx' failed")

    assert isinstance(error, GenkitError)
    assert error.status == status
    assert "MCP connection to 'npx' failed" in str(error)


def test_no_error_is_a_runtime_error() -> None:
    """The SDK's bare RuntimeError is a tool result, so these must never be mistaken for one."""
    assert not issubclass(_ClientError, RuntimeError)


def test_a_status_can_be_given_for_one_error() -> None:
    assert _ClientError('misuse', status='FAILED_PRECONDITION').status == 'FAILED_PRECONDITION'
