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

"""Tests for define_mcp_client argument validation."""

import pytest
from genkit_mcp._client import define_mcp_client
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import McpConnection

from genkit import Genkit


@pytest.mark.parametrize('timeout', [0, -1])
def test_define_mcp_client_rejects_a_timeout_that_is_not_positive(timeout: int) -> None:
    """A zero timeout would fail every request at once instead of meaning no timeout."""
    with pytest.raises(ValueError, match='request_timeout_millis'):
        define_mcp_client(Genkit(), 'everything', command='npx', request_timeout_millis=timeout)


@pytest.mark.parametrize('timeout', [0, -1])
def test_connection_rejects_a_timeout_that_is_not_positive(timeout: int) -> None:
    """The connection guards the timeout too, for callers that build one directly."""
    with pytest.raises(ValueError, match='request_timeout_millis'):
        McpConnection(McpStdioServerConfig(command='npx'), request_timeout_millis=timeout)
