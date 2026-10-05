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

"""Tests for McpStdioServerConfig validation and redaction."""

from typing import Any

import pytest
from genkit_mcp._config import McpStdioServerConfig
from pydantic import ValidationError

SECRET = 'sk-distinctive-secret-7f3a'


def test_repr_omits_env_values() -> None:
    """The env mapping never appears in repr or str."""
    config = McpStdioServerConfig(command='npx', env={'API_KEY': SECRET})

    assert SECRET not in repr(config)
    assert SECRET not in str(config)


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'command': 42, 'env': {'API_KEY': SECRET}}, id='bad-sibling-field'),
        pytest.param({'command': 'npx', 'env': {'API_KEY': SECRET, 'PORT': 8080}}, id='bad-env-entry'),
        pytest.param({'command': 'npx', 'enviroment': {'API_KEY': SECRET}}, id='misspelled-env-key'),
    ],
)
def test_validation_errors_omit_env_values(raw: dict[str, Any]) -> None:
    """A rejected config does not echo a secret in the error text."""
    with pytest.raises(ValidationError) as excinfo:
        McpStdioServerConfig.model_validate(raw)

    assert SECRET not in str(excinfo.value)
    assert SECRET not in repr(excinfo.value)


def test_unknown_key_fails_validation() -> None:
    """A misspelled key fails instead of launching the server without it."""
    with pytest.raises(ValidationError, match='enviroment'):
        McpStdioServerConfig.model_validate({'command': 'npx', 'enviroment': {'API_KEY': SECRET}})


def test_claude_desktop_stdio_block_passes_through() -> None:
    """An mcpServers entry carrying type stdio validates unchanged."""
    block = {
        'type': 'stdio',
        'command': 'npx',
        'args': ['-y', '@modelcontextprotocol/server-filesystem', '/srv/books'],
        'env': {'API_KEY': SECRET},
        'cwd': '/srv/books',
    }

    config = McpStdioServerConfig.model_validate(block)

    assert config.type == 'stdio'
    assert config.env == {'API_KEY': SECRET}


def test_non_stdio_type_fails_validation() -> None:
    """A remote transport block is rejected rather than run as a command."""
    with pytest.raises(ValidationError):
        McpStdioServerConfig.model_validate({'type': 'http', 'command': 'npx'})
