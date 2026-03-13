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


"""Tests for MCP host."""

import os
import pathlib
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mcp.types import Tool

# Defer genkit imports to allow mocking. Type annotations help ty understand these are callable.
Genkit: Any = None
McpServerConfig: Any = None
create_mcp_host: Any = None


def setup_mocks() -> None:
    """Set up mocks for testing."""
    global Genkit, McpServerConfig, create_mcp_host

    # Add test directory to path for fakes
    if pathlib.Path(__file__).parent not in sys.path:
        sys.path.insert(0, str(pathlib.Path(__file__).parent))

    # Add src directory to path if not installed
    src_path = str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, '../src')).resolve())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        # Deferred import: mock_mcp_modules must be called before importing genkit.plugins.mcp
        from fakes import mock_mcp_modules

        mock_mcp_modules()

        # Deferred import: these imports must happen after mock_mcp_modules() is called
        from genkit.ai import Genkit as _Genkit
        from genkit.plugins.mcp import (
            McpServerConfig as _McpServerConfig,
            create_mcp_host as _create_mcp_host,
        )

        Genkit = _Genkit
        McpServerConfig = _McpServerConfig
        create_mcp_host = _create_mcp_host
    except ImportError:
        pass


class TestMcpHost(unittest.IsolatedAsyncioTestCase):
    """Tests for MCP host."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    async def test_connect_and_register(self) -> None:
        """Test connect and register."""
        # Setup configs
        config1 = McpServerConfig(command='echo')
        config2 = McpServerConfig(url='http://localhost:8000')

        host = create_mcp_host({'server1': config1, 'server2': config2})

        # Mock clients within host
        with patch('genkit.plugins.mcp.client.client.McpClient.connect', new_callable=AsyncMock) as mock_connect:
            await host.start()
            self.assertEqual(mock_connect.call_count, 2)

        # Mock session for registration
        host.clients['server1'].session = AsyncMock()
        host.clients['server1'].session = AsyncMock()
        tool1 = Tool(name='tool1', description='tool desc', inputSchema={'type': 'object'})
        host.clients['server1'].session.list_tools.return_value.tools = [tool1]

        ai = MagicMock(spec=Genkit)
        ai.registry = MagicMock()

        await host.register_tools(ai)

        # Verify tool registration
        ai.registry.register_action.assert_called()
        call_args = ai.registry.register_action.call_args[1]
        self.assertIn('server1/tool1', call_args['name'])
