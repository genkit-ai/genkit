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

"""Integration tests for MCP client-server communication."""

import os
import pathlib
import sys
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import (
    AnyUrl,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListResourceTemplatesRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceRequestParams,
    ReadResourceResult,
    Resource,
    TextContent,
    TextResourceContents,
    Tool,
)

from genkit.blocks.resource import ResourceInput
from genkit.core.error import GenkitError

# Defer genkit imports to allow mocking. Type annotations help ty understand these are callable.
Genkit: Any = None
McpClient: Any = None
McpServerConfig: Any = None
create_mcp_host: Any = None
create_mcp_server: Any = None


def setup_mocks() -> None:
    """Set up mocks for testing."""
    global Genkit, McpClient, McpServerConfig, create_mcp_host, create_mcp_server

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
            McpClient as _McpClient,
            McpServerConfig as _McpServerConfig,
            create_mcp_host as _create_mcp_host,
            create_mcp_server as _create_mcp_server,
        )

        Genkit = _Genkit
        McpClient = _McpClient
        McpServerConfig = _McpServerConfig
        create_mcp_host = _create_mcp_host
        create_mcp_server = _create_mcp_server
    except ImportError:
        pass


@pytest.mark.asyncio
class TestClientServerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MCP client-server communication."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    async def test_client_can_list_server_tools(self) -> None:
        """Test that a client can list tools from a server."""
        # Create server with tools
        server_ai = Genkit()

        @server_ai.tool()
        def add(a: int, b: int) -> int:
            return a + b

        # Create client
        client = McpClient(name='test-client', config=McpServerConfig(command='echo', args=['test']))

        # Mock the session to return tools
        mock_session = AsyncMock()
        mock_tool = Tool(name='add', description='Add two numbers', inputSchema={'type': 'object'})
        mock_session.list_tools.return_value = ListToolsResult(tools=[mock_tool])
        client.session = mock_session

        # List tools
        tools = await client.list_tools()

        # Verify
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, 'add')

    async def test_client_can_call_server_tool(self) -> None:
        """Test that a client can call a tool on a server."""
        # Create client
        client = McpClient(name='test-client', config=McpServerConfig(command='echo'))

        # Mock the session
        mock_session = AsyncMock()
        mock_content = TextContent(type='text', text='8')
        mock_result = CallToolResult(content=[mock_content])

        mock_session.call_tool.return_value = mock_result
        client.session = mock_session

        # Call tool
        result = await client.call_tool('add', {'a': 5, 'b': 3})

        # Verify
        self.assertEqual(result, '8')
        mock_session.call_tool.assert_called_once_with('add', {'a': 5, 'b': 3})

    async def test_client_can_list_server_resources(self) -> None:
        """Test that a client can list resources from a server."""
        # Create client
        client = McpClient(name='test-client', config=McpServerConfig(command='echo'))

        # Mock the session
        mock_session = AsyncMock()
        mock_resource = Resource(name='config', uri=AnyUrl('app://config'), description='Configuration')

        mock_session.list_resources.return_value = ListResourcesResult(resources=[mock_resource])
        client.session = mock_session

        # List resources
        resources = await client.list_resources()

        # Verify
        self.assertEqual(len(resources), 1)
        self.assertEqual(resources[0].name, 'config')
        self.assertEqual(str(resources[0].uri), 'app://config')

    async def test_client_can_read_server_resource(self) -> None:
        """Test that a client can read a resource from a server."""
        # Create client
        client = McpClient(name='test-client', config=McpServerConfig(command='echo'))

        # Mock the session
        mock_session = AsyncMock()
        mock_result = ReadResourceResult(
            contents=[TextResourceContents(uri=AnyUrl('app://config'), mimeType='text/plain', text='Resource content')]
        )

        mock_session.read_resource.return_value = mock_result
        client.session = mock_session

        # Read resource
        result = await client.read_resource('app://config')

        # Verify
        self.assertIsNotNone(result)
        mock_session.read_resource.assert_called_once_with(AnyUrl('app://config'))

    async def test_host_manages_multiple_clients(self) -> None:
        """Test that a host can manage multiple clients."""
        # Create host with multiple servers
        config1 = McpServerConfig(command='server1')
        config2 = McpServerConfig(command='server2')

        host = create_mcp_host({'server1': config1, 'server2': config2})

        # Verify clients were created
        self.assertEqual(len(host.clients), 2)
        self.assertIn('server1', host.clients)
        self.assertIn('server2', host.clients)

    async def test_host_can_register_tools_from_multiple_servers(self) -> None:
        """Test that a host can register tools from multiple servers."""
        # Create host
        host = create_mcp_host({'server1': McpServerConfig(command='s1'), 'server2': McpServerConfig(command='s2')})

        # Mock sessions for both clients
        for client_name, client in host.clients.items():
            mock_session = AsyncMock()
            mock_tool = Tool(
                name=f'{client_name}_tool', description=f'Tool from {client_name}', inputSchema={'type': 'object'}
            )

            mock_session.list_tools.return_value = ListToolsResult(tools=[mock_tool])
            client.session = mock_session

        # Register tools
        ai = Genkit()
        await host.register_tools(ai)

        # Verify tools were registered
        # Each client should have registered one tool
        # Tool names should be prefixed with server name

    async def test_client_handles_disabled_server(self) -> None:
        """Test that a client handles disabled servers correctly."""
        # Create client with disabled config
        config = McpServerConfig(command='echo', disabled=True)
        client = McpClient(name='test-client', config=config)

        # Try to connect
        await client.connect()

        # Should not have a session
        self.assertIsNone(client.session)

    async def test_host_can_disable_and_enable_clients(self) -> None:
        """Test that a host can disable and enable clients."""
        host = create_mcp_host({'test': McpServerConfig(command='echo')})

        # Mock the client
        client = host.clients['test']
        client.session = AsyncMock()
        client.close = AsyncMock()
        client.connect = AsyncMock()

        # Disable
        await host.disable('test')
        self.assertTrue(client.config.disabled)

        # Enable
        await host.enable('test')
        self.assertFalse(client.config.disabled)


@pytest.mark.asyncio
class TestResourceIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests specifically for resource handling."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    async def test_end_to_end_resource_flow(self) -> None:
        """Test complete flow: define resource → expose via server → consume via client."""
        # This is a conceptual test showing the flow
        # In practice, we'd need actual MCP transport for true end-to-end

        # 1. Server side: Define resource
        server_ai = Genkit()
        server_ai.define_resource(
            name='config', uri='app://config', fn=lambda req: {'content': [{'text': 'config data'}]}
        )

        # 2. Create MCP server
        # Deferred import: must happen after setup_mocks() is called earlier in this test
        from genkit.plugins.mcp import McpServerOptions

        server = create_mcp_server(server_ai, McpServerOptions(name='test-server'))
        await server.setup()

        # 3. Verify server can list resources
        resources_result = await server.list_resources(ListResourcesRequest(method='resources/list'))
        self.assertEqual(len(resources_result.resources), 1)
        self.assertEqual(str(resources_result.resources[0].uri), 'app://config')

        # 4. Verify server can read resource
        request = ReadResourceRequest(
            method='resources/read', params=ReadResourceRequestParams(uri=AnyUrl('app://config'))
        )
        read_result = await server.read_resource(request)
        assert isinstance(read_result.contents[0], TextResourceContents)
        self.assertEqual(read_result.contents[0].text, 'config data')

    async def test_template_resource_matching(self) -> None:
        """Test that template resources match correctly."""
        server_ai = Genkit()

        def file_resource(req: ResourceInput) -> dict[str, list[dict[str, str]]]:
            uri = req.uri
            return {'content': [{'text': f'Contents of {uri}'}]}

        server_ai.define_resource(name='file', template='file://{+path}', fn=file_resource)

        # Create server
        # Deferred import: must happen after setup_mocks() is called earlier in this test
        from genkit.plugins.mcp import McpServerOptions

        server = create_mcp_server(server_ai, McpServerOptions(name='test-server'))
        await server.setup()

        # List templates
        templates_result = await server.list_resource_templates(
            ListResourceTemplatesRequest(method='resources/templates/list')
        )
        self.assertEqual(len(templates_result.resourceTemplates), 1)
        self.assertEqual(templates_result.resourceTemplates[0].uriTemplate, 'file://{+path}')

        # Read with different URIs
        for test_uri in ['file:///path/to/file.txt', 'file:///another/file.md', 'file:///deep/nested/path/doc.pdf']:
            request = ReadResourceRequest(
                method='resources/read', params=ReadResourceRequestParams(uri=AnyUrl(test_uri))
            )
            result = await server.read_resource(request)
            assert isinstance(result.contents[0], TextResourceContents)
            self.assertIn(test_uri, result.contents[0].text)


@pytest.mark.asyncio
class TestErrorHandling(unittest.IsolatedAsyncioTestCase):
    """Tests for error handling in client-server communication."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        setup_mocks()

    async def test_server_handles_missing_tool(self) -> None:
        """Test that server properly handles requests for non-existent tools."""
        server_ai = Genkit()

        @server_ai.tool()
        def existing_tool(x: int) -> int:
            return x

        # Deferred import: must happen after setup_mocks() is called earlier in this test
        from genkit.plugins.mcp import McpServerOptions

        server = create_mcp_server(server_ai, McpServerOptions(name='test-server'))
        await server.setup()

        # Try to call non-existent tool
        request = CallToolRequest(
            method='tools/call',
            params=CallToolRequestParams(name='nonexistent_tool', arguments={}),
        )

        with self.assertRaises(GenkitError) as context:
            await server.call_tool(request)

        self.assertIn('NOT_FOUND', str(context.exception.status))

    async def test_client_handles_connection_failure(self) -> None:
        """Test that client handles connection failures gracefully."""
        client = McpClient(name='test-client', config=McpServerConfig(command='nonexistent_command'))

        # Mock the connection to fail
        with patch('genkit.plugins.mcp.client.client.stdio_client') as mock_stdio:
            mock_stdio.side_effect = Exception('Connection failed')

            with self.assertRaisesRegex(Exception, 'Connection failed'):
                await client.connect()

            # Client should mark server as disabled
            self.assertTrue(client.config.disabled)


if __name__ == '__main__':
    unittest.main()
