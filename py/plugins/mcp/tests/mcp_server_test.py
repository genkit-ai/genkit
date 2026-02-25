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

"""MCP Server Tests.

Mirrors the functionality of js/plugins/mcp/tests/server_test.ts
Tests tools, prompts, and resources exposed via MCP server.
"""

import os
import pathlib
import sys
import unittest
from typing import Any, cast

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(os.path.join(pathlib.Path(__file__).parent, '../src')).resolve()))


import pytest
from mcp.types import (
    AnyUrl,
    CallToolRequest,
    CallToolRequestParams,
    GetPromptRequest,
    GetPromptRequestParams,
    ListPromptsRequest,
    ListResourcesRequest,
    ListResourceTemplatesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    ReadResourceRequestParams,
    TextContent,
    TextResourceContents,
    Tool,
)

from genkit.ai import Genkit
from genkit.blocks.resource import matches_uri_template
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.plugins.mcp import McpServerOptions, create_mcp_server


@pytest.mark.asyncio
class TestMcpServer(unittest.IsolatedAsyncioTestCase):
    """Test MCP server functionality - mirrors JS server_test.ts."""

    def setUp(self) -> None:
        """Set up test fixtures before each test."""
        self.ai = Genkit()

        # Define test tool
        @self.ai.tool(description='test tool')
        def test_tool(input: dict[str, str]) -> str:
            foo = input.get('foo', '')
            return f'yep {{"foo":"{foo}"}}'

        # Define test prompt
        self.ai.define_prompt(name='testPrompt', model='test-model', prompt='prompt says: {{input}}')

        # Define test resource with fixed URI
        self.ai.define_resource(
            name='testResources', uri='my://resource', fn=lambda req: {'content': [{'text': 'my resource'}]}
        )

        # Define test resource with template
        self.ai.define_resource(
            name='testTmpl',
            template='file://{+path}',
            fn=lambda req: {'content': [{'text': f'file contents for {req.uri}'}]},
        )

        # Create MCP server
        self.server = create_mcp_server(self.ai, McpServerOptions(name='test-server', version='0.0.1'))

    async def asyncSetUp(self) -> None:
        """Async setup - initialize server."""
        await self.server.setup()

    # ===== TOOL TESTS =====

    async def test_list_tools(self) -> None:
        """Test listing tools - mirrors JS 'should list tools'."""
        result = await self.server.list_tools(ListToolsRequest(method='tools/list'))

        # Verify we have the test tool
        self.assertEqual(len(result.tools), 1)
        tool = result.tools[0]

        self.assertEqual(tool.name, 'test_tool')
        self.assertEqual(tool.description, 'test tool')
        # Ensure it is a Tool and input_schema is present
        assert isinstance(tool, Tool)
        assert tool.inputSchema is not None
        self.assertIsNotNone(tool.inputSchema)

    async def test_call_tool(self) -> None:
        """Test calling a tool - mirrors JS 'should call the tool'."""
        # Create mock request
        request = CallToolRequest(
            method='tools/call',
            params=CallToolRequestParams(name='test_tool', arguments={'foo': 'bar'}),
        )

        result = await self.server.call_tool(request)

        # Verify response
        self.assertEqual(len(result.content), 1)
        assert isinstance(result.content[0], TextContent)
        self.assertEqual(result.content[0].type, 'text')
        self.assertEqual(result.content[0].text, 'yep {"foo":"bar"}')

    # ===== PROMPT TESTS =====

    async def test_list_prompts(self) -> None:
        """Test listing prompts - mirrors JS 'should list prompts'."""
        result = await self.server.list_prompts(ListPromptsRequest(method='prompts/list'))

        # Verify we have the test prompt
        prompt_names = [p.name for p in result.prompts]
        self.assertIn('testPrompt', prompt_names)

    async def test_get_prompt(self) -> None:
        """Test rendering a prompt - mirrors JS 'should render prompt'."""
        # Create mock request
        request = GetPromptRequest(
            method='prompts/get',
            params=GetPromptRequestParams(name='testPrompt', arguments={'input': 'hello'}),
        )

        result = await self.server.get_prompt(request)

        # Verify response
        self.assertIsNotNone(result.messages)
        self.assertGreater(len(result.messages), 0)

        # Check message content
        message = result.messages[0]
        self.assertEqual(message.role, 'user')
        self.assertEqual(message.content.type, 'text')
        assert isinstance(message.content, TextContent)
        self.assertIn('prompt says: hello', message.content.text)

    # ===== RESOURCE TESTS =====

    async def test_list_resources(self) -> None:
        """Test listing resources - mirrors JS 'should list resources'."""
        result = await self.server.list_resources(ListResourcesRequest(method='resources/list'))

        # Verify we have the fixed URI resource
        self.assertEqual(len(result.resources), 1)
        resource = result.resources[0]

        self.assertEqual(resource.name, 'testResources')
        self.assertEqual(str(resource.uri), 'my://resource')

    async def test_list_resource_templates(self) -> None:
        """Test listing resource templates - mirrors JS 'should list templates'."""
        result = await self.server.list_resource_templates(
            ListResourceTemplatesRequest(method='resources/templates/list')
        )

        # Verify we have the template resource
        self.assertEqual(len(result.resourceTemplates), 1)
        template = result.resourceTemplates[0]

        self.assertEqual(template.name, 'testTmpl')
        self.assertEqual(template.uriTemplate, 'file://{+path}')

    async def test_read_resource(self) -> None:
        """Test reading a resource - mirrors JS 'should read resource'."""
        # Create mock request
        request = ReadResourceRequest(
            method='resources/read', params=ReadResourceRequestParams(uri=AnyUrl('my://resource'))
        )

        result = await self.server.read_resource(request)

        # Verify response
        self.assertEqual(len(result.contents), 1)
        content = result.contents[0]
        assert isinstance(content, TextResourceContents)

        self.assertEqual(str(content.uri), 'my://resource')
        self.assertEqual(content.text, 'my resource')

    async def test_read_template_resource(self) -> None:
        """Test reading a template resource."""
        # Create mock request
        # Create mock request
        request = ReadResourceRequest(
            method='resources/read', params=ReadResourceRequestParams(uri=AnyUrl('file:///path/to/file.txt'))
        )

        result = await self.server.read_resource(request)

        # Verify response
        self.assertEqual(len(result.contents), 1)
        content = result.contents[0]
        assert isinstance(content, TextResourceContents)

        self.assertEqual(str(content.uri), 'file:///path/to/file.txt')
        self.assertIn('file contents for file:///path/to/file.txt', content.text)

    # ===== ADDITIONAL TESTS =====

    async def test_server_initialization(self) -> None:
        """Test that server initializes correctly."""
        self.assertIsNotNone(self.server)
        self.assertEqual(self.server.options.name, 'test-server')
        self.assertEqual(self.server.options.version, '0.0.1')
        self.assertTrue(self.server.actions_resolved)

    async def test_server_has_all_action_types(self) -> None:
        """Test that server has tools, prompts, and resources."""
        self.assertGreater(len(self.server.tool_actions), 0)
        self.assertGreater(len(self.server.prompt_actions), 0)
        self.assertGreater(len(self.server.resource_actions), 0)

    async def test_tool_not_found(self) -> None:
        """Test calling a non-existent tool."""
        request = CallToolRequest(
            method='tools/call',
            params=CallToolRequestParams(name='nonexistent_tool', arguments={}),
        )

        with self.assertRaises(GenkitError) as context:
            await self.server.call_tool(request)

        self.assertEqual(context.exception.status, 'NOT_FOUND')

    async def test_prompt_not_found(self) -> None:
        """Test getting a non-existent prompt."""
        request = GetPromptRequest(
            method='prompts/get',
            params=GetPromptRequestParams(name='nonexistent_prompt', arguments={}),
        )

        with self.assertRaises(GenkitError) as context:
            await self.server.get_prompt(request)

        self.assertEqual(context.exception.status, 'NOT_FOUND')

    async def test_resource_not_found(self) -> None:
        """Test reading a non-existent resource."""
        request = ReadResourceRequest(
            method='resources/read',
            params=ReadResourceRequestParams(uri=AnyUrl('nonexistent://resource')),
        )

        with self.assertRaises(GenkitError) as context:
            await self.server.read_resource(request)

        self.assertEqual(context.exception.status, 'NOT_FOUND')


# Additional test class for resource-specific functionality
@pytest.mark.asyncio
class TestResourceFunctionality(unittest.IsolatedAsyncioTestCase):
    """Test resource-specific functionality."""

    async def test_resource_registration_with_fixed_uri(self) -> None:
        """Test registering a resource with fixed URI."""
        ai = Genkit()

        action = ai.define_resource(
            name='test_resource', uri='test://resource', fn=lambda req: {'content': [{'text': 'test'}]}
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.RESOURCE)
        assert action.metadata is not None
        metadata = cast(dict[str, Any], action.metadata)
        resource_meta = cast(dict[str, Any], metadata['resource'])
        self.assertEqual(resource_meta['uri'], 'test://resource')

    async def test_resource_registration_with_template(self) -> None:
        """Test registering a resource with URI template."""
        ai = Genkit()

        action = ai.define_resource(
            name='file', template='file://{+path}', fn=lambda req: {'content': [{'text': 'file content'}]}
        )

        self.assertIsNotNone(action)
        self.assertEqual(action.kind, ActionKind.RESOURCE)
        assert action.metadata is not None
        metadata = cast(dict[str, Any], action.metadata)
        resource_meta = cast(dict[str, Any], metadata['resource'])
        self.assertEqual(resource_meta['template'], 'file://{+path}')

    async def test_resource_requires_uri_or_template(self) -> None:
        """Test that resource requires either uri or template."""
        ai = Genkit()

        with self.assertRaises(ValueError) as context:
            ai.define_resource(name='invalid', fn=lambda req: {'content': []})

        self.assertIn('uri', str(context.exception).lower())
        self.assertIn('template', str(context.exception).lower())

    async def test_uri_template_matching(self) -> None:
        """Test URI template matching."""
        # Test exact match
        result = matches_uri_template('file://{+path}', 'file:///home/user/doc.txt')
        assert result is not None
        self.assertIn('path', result)

        # Test no match
        result = matches_uri_template('file://{path}', 'http://example.com')
        self.assertIsNone(result)

        # Test multiple parameters
        result = matches_uri_template('user://{id}/posts/{post_id}', 'user://123/posts/456')
        assert result is not None
        self.assertEqual(result['id'], '123')
        self.assertEqual(result['post_id'], '456')


if __name__ == '__main__':
    unittest.main()
