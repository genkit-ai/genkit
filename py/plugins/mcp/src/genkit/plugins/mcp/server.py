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
# distributed under the License.
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""MCP Server implementation for exposing Genkit actions via Model Context Protocol."""

from typing import Any, cast

import structlog
from pydantic import BaseModel

from genkit.ai import Genkit
from genkit.blocks.resource import matches_uri_template
from genkit.core.action import Action
from genkit.core.action.types import ActionKind
from genkit.core.error import GenkitError
from genkit.core.schema import to_json_schema
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    AnyUrl,
    CallToolRequest,
    CallToolResult,
    GetPromptRequest,
    GetPromptResult,
    ListPromptsRequest,
    ListPromptsResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListResourceTemplatesRequest,
    ListResourceTemplatesResult,
    ListToolsRequest,
    ListToolsResult,
    Prompt,
    PromptArgument,
    ReadResourceRequest,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)

from .util import (
    to_mcp_prompt_arguments,
    to_mcp_prompt_message,
    to_mcp_resource_contents,
    to_mcp_tool_result,
)

logger = structlog.get_logger(__name__)


def _get_resource_meta(metadata: dict[str, object]) -> dict[str, Any]:
    """Extract resource metadata from action metadata with proper typing."""
    raw = metadata.get('resource', {})
    return cast(dict[str, Any], raw) if isinstance(raw, dict) else {}


class McpServerOptions(BaseModel):
    """Options for creating an MCP server.

    Attributes:
        name: The name of the MCP server.
        version: The version of the server (default: "1.0.0").
    """

    name: str
    version: str = '1.0.0'


class McpServer:
    """Exposes Genkit tools, prompts, and resources as an MCP server.

    This class wraps a Genkit instance and makes its registered actions
    (tools, prompts, resources) available to MCP clients via the Model Context Protocol.
    """

    def __init__(self, ai: Genkit, options: McpServerOptions) -> None:
        """Initialize the MCP server.

        Args:
            ai: The Genkit instance whose actions will be exposed.
            options: Configuration options for the MCP server.
        """
        self.ai = ai
        self.options = options
        self.server: Server | None = None
        self.actions_resolved = False
        self.tool_actions: list[Action] = []
        self.prompt_actions: list[Action] = []
        self.resource_actions: list[Action] = []
        self.tool_actions_map: dict[str, Action] = {}
        self.prompt_actions_map: dict[str, Action] = {}
        self.resource_uri_map: dict[str, Action] = {}
        self.resource_templates: list[tuple[str, Action]] = []

    async def setup(self) -> None:
        """Initialize the MCP server and register request handlers.

        This method sets up the MCP Server instance, registers all request handlers,
        and resolves all actions from the Genkit registry. It's idempotent and can
        be called multiple times safely.
        """
        if self.actions_resolved:
            return

        # Create MCP Server instance
        self.server = Server(
            self.options.name,
            version=self.options.version,
        )

        # Register request handlers using decorators
        # MCP library type stubs don't match actual decorator signatures
        self.server.list_tools()(self.list_tools)
        self.server.call_tool()(self.call_tool)
        self.server.list_prompts()(self.list_prompts)
        # pyrefly: ignore[bad-argument-type] - MCP decorator type stub mismatch
        self.server.get_prompt()(self.get_prompt)
        self.server.list_resources()(self.list_resources)
        # pyrefly: ignore[bad-argument-type] - MCP decorator type stub mismatch
        self.server.list_resource_templates()(self.list_resource_templates)
        # pyrefly: ignore[bad-argument-type] - MCP decorator type stub mismatch
        self.server.read_resource()(self.read_resource)

        # Resolve all actions from Genkit registry
        # We need the actual Action objects, not just serializable dicts
        self.tool_actions = []
        self.prompt_actions = []
        self.resource_actions = []

        # Get all actions from the registry
        # We use the internal _entries for local actions and plugins
        with self.ai.registry._lock:
            for kind, entries in self.ai.registry._entries.items():
                for _name, action in entries.items():
                    if kind == ActionKind.TOOL:
                        self.tool_actions.append(action)
                        self.tool_actions_map[action.name] = action
                    elif kind == ActionKind.PROMPT:
                        self.prompt_actions.append(action)
                        self.prompt_actions_map[action.name] = action
                    elif kind == ActionKind.RESOURCE:
                        self.resource_actions.append(action)
                        metadata = action.metadata or {}
                        resource_meta = _get_resource_meta(metadata)
                        if resource_meta.get('uri'):
                            self.resource_uri_map[resource_meta['uri']] = action
                        if resource_meta.get('template'):
                            self.resource_templates.append((resource_meta['template'], action))

        # Also get actions from plugins that might not be in _entries yet
        # (though most plugins register them in _entries during initialization)
        plugin_action_metadata = await self.ai.registry.list_actions()
        for action_meta in plugin_action_metadata:
            kind, name = action_meta.kind, action_meta.name
            action = await self.ai.registry.resolve_action(kind, name)
            if action:
                if kind == ActionKind.TOOL and action not in self.tool_actions:
                    self.tool_actions.append(action)
                    self.tool_actions_map[action.name] = action
                elif kind == ActionKind.PROMPT and action not in self.prompt_actions:
                    self.prompt_actions.append(action)
                    self.prompt_actions_map[action.name] = action
                elif kind == ActionKind.RESOURCE and action not in self.resource_actions:
                    self.resource_actions.append(action)
                    metadata = action.metadata or {}
                    resource_meta = _get_resource_meta(metadata)
                    if resource_meta.get('uri'):
                        self.resource_uri_map[resource_meta['uri']] = action
                    if resource_meta.get('template'):
                        self.resource_templates.append((resource_meta['template'], action))

        self.actions_resolved = True

        logger.info(
            'MCP Server initialized',
            tools=len(self.tool_actions),
            prompts=len(self.prompt_actions),
            resources=len(self.resource_actions),
        )

    async def list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """Handle MCP requests to list available tools.

        Args:
            request: The MCP ListToolsRequest.

        Returns:
            ListToolsResult containing all registered Genkit tools.
        """
        await self.setup()

        tools: list[Tool] = []
        for action in self.tool_actions:
            # Get tool definition
            input_schema = to_json_schema(action.input_schema) if action.input_schema else {'type': 'object'}

            tools.append(
                Tool(
                    name=action.name,
                    description=action.description or '',
                    inputSchema=input_schema,
                )
            )

        return ListToolsResult(tools=tools)

    async def call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle MCP requests to call a specific tool.

        Args:
            request: The MCP CallToolRequest containing tool name and arguments.

        Returns:
            CallToolResult with the tool execution result.

        Raises:
            GenkitError: If the requested tool is not found.
        """
        await self.setup()

        # Find the tool action
        tool = self.tool_actions_map.get(request.params.name)

        if not tool:
            raise GenkitError(
                status='NOT_FOUND', message=f"Tried to call tool '{request.params.name}' but it could not be found."
            )

        # Execute the tool
        result = await tool.arun(request.params.arguments)
        result = result.response

        # Convert result to MCP format
        return CallToolResult(content=to_mcp_tool_result(result))

    async def list_prompts(self, request: ListPromptsRequest) -> ListPromptsResult:
        """Handle MCP requests to list available prompts.

        Args:
            request: The MCP ListPromptsRequest.

        Returns:
            ListPromptsResult containing all registered Genkit prompts.
        """
        await self.setup()

        prompts: list[Prompt] = []
        for action in self.prompt_actions:
            # Convert input schema to MCP prompt arguments
            input_schema = to_json_schema(action.input_schema) if action.input_schema else None
            arguments = to_mcp_prompt_arguments(input_schema) if input_schema else None

            prompts.append(
                Prompt(
                    name=action.name,
                    description=action.description or '',
                    arguments=[PromptArgument(**arg) for arg in arguments] if arguments else None,
                )
            )

        return ListPromptsResult(prompts=prompts)

    async def get_prompt(self, request: GetPromptRequest) -> GetPromptResult:
        """Handle MCP requests to get (render) a specific prompt.

        Args:
            request: The MCP GetPromptRequest containing prompt name and arguments.

        Returns:
            GetPromptResult with the rendered prompt messages.

        Raises:
            GenkitError: If the requested prompt is not found.
        """
        await self.setup()

        # Find the prompt action
        prompt = self.prompt_actions_map.get(request.params.name)

        if not prompt:
            raise GenkitError(
                status='NOT_FOUND',
                message=f"[MCP Server] Tried to call prompt '{request.params.name}' but it could not be found.",
            )

        # Execute the prompt
        result = await prompt.arun(request.params.arguments)
        result = result.response

        # Convert messages to MCP format
        messages = [to_mcp_prompt_message(msg) for msg in result.messages]

        return GetPromptResult(description=prompt.description, messages=messages)

    async def list_resources(self, request: ListResourcesRequest) -> ListResourcesResult:
        """Handle MCP requests to list available resources with fixed URIs.

        Args:
            request: The MCP ListResourcesRequest.

        Returns:
            ListResourcesResult containing resources with fixed URIs.
        """
        await self.setup()

        resources: list[Resource] = []
        for action in self.resource_actions:
            metadata = action.metadata or {}
            resource_meta = _get_resource_meta(metadata)

            # Only include resources with fixed URIs (not templates)
            if resource_meta.get('uri'):
                resources.append(
                    Resource(
                        name=action.name,
                        description=action.description or '',
                        uri=AnyUrl(resource_meta['uri']),
                    )
                )

        return ListResourcesResult(resources=resources)

    async def list_resource_templates(self, request: ListResourceTemplatesRequest) -> ListResourceTemplatesResult:
        """Handle MCP requests to list available resource templates.

        Args:
            request: The MCP ListResourceTemplatesRequest.

        Returns:
            ListResourceTemplatesResult containing resources with URI templates.
        """
        await self.setup()

        templates: list[ResourceTemplate] = []
        for action in self.resource_actions:
            metadata = action.metadata or {}
            resource_meta = _get_resource_meta(metadata)

            # Only include resources with templates
            if resource_meta.get('template'):
                templates.append(
                    ResourceTemplate(
                        name=action.name,
                        description=action.description or '',
                        uriTemplate=resource_meta['template'],
                    )
                )

        return ListResourceTemplatesResult(resourceTemplates=templates)

    async def read_resource(self, request: ReadResourceRequest) -> ReadResourceResult:
        """Handle MCP requests to read a specific resource.

        Args:
            request: The MCP ReadResourceRequest containing the resource URI.

        Returns:
            ReadResourceResult with the resource content.

        Raises:
            GenkitError: If no matching resource is found.
        """
        await self.setup()

        uri = request.params.uri

        # Check for exact URI match
        uri_str = str(uri)
        resource = self.resource_uri_map.get(uri_str)

        # Check for template match if not found by exact URI
        if not resource:
            for template, action in self.resource_templates:
                if matches_uri_template(template, uri_str):
                    resource = action
                    break

        if not resource:
            raise GenkitError(status='NOT_FOUND', message=f"Tried to call resource '{uri}' but it could not be found.")

        # Execute the resource action (uri_str is already a string)
        result = await resource.arun({'uri': uri_str})
        result = result.response

        # Convert content to MCP format
        content = result.get('content', []) if isinstance(result, dict) else result.content
        contents = to_mcp_resource_contents(uri_str, content)

        return ReadResourceResult(contents=contents)

    async def start(self, transport: object = None) -> None:
        """Start the MCP server with the specified transport.

        Args:
            transport: Optional MCP transport instance. If not provided,
                      a StdioServerTransport will be created and used.
        """
        await self.setup()

        if not transport:
            async with stdio_server() as (read, write):
                assert self.server is not None
                await self.server.run(read, write, self.server.create_initialization_options())
        else:
            # Connect the transport
            async with transport as (read, write):  # type: ignore[union-attr]
                assert self.server is not None
                await self.server.run(read, write, self.server.create_initialization_options())

        logger.debug(f"[MCP Server] MCP server '{self.options.name}' started successfully.")


# Schema types from mcp.types
ListToolsRequestSchema = ListToolsRequest
CallToolRequestSchema = CallToolRequest
ListPromptsRequestSchema = ListPromptsRequest
GetPromptRequestSchema = GetPromptRequest
ListResourcesRequestSchema = ListResourcesRequest
ListResourceTemplatesRequestSchema = ListResourceTemplatesRequest
ReadResourceRequestSchema = ReadResourceRequest


def create_mcp_server(ai: Genkit, options: McpServerOptions) -> McpServer:
    """Create an MCP server based on the supplied Genkit instance.

    All tools, prompts, and resources will be automatically converted to MCP compatibility.

    Args:
        ai: Your Genkit instance with registered tools, prompts, and resources.
        options: Configuration metadata for the server.

    Returns:
        GenkitMcpServer instance.

    Example:
        >>> from genkit.ai import Genkit
        >>> from genkit.plugins.mcp import create_mcp_server, McpServerOptions
        >>>
        >>> ai = Genkit()
        >>>
        >>> # Define some tools and resources
        >>> @ai.tool()
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> ai.define_resource(
        ...     name='my_resource',
        ...     uri='my://resource',
        ...     fn=lambda req: {'content': [{'text': 'resource content'}]},
        ... )
        >>>
        >>> # Create and start MCP server
        >>> server = create_mcp_server(ai, McpServerOptions(name='my-server'))
        >>> await server.start()
    """
    return McpServer(ai, options)
