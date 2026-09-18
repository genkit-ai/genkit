# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A small subprocess MCP server used by connection lifecycle tests."""

import asyncio
import os

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolRequest,
    ErrorData,
    ListToolsRequest,
    ListToolsResult,
    ServerResult,
    TextContent,
    Tool,
)

app = Server('fake-server')

TOOLS = [
    Tool(name='echo', description='Echo input.', inputSchema={'type': 'object'}),
    Tool(name='second', description='Another tool.', inputSchema={'type': 'object'}),
    Tool(name='wait', description='Wait before replying.', inputSchema={'type': 'object'}),
    Tool(name='list_requests', description='Count tools/list requests.', inputSchema={'type': 'object'}),
    Tool(name='who_called', description='Report the client that connected.', inputSchema={'type': 'object'}),
    Tool(name='stall', description='Never reply.', inputSchema={'type': 'object'}),
]

PAGE_SIZE = int(os.environ.get('MCP_FAKE_PAGE_SIZE', '2'))

list_requests = 0


@app.list_tools()
async def list_tools(req: ListToolsRequest) -> ListToolsResult:
    # The SDK passes None when it refreshes its own tool cache, and that refresh
    # must see every tool or call_tool skips input validation.
    if req is None:
        return ListToolsResult(tools=TOOLS)

    global list_requests
    list_requests += 1

    if os.environ.get('MCP_FAKE_APP_ERROR') and list_requests == 1:
        # -32000 is the JSON-RPC implementation-defined server error range, which
        # a healthy server may use and the SDK also synthesises on transport death.
        raise McpError(ErrorData(code=CONNECTION_CLOSED, message='application-level failure'))

    if os.environ.get('MCP_FAKE_STUCK_CURSOR'):
        return ListToolsResult(tools=TOOLS[:1], nextCursor='stuck')

    if os.environ.get('MCP_FAKE_ENDLESS_CURSOR'):
        return ListToolsResult(tools=TOOLS[:1], nextCursor=str(list_requests))

    start = int(req.params.cursor) if req.params is not None and req.params.cursor is not None else 0
    end = start + PAGE_SIZE
    return ListToolsResult(tools=TOOLS[start:end], nextCursor=str(end) if end < len(TOOLS) else None)


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, object]) -> list[TextContent]:
    if name == 'wait':
        # Must stay under the SDK's PROCESS_TERMINATION_TIMEOUT of 2.0 seconds.
        await asyncio.sleep(0.25)
    if name == 'stall':
        await asyncio.Event().wait()
    if name == 'list_requests':
        return [TextContent(type='text', text=str(list_requests))]
    if name == 'who_called':
        client = app.request_context.session.client_params
        assert client is not None
        return [TextContent(type='text', text=f'{client.clientInfo.name} {client.clientInfo.version}')]
    return [TextContent(type='text', text=str(arguments.get('message', name)))]


if os.environ.get('MCP_FAKE_APP_ERROR_CALL'):
    # The call_tool decorator turns handler exceptions into isError results, so
    # putting a JSON-RPC error code on the wire needs the raw handler.
    async def failing_call(req: CallToolRequest) -> ServerResult:
        raise McpError(ErrorData(code=CONNECTION_CLOSED, message='application-level failure'))

    app.request_handlers[CallToolRequest] = failing_call


async def main() -> None:
    if os.environ.get('MCP_FAKE_BAD_UTF8'):
        # Bypasses the SDK's text stream deliberately: this is a byte a strict
        # decoder cannot read, not a message.
        os.write(1, b'\xff\xfe not utf-8\n')
    pid_file = os.environ.get('MCP_FAKE_PID_FILE')
    if pid_file:
        await anyio.Path(pid_file).write_text(str(os.getpid()), encoding='utf-8')
    if os.environ.get('MCP_FAKE_STALL'):
        # Never answers initialize, so the client stays in its connecting state.
        await asyncio.Event().wait()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
