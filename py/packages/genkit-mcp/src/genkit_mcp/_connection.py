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

"""Lifecycle of a stdio connection to one MCP server."""

from typing import Any

from mcp.types import CallToolResult, Tool as McpTool

from genkit_mcp._config import McpStdioServerConfig

DEFAULT_REQUEST_TIMEOUT_MILLIS = 60_000
"""How long one MCP request may take before the SDK fails it."""


class McpConnection:
    """Owns the child process and session for one MCP server.

    The MCP SDK opens the transport and the session inside a task group, so both
    must be entered and exited by the same task. One connection is therefore held
    per event loop: the Dev UI reflection server runs its own loop on a separate
    thread and would otherwise await a future belonging to the main loop.
    """

    def __init__(
        self,
        config: McpStdioServerConfig,
        request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
    ) -> None:
        """Prepare a connection. No process starts until the first call.

        Args:
            config: How to launch the server.
            request_timeout_millis: How long one request may take before it
                fails. ``None`` waits forever.
        """
        raise NotImplementedError

    @property
    def is_terminal(self) -> bool:
        """Whether this connection is closed or failed and serves nothing more."""
        raise NotImplementedError

    async def list_tools(self) -> list[McpTool]:
        """List every tool the server offers, following pagination cursors.

        Returns:
            All tools, in server order.
        """
        raise NotImplementedError

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Invoke one tool on the server.

        Args:
            name: Tool name as the server lists it, without the Genkit prefix.
            arguments: Tool input, unvalidated.
            meta: Optional MCP ``_meta`` forwarded with the request.

        Returns:
            The raw result, for the caller to map.
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Shut the session down and reap the child process.

        Safe to call when never connected, and safe to call twice. A closed
        connection serves nothing more.
        """
        raise NotImplementedError
