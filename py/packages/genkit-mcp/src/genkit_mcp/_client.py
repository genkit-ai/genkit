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

"""Genkit client for a single MCP server."""

import asyncio
import threading
import weakref
from typing import Any

from mcp.types import CallToolResult

from genkit import Genkit
from genkit._core._dap import DynamicActionProvider
from genkit.plugin_api import Action
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import DEFAULT_REQUEST_TIMEOUT_MILLIS, McpConnection
from genkit_mcp._errors import McpConnectionClosedError
from genkit_mcp._tools import mcp_tool_name, mcp_tool_to_action, validate_provider_name, validate_tool_prefix


class McpClient:
    """One MCP server, its connections, and the tools it serves as Genkit actions.

    A client holds no Genkit instance. It builds actions and manages the child
    process; registering those actions is :func:`define_mcp_client`, which is
    also what fills :attr:`dynamic_action_provider`.

    Obtain instances from :func:`create_mcp_client` or :func:`define_mcp_client`
    rather than constructing directly.
    """

    dynamic_action_provider: DynamicActionProvider | None
    """The provider serving this client's tools, or ``None`` when unregistered.

    :func:`define_mcp_client` assigns it. Dropping its cache is how this client
    tells Genkit that a listing it handed out no longer holds.
    """

    def __init__(
        self,
        name: str,
        server: McpStdioServerConfig,
        *,
        tool_prefix: str | None = None,
        request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
    ) -> None:
        """Build a client. No process starts until its tools are listed.

        Args:
            name: Client name, which :func:`define_mcp_client` registers as the
                provider name used in tool selectors.
            server: How to launch the server.
            tool_prefix: Namespace for this server's tool names. Defaults to ``name``.
            request_timeout_millis: How long one MCP request may take before it
                fails. ``None`` waits forever.

        Raises:
            ValueError: If ``name`` cannot appear in a tool selector, or the
                prefix in use cannot appear in a model-facing tool name.
        """
        validate_provider_name(name)
        prefix = name if tool_prefix is None else tool_prefix
        validate_tool_prefix(prefix)
        self.dynamic_action_provider = None
        self._name = name
        self._server = server
        self._tool_prefix = prefix
        self._request_timeout_millis = request_timeout_millis
        self._closed = False
        self._connections: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, McpConnection] = (
            weakref.WeakKeyDictionary()
        )
        self._connections_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Name this client was given.

        Once registered it is also the provider name in tool selectors, for
        example ``everything`` in ``everything:tool/*``.
        """
        return self._name

    @property
    def tool_prefix(self) -> str:
        """Namespace every tool name of this server carries.

        The name supplied as ``tool_prefix``, or this client's name.
        """
        return self._tool_prefix

    def tool_name(self, tool: str) -> str:
        """Return the Genkit name of one of this server's tools.

        Use it to build a selector for ``generate`` without assembling the prefix
        by hand::

            tools = [f'{client.name}:tool/{client.tool_name("opening_hours")}']

        Args:
            tool: Tool name as the server lists it, without any prefix.

        Returns:
            The namespaced name, for example ``bookshop_opening_hours``.
        """
        return mcp_tool_name(self._tool_prefix, tool)

    async def get_active_tools(self) -> list[Action[Any, Any]]:
        """List the tools this server currently offers, as Genkit actions.

        Every call lists from the server, so the answer is what the server serves
        now whether or not this client is registered, and ``action.name`` is the
        namespaced name a model is shown. A disabled server offers none.

        The actions are unregistered: a model reaches them through a dynamic
        action provider serving them, which :func:`define_mcp_client` sets up.

        Returns:
            One action per tool, in server order.

        Raises:
            McpConnectionClosedError: If this client is closed.
            McpConnectionFailedError: If the server cannot be started.
        """
        if self._server.disabled:
            return []
        tools = await self._connection().list_tools()
        return [mcp_tool_to_action(tool, self._tool_prefix, self._call_tool) for tool in tools]

    async def restart(self) -> None:
        """Reconnect to the server, replacing any connection this client has.

        The calling loop reconnects and re-lists before this returns, so a server
        which still cannot be started says so here rather than on the next
        generate. Any other loop reconnects when it next asks for a tool.

        Raises:
            McpConnectionClosedError: If this client is closed.
            McpConnectionFailedError: If the server cannot be started.
        """
        with self._connections_lock:
            self._refuse_when_closed()
        await self._disconnect_all()
        self._invalidate_listing()
        await self.get_active_tools()

    async def close(self) -> None:
        """Disconnect from the server, stop its process, and keep it that way.

        A connection belongs to the event loop that opened it, so each one is
        shut down on its own loop. A loop which has already stopped cancelled its
        connection on the way out, taking the server process with it.

        Closing is final, as it is for a file or an ``httpx.AsyncClient``. A
        provider registered for this client stays registered and its tools now
        fail as closed. Use :meth:`restart` to connect again.
        """
        with self._connections_lock:
            self._closed = True
        await self._disconnect_all()
        self._invalidate_listing()

    def _invalidate_listing(self) -> None:
        """Drop the registered listing of this client's tools, if it has one."""
        if self.dynamic_action_provider is not None:
            self.dynamic_action_provider.invalidate_cache()

    async def _disconnect_all(self) -> None:
        """Shut every connection down on the loop which opened it."""
        current = asyncio.get_running_loop()
        with self._connections_lock:
            connections = list(self._connections.items())
            self._connections.clear()
        for loop, connection in connections:
            if loop is current:
                await connection.close()
            elif loop.is_running():
                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(connection.close(), loop))

    def _refuse_when_closed(self) -> None:
        """Raise if this client has been closed. Call under the connections lock."""
        if self._closed:
            raise McpConnectionClosedError(f'MCP client {self._name!r} is closed')

    def _connection(self) -> McpConnection:
        """Return the calling loop's connection, replacing one which has ended."""
        loop = asyncio.get_running_loop()
        with self._connections_lock:
            self._refuse_when_closed()
            existing = self._connections.get(loop)
            if existing is not None and not existing.is_terminal:
                return existing
            connection = McpConnection(self._server, request_timeout_millis=self._request_timeout_millis)
            self._connections[loop] = connection
            return connection

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None,
    ) -> CallToolResult:
        """Invoke a tool on the connection belonging to the calling event loop."""
        try:
            return await self._connection().call_tool(name, arguments, meta)
        except Exception:
            # Any failure can mean the listing is stale, and re-listing costs one
            # round trip, so drop the cache rather than classify the error.
            self._invalidate_listing()
            raise


def create_mcp_client(
    name: str,
    server: McpStdioServerConfig,
    *,
    tool_prefix: str | None = None,
    request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
) -> McpClient:
    """Build a client for one MCP server without registering it with Genkit.

    Nothing reaches a registry, so no selector resolves to this server's tools and
    no Genkit instance is needed. Use :meth:`McpClient.get_active_tools` to obtain
    them as actions and serve them from a dynamic action provider of your own.
    Connection is lazy: the child process starts on the first tool listing.

    Reach for :func:`define_mcp_client` instead to expose the server's tools as
    ``<name>:tool/*`` straight away.

    Args:
        name: Client name, carried by its tool names and its errors.
        server: How to launch the server.
        tool_prefix: Namespace for this server's tool names. Defaults to ``name``.
        request_timeout_millis: How long one MCP request may take before it
            fails. ``None`` waits forever.

    Returns:
        A handle for listing tools, restarting and disconnecting.

    Raises:
        ValueError: If ``name`` cannot appear in a tool selector, or the prefix in
            use cannot appear in a model-facing tool name.
    """
    return McpClient(
        name,
        server,
        tool_prefix=tool_prefix,
        request_timeout_millis=request_timeout_millis,
    )


def define_mcp_client(
    ai: Genkit,
    name: str,
    server: McpStdioServerConfig,
    *,
    tool_prefix: str | None = None,
    cache_ttl_millis: int | None = None,
    request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
) -> McpClient:
    """Register an MCP server's tools with Genkit.

    The server's tools become ``tool.v2`` actions behind a dynamic action provider
    named ``name``. Select them from ``generate`` as ``<name>:tool/*`` for all of
    them, or ``<name>:tool/<name>_<tool>`` for one. Connection is lazy: the child
    process starts on the first tool listing.

    This is a function rather than a Genkit plugin because the registry rewrites a
    plugin's action names to ``<plugin>/<action>``, and a provider name cannot
    contain a slash.

    Args:
        ai: The Genkit instance to register with.
        name: Provider name, used in tool selectors.
        server: How to launch the server.
        tool_prefix: Namespace for this server's tool names. Defaults to ``name``.
        cache_ttl_millis: How long a tool listing stays fresh. Defaults to the
            core dynamic action provider default.
        request_timeout_millis: How long one MCP request may take before it
            fails. ``None`` waits forever.

    Returns:
        A handle for restarting and disconnecting, carrying the registered
        provider as :attr:`McpClient.dynamic_action_provider`.

    Raises:
        ValueError: If ``name`` cannot appear in a tool selector, or the prefix in
            use cannot appear in a model-facing tool name.
    """
    client = McpClient(
        name,
        server,
        tool_prefix=tool_prefix,
        request_timeout_millis=request_timeout_millis,
    )

    async def list_tools() -> dict[str, list[Action[Any, Any]]]:
        """Provide this client's tools to the dynamic action provider."""
        return {'tool': await client.get_active_tools()}

    client.dynamic_action_provider = ai.define_dynamic_action_provider(
        name,
        list_tools,
        description=f'Tools of the MCP server {name}.',
        cache_ttl_millis=cache_ttl_millis,
    )
    return client
