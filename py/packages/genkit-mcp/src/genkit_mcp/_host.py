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

"""Genkit host for several MCP servers behind one dynamic action provider."""

import asyncio
import os
from collections.abc import Mapping
from typing import Any

import structlog

from genkit import DynamicActionProvider, Genkit
from genkit.plugin_api import Action
from genkit_mcp._client import McpClient
from genkit_mcp._config import McpServerEntry, McpStdioServerConfig
from genkit_mcp._connection import DEFAULT_REQUEST_TIMEOUT_MILLIS
from genkit_mcp._errors import _ConnectionClosedError, _ConnectionFailedError
from genkit_mcp._tools import TOOL_NAME_PATTERN, validate_provider_name

logger = structlog.get_logger(__name__)


class McpHost:
    """Several MCP servers, and every tool they serve, behind one provider.

    A host holds one :class:`McpClient` per server and one dynamic action
    provider across all of them, so ``<host>:tool/*`` selects every tool of
    every server and ``<host>:tool/<server>_<tool>`` selects one of them.

    A server that cannot start, or whose process dies, is logged and skipped,
    so one such server leaves the rest selectable. Any other failure to list a
    server, such as a protocol violation, an error the server returns, or two
    of its tools sharing a name, fails the listing.

    A host holds no Genkit instance. Registering the provider is
    :func:`define_mcp_host`.

    Obtain instances from :func:`define_mcp_host` rather than constructing
    directly.
    """

    def __init__(
        self,
        name: str,
        *,
        mcp_servers: Mapping[str, McpServerEntry],
        request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
    ) -> None:
        """Build a host. No server process starts until its tools are listed.

        Args:
            name: Host name, which :func:`define_mcp_host` registers as the
                provider name used in tool selectors.
            mcp_servers: The servers this host manages, keyed by the name that
                prefixes each server's tools, in the order it lists them.
            request_timeout_millis: How long one MCP request may take before it
                fails, for every server. ``None`` waits forever.

        Raises:
            ValueError: If ``name`` cannot appear in a tool selector, if a
                server name cannot appear in a model-facing tool name, or if an
                entry is not a stdio server entry.
        """
        validate_provider_name(name)
        self._name = name
        self._request_timeout_millis = request_timeout_millis
        self._provider: DynamicActionProvider | None = None
        self._clients: dict[str, McpClient] = {}
        self._closed = False
        for server_name, entry in mcp_servers.items():
            config = McpStdioServerConfig.model_validate(entry)
            self._clients[server_name] = self._build_client(
                server_name, command=config.command, args=config.args, env=config.env, cwd=config.cwd
            )

    @property
    def name(self) -> str:
        """Name this host was given.

        Once registered it is also the provider name in tool selectors, for
        example ``hosted`` in ``hosted:tool/*``.
        """
        return self._name

    def _set_provider(self, provider: DynamicActionProvider | None) -> None:
        """Serve every server's tools from ``provider``, now and for servers added later.

        Each client holds the same provider, which is how a tool call that fails
        on one server drops the listing this host handed out.
        """
        self._provider = provider
        for client in self._clients.values():
            client._provider = provider

    def tool_name(self, server_name: str, tool: str) -> str:
        """Return the Genkit name of one tool of one of this host's servers.

        Use it to build a selector for ``generate`` without assembling the
        prefix by hand::

            tools = [f'{host.name}:tool/{host.tool_name("bookshop", "opening_hours")}']

        Args:
            server_name: Name of the server within this host.
            tool: Tool name as the server lists it, without any prefix.

        Returns:
            The namespaced name, for example ``bookshop_opening_hours``.

        Raises:
            ValueError: If this host holds no such server.
        """
        return self._client(server_name).tool_name(tool)

    async def connect(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> None:
        """Add a server to this host, replacing any server of that name.

        A replaced server is disconnected and its process stopped. Connection is
        lazy: the new server starts on the next listing.

        Args:
            server_name: Name for this server within this host, which prefixes
                its tool names.
            command: Executable that runs the server.
            args: Arguments passed to ``command``.
            env: Extra environment for the server process, over the minimal set
                the MCP SDK supplies.
            cwd: Working directory for the server process.

        Raises:
            ValueError: If ``server_name`` cannot appear in a model-facing tool
                name.
            GenkitError: ``FAILED_PRECONDITION`` if this host is closed.
        """
        self._refuse_when_closed()
        client = self._build_client(server_name, command=command, args=args, env=env, cwd=cwd)
        replaced = self._clients.pop(server_name, None)
        try:
            if replaced is not None:
                await self._release(replaced)
        finally:
            self._clients[server_name] = client
            self._invalidate_listing()

    async def disconnect(self, server_name: str) -> None:
        """Remove a server from this host and stop its process.

        Args:
            server_name: Name of the server within this host.

        Raises:
            ValueError: If this host holds no such server.
            GenkitError: ``FAILED_PRECONDITION`` if this host is closed.
        """
        self._refuse_when_closed()
        client = self._client(server_name)
        del self._clients[server_name]
        try:
            await self._release(client)
        finally:
            self._invalidate_listing()

    async def reconnect(self, server_name: str) -> None:
        """Restart one server, replacing the connection this host has to it.

        The calling loop reconnects and re-lists before this returns, so a
        server which still cannot be started says so here rather than on the
        next generate. The server stays in this host either way.

        Args:
            server_name: Name of the server within this host.

        Raises:
            ValueError: If this host holds no such server.
            GenkitError: ``FAILED_PRECONDITION`` if this host is closed, or
                ``UNAVAILABLE`` if the server cannot be started.
        """
        self._refuse_when_closed()
        client = self._client(server_name)
        try:
            await client.restart()
        finally:
            self._invalidate_listing()

    async def get_active_tools(self) -> list[Action[Any, Any]]:
        """List the tools every server currently offers, as Genkit actions.

        Servers are listed concurrently, and each of them is listed afresh, as
        :meth:`McpClient.get_active_tools` is. A server that cannot start, or
        whose process dies, is logged and contributes nothing, so it cannot
        empty this host's catalog. Any other failure fails the listing.

        The actions are unregistered: a model reaches them through a dynamic
        action provider serving them, which :func:`define_mcp_host` sets up.

        Returns:
            One action per tool, servers in listing order and tools in server
            order.

        Raises:
            GenkitError: ``FAILED_PRECONDITION`` if this host is closed, or
                ``INTERNAL`` if a server breaks the protocol.
            McpError: If a server answers the listing with an error.
            ValueError: If two tools share a name, within one server or across
                two. Distinct server names do not rule this out, because the
                name and the tool name are joined by the underscore either of
                them may hold.
        """
        self._refuse_when_closed()
        servers = list(self._clients.items())
        listings = await asyncio.gather(
            *(client.get_active_tools() for _, client in servers),
            return_exceptions=True,
        )
        tools: list[Action[Any, Any]] = []
        served_by: dict[str, str] = {}
        for (server_name, _), listing in zip(servers, listings, strict=True):
            if isinstance(listing, _ConnectionFailedError):
                logger.error(
                    'MCP server could not be reached, serving the other servers without it',
                    server=server_name,
                    host=self._name,
                    exc_info=listing,
                )
                continue
            if isinstance(listing, BaseException):
                raise listing
            for tool in listing:
                other = served_by.setdefault(tool.name, server_name)
                if other != server_name:
                    raise ValueError(
                        f'MCP host {self._name!r} cannot serve servers {other!r} and {server_name!r} '
                        f'together: both serve a tool named {tool.name!r}. Rename one of them so '
                        'their tool names stay apart.'
                    )
            tools.extend(listing)
        return tools

    async def close(self) -> None:
        """Disconnect every server this host holds, stop their processes, and keep it that way.

        Closing is final, as it is for :meth:`McpClient.close`: a closed host
        cannot connect servers again, so build a new one with
        :func:`define_mcp_host`. A provider registered for this host stays
        registered and its listing now fails as closed.

        Raises:
            GenkitError: If a server could not be disconnected. Every server is
                disconnected first, and the first failure is reported.
        """
        self._closed = True
        clients = list(self._clients.values())
        self._clients.clear()
        results = await asyncio.gather(*(self._release(client) for client in clients), return_exceptions=True)
        self._invalidate_listing()
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _release(self, client: McpClient) -> None:
        """Close a client this host no longer holds, and detach it from the listing.

        A client invalidates whatever provider it holds, so a dropped one would
        otherwise still drop the listing of the servers it left behind.
        """
        client._provider = None
        await client.close()

    def _build_client(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str] | None,
        env: dict[str, str] | None,
        cwd: str | os.PathLike[str] | None,
    ) -> McpClient:
        """Build a client for one server, serving this host's provider."""
        if not TOOL_NAME_PATTERN.fullmatch(server_name):
            raise ValueError(
                f'MCP host {self._name!r} cannot hold a server named {server_name!r}: the name prefixes '
                'its tool names, so it must start with a letter or an underscore and hold only letters, '
                'digits, underscores and hyphens.'
            )
        client = McpClient(
            server_name,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
            request_timeout_millis=self._request_timeout_millis,
        )
        client._provider = self._provider
        return client

    def _refuse_when_closed(self) -> None:
        """Raise if this host has been closed."""
        if self._closed:
            raise _ConnectionClosedError(f'MCP host {self._name!r} is closed')

    def _client(self, server_name: str) -> McpClient:
        """Return the client for one server name, or say the host has no such server."""
        client = self._clients.get(server_name)
        if client is None:
            raise ValueError(f'MCP host {self._name!r} holds no server named {server_name!r}')
        return client

    def _invalidate_listing(self) -> None:
        """Drop the registered listing of this host's tools, if it has one."""
        if self._provider is not None:
            self._provider.invalidate_cache()


def define_mcp_host(
    ai: Genkit,
    name: str,
    *,
    mcp_servers: Mapping[str, McpServerEntry],
    cache_ttl_millis: int | None = None,
    request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
) -> McpHost:
    """Register the tools of several MCP servers with Genkit, under one name.

    Every server's tools become ``tool.v2`` actions behind a single dynamic
    action provider named ``name``. Select them from ``generate`` as
    ``<name>:tool/*`` for all of them, or ``<name>:tool/<server>_<tool>`` for
    one. Connection is lazy: a child process starts on the first tool listing.

    ``mcp_servers`` takes the shape of an ``mcpServers`` block from Claude
    Desktop or Cursor, so one can be passed straight from its file::

        host = define_mcp_host(ai, 'hosted', mcp_servers=json.load(f)['mcpServers'])

    Args:
        ai: The Genkit instance to register with.
        name: Provider name, used in tool selectors.
        mcp_servers: The servers this host manages, keyed by the name that
            prefixes each server's tools, in the order it lists them. Each
            entry holds ``command`` and optionally ``args``, ``env``, ``cwd``
            and ``type``, which can only be ``stdio``.
        cache_ttl_millis: How long a tool listing stays fresh. Defaults to the
            core dynamic action provider default.
        request_timeout_millis: How long one MCP request may take before it
            fails, for every server. ``None`` waits forever.

    Returns:
        A handle for listing tools, connecting, reconnecting and disconnecting.

    Raises:
        ValueError: If ``name`` cannot appear in a tool selector, if a server
            name cannot appear in a model-facing tool name, or if an entry is
            not a stdio server entry.
    """
    host = McpHost(name, mcp_servers=mcp_servers, request_timeout_millis=request_timeout_millis)

    async def list_tools() -> dict[str, list[Action[Any, Any]]]:
        """Provide every server's tools to the dynamic action provider."""
        return {'tool': await host.get_active_tools()}

    host._set_provider(
        ai.define_dynamic_action_provider(
            name,
            list_tools,
            description=f'Tools of the MCP servers hosted by {name}.',
            cache_ttl_millis=cache_ttl_millis,
        )
    )
    return host
