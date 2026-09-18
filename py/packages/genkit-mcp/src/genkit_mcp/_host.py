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
from typing import Any

import structlog
from pydantic import BaseModel

from genkit import Genkit
from genkit._core._dap import DynamicActionProvider
from genkit.plugin_api import Action
from genkit_mcp._client import McpClient, create_mcp_client
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import DEFAULT_REQUEST_TIMEOUT_MILLIS

logger = structlog.get_logger(__name__)

_NAME_SEPARATORS = (':', '/')
"""Characters a tool selector uses around the host name, so a name cannot hold them."""


def _validate_host_name(name: str) -> None:
    """Reject a host name no tool selector could carry.

    Looser than the tool name rule a server name has to meet: a host name
    reaches no model, it only names the provider.

    Args:
        name: Candidate host name.

    Raises:
        ValueError: If ``<name>:tool/<tool>`` would not parse as a selector.
    """
    if name and not any(separator in name for separator in _NAME_SEPARATORS):
        return
    raise ValueError(
        f'{name!r} cannot name an MCP host: its tools are selected as '
        f"'{name}:tool/<tool>', so the name must not be empty and must hold no ':' or '/'."
    )


class McpHostServer(BaseModel):
    """One named MCP server in the list a host is built from.

    The name is the host's own label for the server rather than anything the
    server advertises, and it namespaces that server's tools, so a host holding
    ``bookshop`` serves ``bookshop_opening_hours``.

    Attributes:
        name: Name for this server within its host. Unique across the host, and
            usable in a tool name unless ``tool_prefix`` supplies another.
        server: How to launch the server.
        tool_prefix: Namespace for this server's tool names. Defaults to ``name``.
    """

    name: str
    server: McpStdioServerConfig
    tool_prefix: str | None = None


class McpHost:
    """Several MCP servers, and every tool they serve, behind one provider.

    A host holds one :class:`McpClient` per server and one dynamic action
    provider across all of them, so ``<host>:tool/*`` selects every tool of
    every server and ``<host>:tool/<server>_<tool>`` selects one of them.

    A server which cannot be listed is logged and skipped rather than failing
    the listing, so one server which will not start leaves the rest selectable.

    A host holds no Genkit instance. Registering the provider is
    :func:`define_mcp_host`, which is also what fills
    :attr:`dynamic_action_provider`.

    Obtain instances from :func:`create_mcp_host` or :func:`define_mcp_host`
    rather than constructing directly.
    """

    def __init__(
        self,
        name: str,
        servers: list[McpHostServer],
        *,
        request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
    ) -> None:
        """Build a host. No server process starts until its tools are listed.

        Args:
            name: Host name, which :func:`define_mcp_host` registers as the
                provider name used in tool selectors.
            servers: The servers this host manages, in the order it lists them.
            request_timeout_millis: How long one MCP request may take before it
                fails, for every server. ``None`` waits forever.

        Raises:
            ValueError: If ``name`` cannot appear in a tool selector, if two
                servers share a name, or if a server's name or ``tool_prefix``
                cannot appear in a model-facing tool name.
        """
        _validate_host_name(name)
        self._name = name
        self._request_timeout_millis = request_timeout_millis
        self._provider: DynamicActionProvider | None = None
        self._clients: dict[str, McpClient] = {}
        for server in servers:
            if server.name in self._clients:
                raise ValueError(
                    f'MCP host {name!r} was given two servers named {server.name!r}. A server name '
                    "namespaces that server's tools, so it has to be unique within its host."
                )
            self._clients[server.name] = self._build_client(server)

    @property
    def name(self) -> str:
        """Name this host was given.

        Once registered it is also the provider name in tool selectors, for
        example ``hosted`` in ``hosted:tool/*``.
        """
        return self._name

    @property
    def server_names(self) -> list[str]:
        """Names of the servers this host holds, in listing order."""
        return list(self._clients)

    @property
    def dynamic_action_provider(self) -> DynamicActionProvider | None:
        """The provider serving every server's tools, or ``None`` when unregistered.

        :func:`define_mcp_host` assigns it. Assigning it hands the same provider
        to every client this host holds, now and later, which is how a tool call
        that fails on one server drops the listing this host handed out.
        """
        return self._provider

    @dynamic_action_provider.setter
    def dynamic_action_provider(self, provider: DynamicActionProvider | None) -> None:
        self._provider = provider
        for client in self._clients.values():
            client.dynamic_action_provider = provider

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
        server: McpStdioServerConfig,
        *,
        tool_prefix: str | None = None,
    ) -> None:
        """Add a server to this host, replacing any server of that name.

        A replaced server is disconnected and its process stopped. Connection is
        lazy: the new server starts on the next listing.

        Args:
            server_name: Name for this server within this host.
            server: How to launch the server.
            tool_prefix: Namespace for this server's tool names. Defaults to
                ``server_name``.

        Raises:
            ValueError: If ``server_name``, or ``tool_prefix`` when given,
                cannot appear in a model-facing tool name.
        """
        client = self._build_client(McpHostServer(name=server_name, server=server, tool_prefix=tool_prefix))
        replaced = self._clients.pop(server_name, None)
        if replaced is not None:
            await self._release(replaced)
        self._clients[server_name] = client
        self._invalidate_listing()

    async def disconnect(self, server_name: str) -> None:
        """Remove a server from this host and stop its process.

        Args:
            server_name: Name of the server within this host.

        Raises:
            ValueError: If this host holds no such server.
        """
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
            McpConnectionFailedError: If the server cannot be started.
        """
        client = self._client(server_name)
        try:
            await client.restart()
        finally:
            self._invalidate_listing()

    async def get_active_tools(self) -> list[Action[Any, Any]]:
        """List the tools every server currently offers, as Genkit actions.

        Servers are listed concurrently, and each of them is listed afresh, as
        :meth:`McpClient.get_active_tools` is. A server which cannot be listed
        is logged and contributes nothing, so one server which will not start
        cannot empty this host's catalog.

        The actions are unregistered: a model reaches them through a dynamic
        action provider serving them, which :func:`define_mcp_host` sets up.

        Returns:
            One action per tool, servers in listing order and tools in server
            order.
        """
        servers = list(self._clients.items())
        listings = await asyncio.gather(
            *(client.get_active_tools() for _, client in servers),
            return_exceptions=True,
        )
        tools: list[Action[Any, Any]] = []
        for (server_name, _), listing in zip(servers, listings, strict=True):
            if isinstance(listing, asyncio.CancelledError):
                raise listing
            if isinstance(listing, BaseException):
                logger.error(
                    'MCP server could not be listed, serving the other servers without it',
                    server=server_name,
                    host=self._name,
                    exc_info=listing,
                )
                continue
            tools.extend(listing)
        return tools

    async def close(self) -> None:
        """Disconnect every server this host holds and stop their processes.

        Unlike :meth:`McpClient.close` this is not the end of the host: it holds
        no servers afterwards and advertises no tools, and :meth:`connect` puts
        servers back. A provider registered for this host stays registered.

        Raises:
            McpClientError: If a server could not be disconnected. Every server
                is disconnected first, and the first failure is reported.
        """
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
        client.dynamic_action_provider = None
        await client.close()

    def _build_client(self, server: McpHostServer) -> McpClient:
        """Build a client for one server, serving this host's provider."""
        client = create_mcp_client(
            server.name,
            server.server,
            tool_prefix=server.tool_prefix,
            request_timeout_millis=self._request_timeout_millis,
        )
        client.dynamic_action_provider = self._provider
        return client

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


def create_mcp_host(
    name: str,
    servers: list[McpHostServer],
    *,
    request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
) -> McpHost:
    """Build a host for several MCP servers without registering it with Genkit.

    Nothing reaches a registry, so no selector resolves to these servers' tools
    and no Genkit instance is needed. Use :meth:`McpHost.get_active_tools` to
    obtain them as actions and serve them from a dynamic action provider of your
    own. Connection is lazy: a child process starts on the first tool listing.

    Reach for :func:`define_mcp_host` instead to expose every server's tools as
    ``<name>:tool/*`` straight away.

    Args:
        name: Host name, carried by its provider and its errors.
        servers: The servers this host manages, in the order it lists them.
        request_timeout_millis: How long one MCP request may take before it
            fails, for every server. ``None`` waits forever.

    Returns:
        A handle for listing tools, connecting, reconnecting and disconnecting.

    Raises:
        ValueError: If ``name`` cannot appear in a tool selector, if two servers
            share a name, or if a server's name or ``tool_prefix`` cannot appear
            in a model-facing tool name.
    """
    return McpHost(name, servers, request_timeout_millis=request_timeout_millis)


def define_mcp_host(
    ai: Genkit,
    name: str,
    servers: list[McpHostServer],
    *,
    cache_ttl_millis: int | None = None,
    request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
) -> McpHost:
    """Register the tools of several MCP servers with Genkit, under one name.

    Every server's tools become ``tool.v2`` actions behind a single dynamic
    action provider named ``name``. Select them from ``generate`` as
    ``<name>:tool/*`` for all of them, or ``<name>:tool/<server>_<tool>`` for
    one. Connection is lazy: a child process starts on the first tool listing.

    Args:
        ai: The Genkit instance to register with.
        name: Provider name, used in tool selectors.
        servers: The servers this host manages, in the order it lists them.
        cache_ttl_millis: How long a tool listing stays fresh. Defaults to the
            core dynamic action provider default.
        request_timeout_millis: How long one MCP request may take before it
            fails, for every server. ``None`` waits forever.

    Returns:
        A handle for connecting, reconnecting and disconnecting, carrying the
        registered provider as :attr:`McpHost.dynamic_action_provider`.

    Raises:
        ValueError: If ``name`` cannot appear in a tool selector, if two servers
            share a name, or if a server's name or ``tool_prefix`` cannot appear
            in a model-facing tool name.
    """
    host = McpHost(name, servers, request_timeout_millis=request_timeout_millis)

    async def list_tools() -> dict[str, list[Action[Any, Any]]]:
        """Provide every server's tools to the dynamic action provider."""
        return {'tool': await host.get_active_tools()}

    host.dynamic_action_provider = ai.define_dynamic_action_provider(
        name,
        list_tools,
        description=f'Tools of the MCP servers hosted by {name}.',
        cache_ttl_millis=cache_ttl_millis,
    )
    return host
