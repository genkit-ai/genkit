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

from genkit import Genkit
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import DEFAULT_REQUEST_TIMEOUT_MILLIS


class McpClient:
    """One MCP server, registered with Genkit as a dynamic action provider.

    Obtain instances via :func:`define_mcp_client` rather than constructing directly.
    """

    @property
    def name(self) -> str:
        """Provider name used in tool selectors.

        For example, ``everything`` in the selector ``everything:tool/*``.
        """
        raise NotImplementedError

    async def restart(self) -> None:
        """Reconnect to the server, replacing any connection it already has."""
        raise NotImplementedError

    async def close(self) -> None:
        """Disconnect from the server and stop its process, for good."""
        raise NotImplementedError


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
        tool_prefix: Namespace for this server's tool names. Defaults to
            ``name``.
        cache_ttl_millis: How long a tool listing stays fresh. Defaults to the
            core dynamic action provider default.
        request_timeout_millis: How long one MCP request may take before it
            fails. ``None`` waits forever.

    Returns:
        A handle for disconnecting.

    Raises:
        ValueError: If the tool prefix cannot appear in a model-facing tool name.
    """
    raise NotImplementedError
