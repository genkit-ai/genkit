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

"""Transport utilities for MCP plugin.

This module contains helper functions for creating and managing
MCP transport connections (stdio, SSE, custom).
"""

import importlib.util
from typing import cast

import structlog

from mcp import StdioServerParameters

logger = structlog.get_logger(__name__)


def create_stdio_params(
    command: str, args: list[str] | None = None, env: dict[str, str] | None = None
) -> StdioServerParameters:
    """Create StdioServerParameters for MCP connection.

    Args:
        command: Command to execute
        args: Command arguments
        env: Environment variables

    Returns:
        StdioServerParameters object
    """
    return StdioServerParameters(command=command, args=args or [], env=env)


async def transport_from(config: dict[str, object], session_id: str | None = None) -> tuple[object, str]:
    """Create an MCP transport instance based on the provided server configuration.

    Supports creating SSE, Stdio, or using a pre-configured custom transport.

    Args:
        config: Configuration for the MCP server
        session_id: Optional session ID for HTTP transport

    Returns:
        Tuple of (transport instance or None, transport type string)

    Note:
        This function mirrors the JS SDK's transportFrom() function.
    """
    # Handle pre-configured transport first
    if config.get('transport'):
        return (config['transport'], 'custom')

    # Handle SSE/HTTP config
    if config.get('url'):
        # Check if SSE client is available
        if importlib.util.find_spec('mcp.client.sse') is None:
            logger.warning('SSE client not available')
            return (None, 'http')

        # Note: Python MCP SDK may have different SSE client API
        # This is a placeholder that matches the pattern
        logger.info(f'Creating SSE transport for URL: {config["url"]}')
        return (config['url'], 'http')  # Simplified for now

    # Handle Stdio config
    if config.get('command'):
        cmd = str(config['command'])
        args_raw = config.get('args')
        args: list[str] | None = cast(list[str], args_raw) if isinstance(args_raw, list) else None
        env_raw = config.get('env')
        env: dict[str, str] | None = cast(dict[str, str], env_raw) if isinstance(env_raw, dict) else None
        stdio_params = create_stdio_params(command=cmd, args=args, env=env)
        return (stdio_params, 'stdio')

    return (None, 'unknown')
