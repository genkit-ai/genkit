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

"""Model Context Protocol client for Genkit.

Exposes the tools of one or more stdio MCP servers as Genkit ``tool.v2`` actions.
"""

from genkit_mcp._client import McpClient, create_mcp_client, define_mcp_client
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._errors import (
    McpClientError,
    McpConnectionClosedError,
    McpConnectionFailedError,
    McpProtocolError,
)
from genkit_mcp._host import McpHost, McpHostServer, create_mcp_host, define_mcp_host

__all__ = [
    'McpClient',
    'McpClientError',
    'McpConnectionClosedError',
    'McpConnectionFailedError',
    'McpHost',
    'McpHostServer',
    'McpProtocolError',
    'McpStdioServerConfig',
    'create_mcp_client',
    'create_mcp_host',
    'define_mcp_client',
    'define_mcp_host',
]
