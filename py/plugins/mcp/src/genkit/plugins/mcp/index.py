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

"""MCP Plugin Index.

This module serves as the main entry point for the MCP plugin,
similar to js/plugins/mcp/src/index.ts.

In Python, the actual exports are handled by the parent __init__.py,
but this file exists for structural parity with the JS SDK.
"""

from .client.client import McpClient, McpServerConfig, create_mcp_client
from .client.host import McpHost, create_mcp_host
from .server import McpServer, McpServerOptions, create_mcp_server

__all__ = [
    'McpClient',
    'McpHost',
    'McpServer',
    'McpServerConfig',
    'McpServerOptions',
    'create_mcp_client',
    'create_mcp_host',
    'create_mcp_server',
]
