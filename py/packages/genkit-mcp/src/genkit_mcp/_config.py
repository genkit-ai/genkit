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

"""Connection configuration for MCP servers."""

from pydantic import BaseModel


class McpStdioServerConfig(BaseModel):
    """How to launch an MCP server as a child process over stdio.

    Field names match the `mcpServers` entries used by Claude Desktop, Cursor and
    the JS plugin, so an existing config block can be passed through unchanged.

    Attributes:
        command: Executable to run, for example ``npx`` or ``uv``.
        args: Arguments passed to ``command``.
        env: Environment for the child process. ``None`` inherits the parent's.
        cwd: Working directory for the child process.
        disabled: When true the client connects to nothing and advertises no tools.
    """

    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    disabled: bool = False
