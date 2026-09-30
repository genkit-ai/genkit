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

import os

from pydantic import BaseModel


class McpStdioServerConfig(BaseModel):
    """How to launch an MCP server as a child process over stdio.

    Field names match the `mcpServers` entries used by Claude Desktop, Cursor and
    the JS plugin, so an existing config block can be passed through unchanged.

    Attributes:
        command: Executable to run, for example ``npx`` or ``uv``.
        args: Arguments passed to ``command``.
        env: Extra environment for the child process, merged over the minimal
            set the MCP SDK always supplies: ``HOME``, ``LOGNAME``, ``PATH``,
            ``SHELL``, ``TERM`` and ``USER`` on posix, and the ``APPDATA``,
            ``HOMEDRIVE``, ``HOMEPATH``, ``LOCALAPPDATA``, ``PATH``,
            ``PATHEXT``, ``PROCESSOR_ARCHITECTURE``, ``SYSTEMDRIVE``,
            ``SYSTEMROOT``, ``TEMP``, ``USERNAME`` and ``USERPROFILE`` set on
            Windows. ``None`` leaves the child with that minimal set alone: the
            parent environment is never inherited, so a variable the server
            needs, an API key for example, has to be listed here even when this
            process already has it.
        cwd: Working directory for the child process.
        disabled: When true the client connects to nothing and advertises no tools.
    """

    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    cwd: str | os.PathLike[str] | None = None
    disabled: bool = False
