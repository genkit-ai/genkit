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

"""Errors raised by the MCP client."""


class McpClientError(Exception):
    """Base class for every error this package raises.

    Catch this to handle any MCP client failure.
    """


class McpConnectionClosedError(McpClientError):
    """The client was shut down by :meth:`McpClient.close`.

    Closing is final. :meth:`McpClient.restart` connects again.
    """


class McpConnectionFailedError(McpClientError):
    """The server could not be started, or its session died.

    ``__cause__`` carries what the transport or the server raised, when it is
    known.
    """


class McpProtocolError(McpClientError):
    """The server answered in a way the protocol does not allow."""


class McpToolResultError(McpClientError):
    """A server's tool result violated the tool's declared output schema.

    The MCP SDK reports this as a bare ``RuntimeError``; the connection
    translates it into this error at the ``session.call_tool`` site, so a tool
    action can report it to the model without catching unrelated runtime
    failures.
    """
