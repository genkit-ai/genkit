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

"""Errors raised by the MCP client.

None of these is exported. Callers catch ``GenkitError`` and read its
``status``; the classes exist so this package can tell its own failures apart.
"""

from typing import ClassVar

from genkit import GenkitError
from genkit.plugin_api import StatusName


class _ClientError(GenkitError):
    """Base class for every error this package raises."""

    _status: ClassVar[StatusName] = 'INTERNAL'

    def __init__(self, message: str, *, status: StatusName | None = None, cause: BaseException | None = None) -> None:
        """Build the error with this class's status unless ``status`` overrides it.

        Args:
            message: What went wrong.
            status: Status to report instead of the class default.
            cause: The provider error this one reports, if any.
        """
        super().__init__(message=message, status=status or self._status, cause=cause)


class _ConnectionClosedError(_ClientError):
    """The client was shut down by :meth:`McpClient.close`. Status ``FAILED_PRECONDITION``.

    Closing is final: a closed client cannot be restarted. Build a new one with
    :func:`define_mcp_client`.
    """

    _status: ClassVar[StatusName] = 'FAILED_PRECONDITION'


class _ConnectionFailedError(_ClientError):
    """The server could not be started, or its session died. Status ``UNAVAILABLE``.

    ``__cause__`` carries what the transport or the server raised, when it is
    known, without the traceback it was raised with.
    """

    _status: ClassVar[StatusName] = 'UNAVAILABLE'


class _RequestTimeoutError(_ClientError):
    """The server did not answer one request in time. Status ``DEADLINE_EXCEEDED``.

    The session stays open, since a slow server is not a dead one.
    ``__cause__`` carries the timeout the MCP SDK raised.
    """

    _status: ClassVar[StatusName] = 'DEADLINE_EXCEEDED'


class _ServerError(_ClientError):
    """The server answered one request with a JSON-RPC error. Status ``UNKNOWN`` unless its code names one.

    The session stays open, since the server is still answering.
    ``__cause__`` carries the ``McpError`` the MCP SDK raised.
    """

    _status: ClassVar[StatusName] = 'UNKNOWN'


class _ProtocolError(_ClientError):
    """The server answered in a way the protocol does not allow. Status ``INTERNAL``."""


class _ToolResultError(_ClientError):
    """A server's tool result violated the tool's declared output schema.

    The MCP SDK reports this as a bare ``RuntimeError``; the connection
    translates it into this error at the ``session.call_tool`` site, so a tool
    action can report it to the model without catching unrelated runtime
    failures.
    """
