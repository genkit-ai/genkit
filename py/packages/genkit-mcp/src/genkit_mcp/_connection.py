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

"""Lifecycle of a stdio connection to one MCP server."""

import asyncio
import contextvars
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from importlib import metadata
from typing import Any, Generic, TypeVar

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolResult,
    Implementation,
    PaginatedRequestParams,
    Tool as McpTool,
)

from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._errors import (
    McpClientError,
    McpConnectionClosedError,
    McpConnectionFailedError,
    McpProtocolError,
    McpToolResultError,
)

_T = TypeVar('_T')

_TRANSPORT_ERRORS = (anyio.BrokenResourceError, anyio.ClosedResourceError, anyio.EndOfStream)

_OUTPUT_SCHEMA_VIOLATIONS = (
    'has an output schema but did not return structured content',
    'Invalid structured content returned by tool',
    'Invalid schema for tool',
)
"""How the SDK words the bare ``RuntimeError`` it raises for a tool result that breaks its output schema."""

_PACKAGE = 'genkit-mcp'

MAX_TOOL_PAGES = 1000
"""Ceiling on ``tools/list`` pages followed before a server is called broken."""

DEFAULT_REQUEST_TIMEOUT_MILLIS = 60_000
"""How long one MCP request may take before the SDK fails it."""

LIVENESS_PROBE_SECONDS = 5.0
"""Bound on the ping that tells a dead transport from a server error reply.

A ping which does not return inside this window is read as a slow server, not a
dead one: the opposite reading tears down a healthy connection over one slow
reply.
"""


def _client_info() -> Implementation:
    """Name this package and its version for the server's session log."""
    try:
        return Implementation(name=_PACKAGE, version=metadata.version(_PACKAGE))
    except metadata.PackageNotFoundError:
        return Implementation(name=_PACKAGE, version='0.0.0')


CLIENT_INFO = _client_info()
"""What servers see in the ``clientInfo`` of every initialize request."""

_serving: contextvars.ContextVar['McpConnection | None'] = contextvars.ContextVar('genkit_mcp_serving', default=None)
"""The connection whose operation the current task, or a task it spawned, is running."""


class _ConnectionState(Enum):
    NEW = 'new'
    CONNECTING = 'connecting'
    OPEN = 'open'
    FAILED = 'failed'
    CLOSED = 'closed'


@dataclass
class _Request(Generic[_T]):
    """An operation waiting for the owner to start it, and the future its caller awaits."""

    operation: Callable[[ClientSession], Awaitable[_T]]
    result: asyncio.Future[_T]


async def _transport_is_dead(session: ClientSession, error: Exception) -> bool:
    """Report whether an operation failed because the server is unreachable.

    The SDK synthesises ``CONNECTION_CLOSED`` locally when the read stream ends,
    but that is also the JSON-RPC range a healthy server may answer with, so the
    code alone proves nothing and the server is asked directly. Any answer at
    all is read as liveness, not correctness.

    Args:
        session: Session to probe when the error is ambiguous.
        error: What the operation raised.

    Returns:
        True when the server can no longer be reached.
    """
    if isinstance(error, _TRANSPORT_ERRORS):
        return True
    if not isinstance(error, McpError) or error.error.code != CONNECTION_CLOSED:
        return False
    try:
        await asyncio.wait_for(session.send_ping(), timeout=LIVENESS_PROBE_SECONDS)
    except McpError as probe:
        return probe.error.code == CONNECTION_CLOSED
    except asyncio.TimeoutError:
        return False
    except Exception:
        return True
    return False


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
    meta: dict[str, Any] | None,
) -> CallToolResult:
    """Invoke one tool, reporting an output schema violation as this package's own error.

    Args:
        session: Session to send the request on.
        name: Tool name as the server lists it.
        arguments: Tool input, unvalidated.
        meta: Optional MCP ``_meta`` forwarded with the request.

    Returns:
        The raw result, for the caller to map.

    Raises:
        McpToolResultError: If the result violates the tool's declared output schema.
    """
    try:
        return await session.call_tool(name, arguments, meta=meta)
    except RuntimeError as error:
        message = str(error)
        if any(marker in message for marker in _OUTPUT_SCHEMA_VIOLATIONS):
            raise McpToolResultError(message) from error
        raise


class McpConnection:
    """Owns the child process and session for one MCP server.

    The MCP SDK opens the transport and the session inside a task group, so both
    must be entered and exited by the same task. One connection is therefore held
    per event loop: the Dev UI reflection server runs its own loop on a separate
    thread and would otherwise await a future belonging to the main loop.
    """

    def __init__(
        self,
        config: McpStdioServerConfig,
        request_timeout_millis: int | None = DEFAULT_REQUEST_TIMEOUT_MILLIS,
    ) -> None:
        """Prepare a connection. No process starts until the first call.

        Args:
            config: How to launch the server.
            request_timeout_millis: How long one request may take before it
                fails. ``None`` waits forever.
        """
        self._config = config
        self._request_timeout = (
            timedelta(milliseconds=request_timeout_millis) if request_timeout_millis is not None else None
        )
        self._state = _ConnectionState.NEW
        self._failure: Exception | None = None
        self._queue: asyncio.Queue[_Request[Any]] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()
        self._state_lock = asyncio.Lock()
        self._owner_task: asyncio.Task[None] | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether this connection is closed or failed and serves nothing more."""
        return self._state in (_ConnectionState.FAILED, _ConnectionState.CLOSED)

    async def list_tools(self) -> list[McpTool]:
        """List every tool the server offers, following pagination cursors.

        Returns:
            All tools, in server order.

        Raises:
            McpProtocolError: If the server's cursors never terminate.
        """

        async def list_all(session: ClientSession) -> list[McpTool]:
            tools: list[McpTool] = []
            cursor: str | None = None
            seen: set[str] = set()
            for _ in range(MAX_TOOL_PAGES):
                result = await session.list_tools(
                    params=PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                )
                tools.extend(result.tools)
                cursor = result.nextCursor
                if cursor is None:
                    return tools
                if cursor in seen:
                    raise McpProtocolError(f'{self._config.command!r} repeated the tools/list cursor {cursor!r}')
                seen.add(cursor)
            raise McpProtocolError(f'{self._config.command!r} served over {MAX_TOOL_PAGES} tools/list pages')

        return await self._request(list_all)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Invoke one tool on the server.

        Args:
            name: Tool name as the server lists it, without the Genkit prefix.
            arguments: Tool input, unvalidated.
            meta: Optional MCP ``_meta`` forwarded with the request.

        Returns:
            The raw result, for the caller to map.

        Raises:
            McpToolResultError: If the result violates the tool's declared output schema.
        """

        async def call(session: ClientSession) -> CallToolResult:
            return await _call_tool(session, name, arguments, meta)

        return await self._request(call)

    async def close(self) -> None:
        """Shut the session down and reap the child process.

        Safe to call when never connected, and safe to call twice. A closed
        connection serves nothing more.

        Raises:
            McpClientError: If called from inside one of this connection's
                own operations, which teardown would wait on forever.
        """
        task: asyncio.Task[None] | None = None
        async with self._state_lock:
            if _serving.get() is self:
                raise McpClientError('MCP connection cannot be closed from inside one of its own requests')
            teardown_started = self._state is _ConnectionState.CLOSED
            if not teardown_started:
                self._state = _ConnectionState.CLOSED
                task = self._owner_task
                self._owner_task = None

        if teardown_started:
            # Returning here while the first caller is still reaping would report
            # a shutdown that has not happened.
            await self._closed.wait()
            return

        # Cancelling the owner before its body is ever scheduled skips the
        # ``finally`` that normally releases callers parked in ``_start``.
        self._ready.set()

        try:
            if task is not None:
                # Cancelling the owner makes its ``async with`` blocks exit in
                # the same task that entered them.
                task.cancel()
                # ``gather`` reports the owner's cancellation as a result instead
                # of raising, leaving this caller's own cancellation free to
                # propagate.
                await asyncio.gather(task, return_exceptions=True)
        finally:
            # Runs even when this caller is cancelled mid-teardown: the owner is
            # already cancelled, and leaving the gate shut would hang every
            # later close instead.
            self._fail_queued(self._closed_error)
            self._closed.set()

    async def _request(self, operation: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        """Hand an operation to the owner and wait for its outcome."""
        await self._start()
        loop = asyncio.get_running_loop()
        result: asyncio.Future[_T] = loop.create_future()
        async with self._state_lock:
            if self._state is _ConnectionState.FAILED:
                raise self._failed_error()
            if self._state is _ConnectionState.CLOSED:
                raise self._closed_error()
            # Enqueueing must not await: ``_settle`` runs between awaits and
            # would otherwise drain the queue before this request joined it.
            self._queue.put_nowait(_Request(operation, result))
        return await result

    async def _start(self) -> None:
        """Start lazily, then wait until initialization finishes, however it ends."""
        async with self._state_lock:
            if self._state is _ConnectionState.NEW:
                self._state = _ConnectionState.CONNECTING
                self._owner_task = asyncio.create_task(self._run(), name='genkit-mcp-connection')
        await self._ready.wait()

    async def _run(self) -> None:
        """Own the SDK context managers for the lifetime of this connection."""
        try:
            parameters = StdioServerParameters(
                command=self._config.command,
                args=self._config.args or [],
                env=self._config.env,
                cwd=os.fspath(self._config.cwd) if self._config.cwd is not None else None,
                # A server which emits one invalid byte kills a strict decoder,
                # and with it every later request on this connection.
                encoding_error_handler='replace',
            )
            async with stdio_client(parameters) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._request_timeout,
                    client_info=CLIENT_INFO,
                ) as session:
                    await session.initialize()
                    async with self._state_lock:
                        if self._state is _ConnectionState.CONNECTING:
                            self._state = _ConnectionState.OPEN
                    self._ready.set()
                    await self._serve(session)
        except BaseException as error:
            self._settle(error if isinstance(error, Exception) else None)
            if not isinstance(error, Exception):
                raise
        finally:
            # anyio cancel scopes inside the SDK can absorb whatever ended the
            # serve loop, so this is the only settlement guaranteed to run.
            self._settle(None)
            self._ready.set()

    async def _serve(self, session: ClientSession) -> None:
        """Start each queued operation as a child task until the transport dies."""
        async with anyio.create_task_group() as operations:
            while True:
                request = await self._queue.get()
                if request.result.cancelled():
                    continue
                operations.start_soon(self._execute, session, request, operations.cancel_scope)

    async def _execute(self, session: ClientSession, request: _Request[Any], serving: anyio.CancelScope) -> None:
        """Run one operation and release its caller, however the body is left."""
        _serving.set(self)
        try:
            try:
                value = await request.operation(session)
            except Exception as error:
                if await _transport_is_dead(session, error):
                    # Settle now rather than leaving callers waiting the seconds the SDK spends terminating the child.
                    self._settle(error)
                    serving.cancel()
                elif not request.result.done():
                    request.result.set_exception(error)
            else:
                if not request.result.done():
                    request.result.set_result(value)
        except BaseException:
            # Only an ending connection cancels a child, and the state must say so before any caller is released.
            self._settle(None)
            raise
        finally:
            # Nothing else releases a started request, and a caller must never see the child's own CancelledError.
            if not request.result.done():
                request.result.set_exception(self._terminal_error())

    def _terminal_error(self) -> McpConnectionClosedError | McpConnectionFailedError:
        """Build the error for a caller the connection can no longer serve."""
        if self._state is _ConnectionState.CLOSED:
            return self._closed_error()
        return self._failed_error()

    def _settle(self, error: Exception | None) -> None:
        """Move to a terminal state and release every caller. Idempotent.

        Stays synchronous so that ``_run`` can call it from ``finally`` during
        cancellation unwinding, where awaiting ``_state_lock`` could deadlock
        against the ``close`` that is doing the cancelling.

        Args:
            error: What ended the session, when the caller knows.
        """
        if self._state is _ConnectionState.CLOSED:
            self._fail_queued(self._closed_error)
            return
        self._state = _ConnectionState.FAILED
        if self._failure is None:
            self._failure = error
        self._fail_queued(self._failed_error)

    def _fail_queued(self, build_error: Callable[[], BaseException]) -> None:
        """Release callers whose work was not yet started.

        Args:
            build_error: Makes one error per caller, so tracebacks stay separate.
        """
        while True:
            try:
                request = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if not request.result.done():
                request.result.set_exception(build_error())

    def _closed_error(self) -> McpConnectionClosedError:
        """Build the error reported once this connection is shut down."""
        return McpConnectionClosedError(f'MCP connection to {self._config.command!r} is closed')

    def _failed_error(self) -> McpConnectionFailedError:
        """Build the error reported when startup or the session died."""
        error = McpConnectionFailedError(f'MCP connection to {self._config.command!r} failed')
        error.__cause__ = self._failure
        return error
