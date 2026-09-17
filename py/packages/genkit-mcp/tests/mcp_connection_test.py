# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for MCP stdio connection lifecycle."""

import asyncio
import gc
import os
import signal
import sys
import threading
from importlib import metadata
from pathlib import Path
from typing import cast

import anyio
import pytest
from genkit_mcp import _connection
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import McpConnection, _call_tool, _ConnectionState, _Request
from genkit_mcp._errors import (
    McpClientError,
    McpConnectionClosedError,
    McpConnectionFailedError,
    McpProtocolError,
    McpToolResultError,
)
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData, TextContent, Tool as McpTool

ALL_TOOLS = [
    'echo',
    'second',
    'wait',
    'list_requests',
    'who_called',
    'stall',
    'rendezvous',
    'await_stall',
    'bad_output',
]

# Every await on connection work is bounded by this, so a lifecycle regression
# fails the suite rather than hanging it. There is no global pytest timeout, so
# new awaits on a connection have to be wrapped too.
LIMIT = 15


def config(tmp_path: Path, pid_file: Path | None = None, **env: str) -> McpStdioServerConfig:
    """Launch the test server with the same interpreter as the test suite."""
    return McpStdioServerConfig(
        command=sys.executable,
        args=[str(Path(__file__).with_name('fake_server.py'))],
        env={
            **os.environ,
            'MCP_FAKE_PID_FILE': str(pid_file if pid_file is not None else tmp_path / 'server.pid'),
            **env,
        },
    )


def text_of(result: CallToolResult) -> str:
    """Read the one text block a fake-server tool returns."""
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def server_pid(pid_file: Path) -> int:
    """Wait for the pid the server writes before it starts serving."""

    async def read() -> int:
        while True:
            try:
                return int(await anyio.Path(pid_file).read_text(encoding='utf-8'))
            except (FileNotFoundError, ValueError):
                await asyncio.sleep(0.01)

    return await asyncio.wait_for(read(), timeout=LIMIT)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def settle(owner: asyncio.Task[None]) -> None:
    """Wait for the owner task to finish unwinding, without cancelling it."""
    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=LIMIT)


class Blocker:
    """A request that stays in flight until the test releases it."""

    def __init__(self) -> None:
        self.running = asyncio.Event()
        self.release = asyncio.Event()
        self.result: asyncio.Future[list[McpTool]] = asyncio.get_running_loop().create_future()

    async def operation(self, session: ClientSession) -> list[McpTool]:
        self.running.set()
        await self.release.wait()
        return []


async def hold_in_flight(connection: McpConnection) -> Blocker:
    """Start a request the connection cannot finish on its own."""
    blocker = Blocker()
    connection._queue.put_nowait(_Request(blocker.operation, blocker.result))
    await asyncio.wait_for(blocker.running.wait(), timeout=LIMIT)
    return blocker


async def close_with_queued(connection: McpConnection, *requests: _Request[list[McpTool]]) -> None:
    """Enqueue and close in one scheduler step, so the owner never starts the entries.

    The owner starts whatever it dequeues at once, so an entry only stays
    queued when close cancels the owner before its next turn. Nothing in close
    suspends before that cancel.
    """
    for request in requests:
        connection._queue.put_nowait(request)
    await connection.close()


@pytest.mark.asyncio
async def test_connect_list_call_and_close_reaps_server(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))

    assert [tool.name for tool in await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)] == ALL_TOOLS
    assert text_of(await asyncio.wait_for(connection.call_tool('echo', {'message': 'hello'}), timeout=LIMIT)) == 'hello'
    pid = int((tmp_path / 'server.pid').read_text(encoding='utf-8'))

    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    assert not is_running(pid)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_list_tools_follows_pagination_cursors(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path, MCP_FAKE_PAGE_SIZE='1'))
    try:
        assert [tool.name for tool in await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)] == ALL_TOOLS
        served = await asyncio.wait_for(connection.call_tool('list_requests', {}), timeout=LIMIT)
        assert text_of(served) == str(len(ALL_TOOLS))
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_list_tools_rejects_a_non_advancing_cursor(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path, MCP_FAKE_STUCK_CURSOR='1'))
    try:
        with pytest.raises(McpProtocolError, match='repeated the tools/list cursor'):
            await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_concurrent_requests_each_receive_their_own_result(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        first, tools, second, third = await asyncio.gather(
            connection.call_tool('echo', {'message': 'one'}),
            connection.list_tools(),
            connection.call_tool('echo', {'message': 'two'}),
            connection.call_tool('second', {'message': 'three'}),
        )
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)

    assert text_of(first) == 'one'
    assert text_of(second) == 'two'
    assert text_of(third) == 'three'
    assert [tool.name for tool in tools] == ALL_TOOLS


@pytest.mark.asyncio
async def test_a_cancelled_caller_is_never_sent_to_the_server(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
        started: list[str] = []

        async def abandoned(session: ClientSession) -> list[McpTool]:
            started.append('ran')
            return (await session.list_tools()).tools

        cancelled: asyncio.Future[list[McpTool]] = asyncio.get_running_loop().create_future()
        cancelled.cancel()
        connection._queue.put_nowait(_Request(abandoned, cancelled))

        behind = await asyncio.wait_for(connection.call_tool('echo', {'message': 'behind'}), timeout=LIMIT)
        assert started == []
        assert text_of(behind) == 'behind'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_close_interrupts_an_in_flight_call(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    call = asyncio.create_task(connection.call_tool('wait', {}))
    await asyncio.sleep(0.05)

    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(call, timeout=LIMIT)


@pytest.mark.asyncio
async def test_close_releases_a_request_that_never_reached_the_server(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    blocker = await hold_in_flight(connection)
    started: list[str] = []

    async def unstarted(session: ClientSession) -> list[McpTool]:
        started.append('ran')
        return (await session.list_tools()).tools

    queued: asyncio.Future[list[McpTool]] = asyncio.get_running_loop().create_future()

    await asyncio.wait_for(close_with_queued(connection, _Request(unstarted, queued)), timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(queued, timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(blocker.result, timeout=LIMIT)
    assert started == []


@pytest.mark.asyncio
async def test_close_propagates_the_callers_own_cancellation(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    pid = int((tmp_path / 'server.pid').read_text(encoding='utf-8'))
    owner = connection._owner_task
    assert owner is not None

    closing = asyncio.create_task(connection.close())
    await asyncio.sleep(0)
    # close reaches its first suspension point, the owner's teardown, in one
    # scheduler pass, because nothing before it awaits a contended primitive.
    assert connection._owner_task is None
    assert not closing.done()
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=LIMIT)
    await settle(owner)
    assert not is_running(pid)
    await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_close_before_the_owner_task_starts_releases_the_waiter(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    waiter = asyncio.create_task(connection.list_tools())
    await asyncio.sleep(0)
    assert connection._state is _ConnectionState.CONNECTING

    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(waiter, timeout=LIMIT)


@pytest.mark.asyncio
async def test_close_during_startup_releases_the_waiter(tmp_path: Path) -> None:
    pid_file = tmp_path / 'stalled.pid'
    connection = McpConnection(config(tmp_path, pid_file=pid_file, MCP_FAKE_STALL='1'))
    waiter = asyncio.create_task(connection.list_tools())
    pid = await server_pid(pid_file)

    assert connection._state is _ConnectionState.CONNECTING
    await asyncio.wait_for(connection.close(), timeout=LIMIT)

    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(waiter, timeout=LIMIT)
    assert connection._state is _ConnectionState.CLOSED
    assert not is_running(pid)


@pytest.mark.asyncio
async def test_close_from_inside_a_request_is_rejected(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

        async def close_from_operation(session: ClientSession) -> None:
            # Bounded by the wait_for around _request.
            await connection.close()

        with pytest.raises(McpClientError, match='from inside one of its own requests'):
            await asyncio.wait_for(connection._request(close_from_operation), timeout=LIMIT)

        assert connection._state is _ConnectionState.OPEN
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_server_death_makes_the_connection_terminal(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    owner = connection._owner_task
    assert owner is not None
    os.kill(int((tmp_path / 'server.pid').read_text(encoding='utf-8')), signal.SIGKILL)

    with pytest.raises(McpConnectionFailedError, match='failed') as first:
        await asyncio.wait_for(connection.call_tool('echo', {}), timeout=LIMIT)
    assert first.value.__cause__ is not None
    # Settled by the serve loop, not by the owner's unwinding, so callers
    # arriving during the SDK's process teardown are turned away at once.
    assert connection._state is _ConnectionState.FAILED

    await settle(owner)
    assert connection._state is _ConnectionState.FAILED
    with pytest.raises(McpConnectionFailedError, match='failed'):
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_a_server_error_code_does_not_poison_a_live_connection(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path, MCP_FAKE_APP_ERROR='1'))
    try:
        pid_file = tmp_path / 'server.pid'
        with pytest.raises(McpError, match='application-level failure'):
            await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

        assert connection._state is _ConnectionState.OPEN
        assert is_running(await server_pid(pid_file))
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
        assert [tool.name for tool in await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)] == ALL_TOOLS
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_list_tools_rejects_endlessly_advancing_cursors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_connection, 'MAX_TOOL_PAGES', 3)
    connection = McpConnection(config(tmp_path, MCP_FAKE_ENDLESS_CURSOR='1'))
    try:
        with pytest.raises(McpProtocolError, match='served over 3 tools/list pages'):
            await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
        # The server counts what it served, so an off-by-N cap cannot hide
        # behind a message built from the same constant.
        served = await asyncio.wait_for(connection.call_tool('list_requests', {}), timeout=LIMIT)
        assert text_of(served) == '3'
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_a_concurrent_close_waits_for_the_child_to_be_reaped(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    pid = int((tmp_path / 'server.pid').read_text(encoding='utf-8'))
    call = asyncio.create_task(connection.call_tool('wait', {}))
    await asyncio.sleep(0.05)

    first = asyncio.create_task(connection.close())
    await asyncio.sleep(0)
    # close takes the owner handle before its first suspension, so the teardown
    # this second caller must wait behind is provably under way.
    assert connection._owner_task is None
    assert not connection._closed.is_set()
    second = asyncio.create_task(connection.close())
    await asyncio.sleep(0)

    try:
        assert is_running(pid)
        assert not second.done()
    finally:
        await asyncio.wait_for(asyncio.gather(first, second, return_exceptions=True), timeout=LIMIT)

    assert not is_running(pid)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(call, timeout=LIMIT)


class _ProbeSession:
    """Stands in for a ClientSession so the probe's branches can be driven."""

    def __init__(self, outcome: Exception | None = None, delay: float = 0.0) -> None:
        self.outcome = outcome
        self.delay = delay
        self.pings = 0

    async def send_ping(self) -> str:
        self.pings += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.outcome is not None:
            raise self.outcome
        return 'pong'


def leaf_causes(error: BaseException | None) -> list[BaseException]:
    """Flatten an exception and the task groups it may be wrapped in."""
    if error is None:
        return []
    nested = getattr(error, 'exceptions', None)
    if nested is None:
        return [error]
    return [leaf for child in nested for leaf in leaf_causes(child)]


def closed_mcp_error(message: str = 'connection closed') -> McpError:
    return McpError(ErrorData(code=CONNECTION_CLOSED, message=message))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('error', 'dead'),
    [
        (anyio.BrokenResourceError(), True),
        (anyio.ClosedResourceError(), True),
        (anyio.EndOfStream(), True),
        (RuntimeError('unrelated'), False),
        (McpError(ErrorData(code=-32601, message='method not found')), False),
        (McpError(ErrorData(code=408, message='Timed out while waiting for response')), False),
    ],
)
async def test_transport_death_is_decided_without_a_probe(error: Exception, dead: bool) -> None:
    session = _ProbeSession()
    assert await _connection._transport_is_dead(cast(ClientSession, session), error) is dead
    assert session.pings == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('outcome', 'dead'),
    [
        (None, False),
        (McpError(ErrorData(code=-32601, message='method not found')), False),
        (closed_mcp_error(), True),
        (anyio.ClosedResourceError(), True),
    ],
)
async def test_an_ambiguous_code_is_decided_by_the_probe(outcome: Exception | None, dead: bool) -> None:
    session = _ProbeSession(outcome)
    assert await _connection._transport_is_dead(cast(ClientSession, session), closed_mcp_error()) is dead
    assert session.pings == 1


@pytest.mark.asyncio
async def test_a_probe_that_times_out_reads_the_server_as_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_connection, 'LIVENESS_PROBE_SECONDS', 0.01)
    session = _ProbeSession(delay=LIMIT)
    assert await _connection._transport_is_dead(cast(ClientSession, session), closed_mcp_error()) is False


@pytest.mark.asyncio
async def test_close_during_a_probe_settles_the_in_flight_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probing = asyncio.Event()

    async def hang(_self: ClientSession) -> None:
        probing.set()
        await asyncio.sleep(LIMIT * 2)

    monkeypatch.setattr(ClientSession, 'send_ping', hang)
    connection = McpConnection(config(tmp_path, MCP_FAKE_APP_ERROR='1'))
    waiter = asyncio.create_task(connection.list_tools())
    await asyncio.wait_for(probing.wait(), timeout=LIMIT)

    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(waiter, timeout=LIMIT)


@pytest.mark.asyncio
async def test_each_released_caller_gets_its_own_error(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

    blocker = await hold_in_flight(connection)

    async def unreachable(session: ClientSession) -> list[McpTool]:
        return (await session.list_tools()).tools

    loop = asyncio.get_running_loop()
    queued: list[asyncio.Future[list[McpTool]]] = [loop.create_future() for _ in range(2)]

    await asyncio.wait_for(
        close_with_queued(connection, *(_Request(unreachable, result) for result in queued)), timeout=LIMIT
    )
    errors = []
    for result in queued:
        with pytest.raises(McpConnectionClosedError, match='closed') as raised:
            await asyncio.wait_for(result, timeout=LIMIT)
        errors.append(raised.value)
    assert errors[0] is not errors[1]
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(blocker.result, timeout=LIMIT)


@pytest.mark.asyncio
async def test_a_tool_call_error_code_does_not_poison_a_live_connection(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path, MCP_FAKE_APP_ERROR_CALL='1'))
    try:
        with pytest.raises(McpError, match='application-level failure'):
            await asyncio.wait_for(connection.call_tool('echo', {}), timeout=LIMIT)
        assert connection._state is _ConnectionState.OPEN
        assert is_running(await server_pid(tmp_path / 'server.pid'))
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_settle_makes_a_silent_owner_exit_terminal(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

        async def unreachable(session: ClientSession) -> list[McpTool]:
            return (await session.list_tools()).tools

        queued: asyncio.Future[list[McpTool]] = asyncio.get_running_loop().create_future()
        connection._queue.put_nowait(_Request(unreachable, queued))

        # What the owner's finally does when anyio's cancel scopes absorb
        # whatever ended the serve loop.
        connection._settle(None)

        assert connection._state is _ConnectionState.FAILED
        with pytest.raises(McpConnectionFailedError, match='failed'):
            await asyncio.wait_for(queued, timeout=LIMIT)
        with pytest.raises(McpConnectionFailedError, match='failed'):
            await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('state', 'error', 'message'),
    [
        (_ConnectionState.CLOSED, McpConnectionClosedError, 'closed'),
        (_ConnectionState.FAILED, McpConnectionFailedError, 'failed'),
    ],
)
async def test_request_rejects_a_state_change_after_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: _ConnectionState,
    error: type[Exception],
    message: str,
) -> None:
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    started = connection._start

    async def start_then_terminate() -> None:
        await started()
        connection._state = state

    monkeypatch.setattr(connection, '_start', start_then_terminate)
    with pytest.raises(error, match=message):
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

    connection._state = _ConnectionState.OPEN
    await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_startup_failure_is_terminal_and_names_the_command(tmp_path: Path) -> None:
    missing = tmp_path / 'no-such-mcp-server'
    connection = McpConnection(McpStdioServerConfig(command=str(missing)))
    unhandled: list[dict[str, object]] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unhandled.append(context))

    with pytest.raises(McpConnectionFailedError, match='failed') as failure:
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    assert str(missing) in str(failure.value)
    assert isinstance(failure.value.__cause__, OSError)
    assert connection._state is _ConnectionState.FAILED

    with pytest.raises(McpConnectionFailedError, match='failed'):
        await asyncio.wait_for(connection.call_tool('echo', {}), timeout=LIMIT)

    await asyncio.wait_for(connection.close(), timeout=LIMIT)
    del connection, failure
    gc.collect()
    assert unhandled == []


def test_a_connection_belongs_to_the_loop_that_opened_it(tmp_path: Path) -> None:
    pids: list[int] = []
    errors: list[BaseException] = []

    async def use_connection(index: int) -> None:
        pid_file = tmp_path / f'server-{index}.pid'
        connection = McpConnection(config(tmp_path, pid_file=pid_file))
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
        pids.append(await server_pid(pid_file))
        await asyncio.wait_for(connection.close(), timeout=LIMIT)

    def threaded() -> None:
        try:
            asyncio.run(use_connection(1))
        except BaseException as error:  # pragma: no cover - assertion reports it
            errors.append(error)

    asyncio.run(use_connection(0))
    thread = threading.Thread(target=threaded)
    thread.start()
    thread.join(timeout=LIMIT * 2)

    assert not thread.is_alive()
    assert errors == []
    assert len(set(pids)) == 2
    assert [is_running(pid) for pid in pids] == [False, False]


@pytest.mark.asyncio
async def test_a_request_timeout_fails_that_call_alone(tmp_path: Path) -> None:
    """A tool which never replies gives up without taking the connection with it."""
    connection = McpConnection(config(tmp_path), request_timeout_millis=3000)
    try:
        with pytest.raises(McpError, match='Timed out') as timed_out:
            await asyncio.wait_for(connection.call_tool('stall', {}), timeout=LIMIT)
        assert timed_out.value.error.code == 408

        assert connection._state is _ConnectionState.OPEN
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_a_handshake_which_never_answers_gives_up(tmp_path: Path) -> None:
    """Without the timeout a server that never initializes parks every caller forever."""
    connection = McpConnection(config(tmp_path, MCP_FAKE_STALL='1'), request_timeout_millis=100)

    with pytest.raises(McpConnectionFailedError, match='failed') as failure:
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)

    causes = leaf_causes(failure.value.__cause__)
    assert [error.error.code for error in causes if isinstance(error, McpError)] == [408]
    await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_no_request_timeout_waits_for_a_slow_server(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path), request_timeout_millis=None)
    try:
        assert text_of(await asyncio.wait_for(connection.call_tool('wait', {}), timeout=LIMIT)) == 'wait'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_an_owner_cancelled_from_outside_never_hands_on_its_cancellation(tmp_path: Path) -> None:
    """A caller that was not cancelled must not be told that it was."""
    connection = McpConnection(config(tmp_path))
    await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
    owner = connection._owner_task
    assert owner is not None

    call = asyncio.create_task(connection.call_tool('wait', {}))
    await asyncio.sleep(0.05)
    owner.cancel()

    with pytest.raises(McpConnectionFailedError, match='failed'):
        await asyncio.wait_for(call, timeout=LIMIT)
    # Terminal before the caller hears, or the client hands the next one a corpse.
    assert connection.is_terminal
    await settle(owner)
    await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_undecodable_output_does_not_kill_the_transport(tmp_path: Path) -> None:
    """One bad byte on stdout costs that message, not the connection."""
    connection = McpConnection(config(tmp_path, MCP_FAKE_BAD_UTF8='1'))
    try:
        assert [tool.name for tool in await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)] == ALL_TOOLS
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_the_server_is_told_which_client_connected(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        seen = text_of(await asyncio.wait_for(connection.call_tool('who_called', {}), timeout=LIMIT))
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)

    assert seen == f'genkit-mcp {metadata.version("genkit-mcp")}'


@pytest.mark.asyncio
async def test_two_calls_on_one_connection_overlap_in_time(tmp_path: Path) -> None:
    """The server answers rendezvous only once both calls have reached it."""
    connection = McpConnection(config(tmp_path))
    try:
        first, second = await asyncio.wait_for(
            asyncio.gather(connection.call_tool('rendezvous', {}), connection.call_tool('rendezvous', {})),
            timeout=LIMIT,
        )
        assert text_of(first) == '2'
        assert text_of(second) == '2'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


@pytest.mark.asyncio
async def test_list_tools_completes_while_a_call_is_stalled(tmp_path: Path) -> None:
    """The server answers await_stall only once the stall call has reached it."""
    connection = McpConnection(config(tmp_path))
    stalled = asyncio.create_task(connection.call_tool('stall', {}))
    try:
        assert text_of(await asyncio.wait_for(connection.call_tool('await_stall', {}), timeout=LIMIT)) == 'stalled'
        assert [tool.name for tool in await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)] == ALL_TOOLS
        assert not stalled.done()
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)
    with pytest.raises(McpConnectionClosedError, match='closed'):
        await asyncio.wait_for(stalled, timeout=LIMIT)


@pytest.mark.asyncio
async def test_an_output_schema_violation_is_a_tool_result_error(tmp_path: Path) -> None:
    connection = McpConnection(config(tmp_path))
    try:
        # The SDK validates against the schemas it has listed, and its own refresh reads one tools/list page.
        await asyncio.wait_for(connection.list_tools(), timeout=LIMIT)
        with pytest.raises(
            McpToolResultError, match='Invalid structured content returned by tool bad_output'
        ) as raised:
            await asyncio.wait_for(connection.call_tool('bad_output', {}), timeout=LIMIT)
        assert isinstance(raised.value.__cause__, RuntimeError)

        assert connection._state is _ConnectionState.OPEN
        alive = await asyncio.wait_for(connection.call_tool('echo', {'message': 'alive'}), timeout=LIMIT)
        assert text_of(alive) == 'alive'
    finally:
        await asyncio.wait_for(connection.close(), timeout=LIMIT)


class _RaisingSession:
    """Stands in for a ClientSession whose call_tool raises what the test chooses."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def call_tool(self, name: str, arguments: dict[str, object], meta: dict[str, object] | None) -> None:
        raise self.error


@pytest.mark.asyncio
async def test_other_runtime_errors_from_the_session_are_not_translated() -> None:
    session = cast(ClientSession, _RaisingSession(RuntimeError('Unsupported protocol version from the server: 0')))
    with pytest.raises(RuntimeError, match='Unsupported protocol version') as raised:
        await _call_tool(session, 'echo', {}, None)
    assert type(raised.value) is RuntimeError


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'message',
    [
        'Tool echo has an output schema but did not return structured content',
        'Invalid structured content returned by tool echo: 1 is not of type string',
        'Invalid schema for tool echo: bad schema',
    ],
)
async def test_each_output_schema_wording_is_translated(message: str) -> None:
    session = cast(ClientSession, _RaisingSession(RuntimeError(message)))
    with pytest.raises(McpToolResultError) as raised:
        await _call_tool(session, 'echo', {}, None)
    assert str(raised.value) == message
    assert isinstance(raised.value.__cause__, RuntimeError)
