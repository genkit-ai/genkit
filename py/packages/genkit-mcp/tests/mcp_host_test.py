# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for several MCP servers behind one dynamic action provider."""

import asyncio
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import anyio
import psutil
import pytest
import structlog
from genkit_mcp import McpClient, McpHost, define_mcp_host
from genkit_mcp._config import McpServerEntry
from pydantic import ValidationError
from structlog.testing import capture_logs

from genkit import Genkit, GenkitError, Message, ModelResponse, Part, Tool
from genkit._ai._generate import expand_wildcard_tools
from genkit._core._action import ActionKind
from genkit._core._typing import (
    FinishReason,
    Role,
    ToolRequest,
)
from genkit.testing import define_scripted_model

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

# A listing must not expire by the clock in these tests: every re-list they
# assert is one the host asked for.
LONG_TTL_MILLIS = 600_000

# Long enough for a healthy test server to start and answer on a slow machine,
# since the same timeout bounds its handshake.
STALL_TIMEOUT_MILLIS = 3_000

# Bounds every wait on a test server, so a regression fails rather than hangs.
LIMIT = 15

# Module level, so no test frame's locals can put it into a rendered traceback.
SECRET = 'genkit-mcp-redaction-canary-7f3a'


def server(tmp_path: Path, pid_name: str = 'server', **env: str) -> McpServerEntry:
    """One mcpServers entry for the test server."""
    return {
        'command': sys.executable,
        'args': [str(Path(__file__).with_name('fake_server.py'))],
        'env': {
            **os.environ,
            'MCP_FAKE_PID_FILE': str(tmp_path / f'{pid_name}.pid'),
            **env,
        },
    }


def servers(tmp_path: Path, *names: str, **env: str) -> dict[str, McpServerEntry]:
    """An mcpServers mapping holding one test server per name, each writing its own pid file."""
    return {name: server(tmp_path, pid_name=name, **env) for name in names}


def missing_server(tmp_path: Path) -> McpServerEntry:
    """A server whose command does not exist, so it can never start."""
    return {'command': str(tmp_path / 'no-such-server')}


def rendered_for_logs(error: BaseException) -> str:
    """Print the error's whole chain with frame locals, as structlog's default renderer does for logger.exception."""
    output = io.StringIO()
    saved = structlog.get_config()
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(output))
    try:
        structlog.get_logger('test').exception('listing failed', exc_info=(type(error), error, error.__traceback__))
    finally:
        structlog.configure(**saved)
    captured = traceback.TracebackException(type(error), error, error.__traceback__, capture_locals=True)
    return output.getvalue() + ''.join(captured.format(chain=True))


async def stalled(marker: Path) -> None:
    """Wait until a server started with ``MCP_FAKE_STALL_LIST`` is stuck in its listing."""

    async def poll() -> None:
        while True:
            try:
                await anyio.Path(marker).read_text(encoding='utf-8')
                return
            except FileNotFoundError:
                await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=LIMIT)


def server_processes() -> list[psutil.Process]:
    """Every fake MCP server this test process still owns."""
    running: list[psutil.Process] = []
    for child in psutil.Process().children(recursive=True):
        try:
            if any('fake_server.py' in argument for argument in child.cmdline()):
                running.append(child)
        except psutil.Error:  # pragma: no cover - the child exited mid-scan
            continue
    return running


def server_process(tmp_path: Path, pid_name: str) -> psutil.Process:
    """The process of one named test server, from the pid file it wrote."""
    return psutil.Process(int((tmp_path / f'{pid_name}.pid').read_text(encoding='utf-8')))


def tool_names(tools: list[Tool]) -> list[str]:
    """Names of the listed tools, as a model is shown them."""
    return [tool.name for tool in tools]


def hosted(*servers: str) -> list[str]:
    """Every tool name a host holding these servers serves, in listing order."""
    return [f'{server}_{tool}' for server in servers for tool in ALL_TOOLS]


def text_response(text: str) -> ModelResponse:
    """Build a model response carrying one text part."""
    return ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(text=text)]),
        finish_reason=FinishReason.STOP,
    )


def tool_call_response(tool_name: str, input: dict[str, Any]) -> ModelResponse:
    """Build a model response requesting one tool call."""
    return ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=tool_name, input=input, ref=tool_name))],
        ),
        finish_reason=FinishReason.STOP,
    )


@pytest.mark.asyncio
async def test_every_server_is_listed_under_its_own_name(tmp_path: Path) -> None:
    """One host, two servers, and each tool named after the server serving it."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    try:
        assert list(host._clients) == ['alpha', 'beta']
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha', 'beta')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_two_servers_whose_tool_names_collide_cannot_be_listed(tmp_path: Path) -> None:
    """Server name and tool join on an underscore either may hold, so distinct names still collide."""
    host = McpHost(
        'hosted',
        mcp_servers={
            'alpha_beta': server(tmp_path, pid_name='alpha_beta'),
            'alpha': server(tmp_path, pid_name='alpha', MCP_FAKE_TOOL_NAME_PREFIX='beta_'),
        },
    )
    try:
        assert host.tool_name('alpha_beta', 'echo') == host.tool_name('alpha', 'beta_echo')
        with pytest.raises(
            GenkitError, match="servers 'alpha_beta' and 'alpha' together: both serve a tool named 'alpha_beta_echo'"
        ) as refused:
            await host.get_active_tools()
        assert refused.value.status == 'INVALID_ARGUMENT'
    finally:
        await host.close()

    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_server_which_cannot_start_leaves_the_others_listed(tmp_path: Path) -> None:
    """One unreachable server is skipped, not fatal: it must not blank the catalog."""
    host = McpHost(
        'hosted',
        mcp_servers={'broken': missing_server(tmp_path), **servers(tmp_path, 'alpha')},
    )
    try:
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


@pytest.mark.asyncio
async def test_a_server_without_the_tools_capability_leaves_the_others_listed(tmp_path: Path) -> None:
    host = McpHost(
        'hosted',
        mcp_servers={**servers(tmp_path, 'resources', MCP_FAKE_NO_TOOLS='1'), **servers(tmp_path, 'alpha')},
    )
    try:
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


@pytest.mark.asyncio
async def test_servers_which_do_not_answer_in_time_leave_the_others_listed(tmp_path: Path) -> None:
    """A server stuck in its handshake or its listing is as unreachable as one that cannot start."""
    host = McpHost(
        'hosted',
        mcp_servers={
            **servers(tmp_path, 'alpha'),
            **servers(tmp_path, 'handshake', MCP_FAKE_STALL='1'),
            **servers(tmp_path, 'listing', MCP_FAKE_STALL_LIST=str(tmp_path / 'listing.stalled')),
        },
        request_timeout_millis=STALL_TIMEOUT_MILLIS,
    )
    try:
        with capture_logs() as logs:
            tools = await asyncio.wait_for(host.get_active_tools(), timeout=LIMIT)
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')
    skipped = {log['server']: log['status'] for log in logs if log['event'].startswith('MCP server could not be')}
    assert skipped == {'handshake': 'UNAVAILABLE', 'listing': 'DEADLINE_EXCEEDED'}
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_skipped_server_is_logged_without_its_env(tmp_path: Path) -> None:
    """The skip log carries the error, and structlog's default renderer prints every frame's locals in its chain."""
    broken: McpServerEntry = {'command': str(tmp_path / 'no-such-server'), 'env': {'GENKIT_MCP_TEST_API_KEY': SECRET}}
    host = McpHost('hosted', mcp_servers={'broken': broken})
    try:
        with capture_logs() as logs:
            tools = await asyncio.wait_for(host.get_active_tools(), timeout=LIMIT)
    finally:
        await host.close()

    assert tools == []
    [skipped] = [log for log in logs if log['event'].startswith('MCP server could not be')]
    assert skipped['status'] == 'UNAVAILABLE'
    assert SECRET not in rendered_for_logs(skipped['exc_info'])


@pytest.mark.asyncio
async def test_a_server_disconnected_during_a_listing_is_left_out_of_it(tmp_path: Path) -> None:
    """The listing in flight still holds the old client, and that client's closed error is not a failure."""
    marker = tmp_path / 'beta.stalled'
    host = McpHost(
        'hosted',
        mcp_servers={**servers(tmp_path, 'alpha'), **servers(tmp_path, 'beta', MCP_FAKE_STALL_LIST=str(marker))},
    )
    try:
        listing = asyncio.create_task(host.get_active_tools())
        await stalled(marker)
        await host.disconnect('beta')
        tools = await asyncio.wait_for(listing, timeout=LIMIT)
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_host_closed_during_a_listing_fails_it_as_closed(tmp_path: Path) -> None:
    """Every server left the host, but the listing must not pass for an empty catalog."""
    marker = tmp_path / 'alpha.stalled'
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha', MCP_FAKE_STALL_LIST=str(marker)))
    listing = asyncio.create_task(host.get_active_tools())
    await stalled(marker)
    await host.close()

    with pytest.raises(GenkitError, match="MCP host 'hosted' is closed") as raised:
        await asyncio.wait_for(listing, timeout=LIMIT)
    assert raised.value.status == 'FAILED_PRECONDITION'
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_wildcard_selector_binds_tools_from_every_server(tmp_path: Path) -> None:
    """``<host>:tool/*`` reaches every server behind the one provider."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    try:
        expanded = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha', 'beta'))


@pytest.mark.asyncio
async def test_generate_calls_a_tool_on_one_hosted_server(tmp_path: Path) -> None:
    """A model sees every hosted tool and reads back what one of them returned."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    pm.responses = [
        tool_call_response('beta_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        response = await ai.generate(model='scriptedModel', prompt='echo hello', tools=['hosted:tool/*'])
    finally:
        await host.close()

    assert response.text == 'done'
    assert pm.request_count == 2
    assert pm.last_request is not None
    assert sorted(tool.name for tool in pm.last_request.tools or []) == sorted(hosted('alpha', 'beta'))
    tool_message = pm.last_request.messages[-1]
    assert tool_message.role == Role.TOOL
    assert tool_message.content[0].tool_response is not None
    assert tool_message.content[0].tool_response.output == 'hello'


@pytest.mark.asyncio
async def test_generate_accepts_the_listed_tools(tmp_path: Path) -> None:
    """The listing is made of tool handles, so it goes into ``generate`` as it is."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    pm.responses = [
        tool_call_response('beta_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        tools = await host.get_active_tools()
        response = await ai.generate(model='scriptedModel', prompt='echo hello', tools=tools)
    finally:
        await host.close()

    assert response.text == 'done'
    assert pm.last_request is not None
    assert sorted(tool.name for tool in pm.last_request.tools or []) == sorted(hosted('alpha', 'beta'))
    tool_message = pm.last_request.messages[-1]
    assert tool_message.content[0].tool_response is not None
    assert tool_message.content[0].tool_response.output == 'hello'


@pytest.mark.asyncio
async def test_tool_name_builds_a_selector_for_one_hosted_tool(tmp_path: Path) -> None:
    """A selector built from the helpers reaches one tool of one server."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    pm.responses = [
        tool_call_response('beta_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        selector = f'{host.name}:tool/{host.tool_name("beta", "echo")}'
        response = await ai.generate(model='scriptedModel', prompt='echo hello', tools=[selector])
    finally:
        await host.close()

    assert selector == 'hosted:tool/beta_echo'
    assert response.text == 'done'
    assert pm.last_request is not None
    assert [tool.name for tool in pm.last_request.tools or []] == ['beta_echo']


@pytest.mark.asyncio
async def test_connect_adds_a_server_and_drops_the_listing(tmp_path: Path) -> None:
    """A server added after registration is selectable without waiting for the TTL."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha'), cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        before = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        await host.connect('beta', server(tmp_path, pid_name='beta'))
        after = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert list(host._clients) == ['alpha', 'beta']
    finally:
        await host.close()

    assert sorted(before) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert sorted(after) == sorted(f'/tool.v2/{name}' for name in hosted('alpha', 'beta'))


@pytest.mark.asyncio
async def test_connect_takes_an_entry_of_an_mcp_servers_block(tmp_path: Path) -> None:
    """One input shape: an entry the constructor accepts, ``type`` included, connects as it is."""
    config = json.loads(json.dumps({'mcpServers': {'beta': {'type': 'stdio', **server(tmp_path, pid_name='beta')}}}))
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        await host.connect('beta', config['mcpServers']['beta'])
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha', 'beta')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_connect_refuses_an_entry_which_is_not_stdio_by_name(tmp_path: Path) -> None:
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        with pytest.raises(ValueError, match="MCP host 'hosted': server 'linear' is not a stdio entry"):
            await host.connect('linear', json.loads(json.dumps({'url': 'https://mcp.example.com/sse'})))
        assert list(host._clients) == ['alpha']
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_connect_replaces_a_server_of_the_same_name(tmp_path: Path) -> None:
    """Reconfiguring a name starts the new server, then stops the one it replaces."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        await host.get_active_tools()
        assert len(server_processes()) == 1

        await host.connect('alpha', server(tmp_path, pid_name='replacement'))
        assert [process.pid for process in server_processes()] == [server_process(tmp_path, 'replacement').pid]

        tools = await host.get_active_tools()
        assert len(server_processes()) == 1
        assert list(host._clients) == ['alpha']
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


@pytest.mark.asyncio
async def test_connect_reports_a_server_which_cannot_be_started_and_keeps_the_host_as_it_was(tmp_path: Path) -> None:
    """A typo in a command fails the call that added it, not a later generate, and costs nothing else."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha'), cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        alpha = host._clients['alpha']
        running = server_process(tmp_path, 'alpha')

        with pytest.raises(GenkitError, match='failed') as adding:
            await host.connect('beta', missing_server(tmp_path))
        with pytest.raises(GenkitError, match='failed') as replacing:
            await host.connect('alpha', missing_server(tmp_path))

        assert adding.value.status == replacing.value.status == 'UNAVAILABLE'
        assert host._clients == {'alpha': alpha}
        assert [process.pid for process in server_processes()] == [running.pid]
        expanded = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert server_processes() == []


@pytest.mark.asyncio
async def test_connect_keeps_the_env_of_a_server_which_cannot_be_started_out_of_its_traceback(tmp_path: Path) -> None:
    """A flow awaiting connect fails with this error, and core logs a failed action with exc_info."""
    host = McpHost('hosted', mcp_servers={})
    try:
        with pytest.raises(GenkitError, match='failed') as failure:
            await asyncio.wait_for(
                host.connect(
                    'broken', {'command': str(tmp_path / 'no-such-server'), 'env': {'GENKIT_MCP_TEST_API_KEY': SECRET}}
                ),
                timeout=LIMIT,
            )
    finally:
        await host.close()

    assert failure.value.status == 'UNAVAILABLE'
    assert SECRET not in rendered_for_logs(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('server_name', 'extra'),
    [('bad', {'bogus': 1}), ('bad name', {})],
)
async def test_connect_keeps_the_env_of_a_refused_entry_out_of_its_traceback(
    tmp_path: Path, server_name: str, extra: dict[str, Any]
) -> None:
    host = McpHost('hosted', mcp_servers={})
    try:
        with pytest.raises(ValueError) as refused:
            await host.connect(
                server_name,
                json.loads(json.dumps({'command': 'npx', 'env': {'GENKIT_MCP_TEST_API_KEY': SECRET}, **extra})),
            )
    finally:
        await host.close()

    assert SECRET not in rendered_for_logs(refused.value)


@pytest.mark.asyncio
async def test_connect_to_a_host_closed_meanwhile_stops_the_new_server(tmp_path: Path) -> None:
    """A server listed after the host closed would otherwise run with nothing left to stop it."""
    host = McpHost('hosted', mcp_servers={})
    list_actions = McpClient._list_actions

    async def close_host_then_list(client: McpClient) -> list[Any]:
        await host.close()
        return await list_actions(client)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(McpClient, '_list_actions', close_host_then_list)
        with pytest.raises(GenkitError, match="MCP host 'hosted' is closed") as raised:
            await host.connect('alpha', server(tmp_path, pid_name='alpha'))

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert host._clients == {}
    assert server_processes() == []


@pytest.mark.asyncio
async def test_connect_installs_the_replacement_when_the_replaced_server_fails_to_close(tmp_path: Path) -> None:
    """A close that raises still hands the name to the new server and drops the listing."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha'), cache_ttl_millis=LONG_TTL_MILLIS)
    close = McpClient.close

    async def close_and_complain(client: McpClient) -> None:
        await close(client)
        raise RuntimeError('close failed after the fact')

    try:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(McpClient, 'close', close_and_complain)
            with pytest.raises(RuntimeError, match='close failed after the fact'):
                await host.connect('alpha', server(tmp_path, pid_name='replacement'))
        assert list(host._clients) == ['alpha']
        assert [process.pid for process in server_processes()] == [server_process(tmp_path, 'replacement').pid]

        expanded = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert server_processes() == []


@pytest.mark.asyncio
async def test_disconnect_removes_one_server_and_stops_it(tmp_path: Path) -> None:
    """The other servers keep running, and the listing loses only the one removed."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers=servers(tmp_path, 'alpha', 'beta'),
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert len(server_processes()) == 2

        await host.disconnect('beta')

        assert list(host._clients) == ['alpha']
        assert [process.pid for process in server_processes()] == [server_process(tmp_path, 'alpha').pid]
        expanded = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert server_processes() == []


@pytest.mark.asyncio
async def test_reconnect_replaces_one_server_process(tmp_path: Path) -> None:
    """Reconnecting restarts that server alone and keeps the tools it serves."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    try:
        listed = tool_names(await host.get_active_tools())
        alpha = server_process(tmp_path, 'alpha')
        beta = server_process(tmp_path, 'beta')

        await host.reconnect('alpha')

        restarted = server_process(tmp_path, 'alpha')
        assert restarted.pid != alpha.pid
        assert sorted(process.pid for process in server_processes()) == sorted([restarted.pid, beta.pid])
        assert tool_names(await host.get_active_tools()) == listed
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_reconnect_reports_a_server_which_cannot_be_started(tmp_path: Path) -> None:
    """The caller of reconnect hears about a broken server, and keeps holding it."""
    host = McpHost('hosted', mcp_servers={'broken': missing_server(tmp_path)})
    try:
        with pytest.raises(GenkitError, match='failed') as raised:
            await host.reconnect('broken')
        assert raised.value.status == 'UNAVAILABLE'
        assert list(host._clients) == ['broken']
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_close_is_final(tmp_path: Path) -> None:
    """A closed host stops every server and refuses every operation, as a client does."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    assert tool_names(await host.get_active_tools()) == hosted('alpha', 'beta')
    assert len(server_processes()) == 2

    await host.close()
    await host.close()

    assert server_processes() == []
    closed = "MCP host 'hosted' is closed"
    with pytest.raises(GenkitError, match=closed) as listing:
        await host.get_active_tools()
    with pytest.raises(GenkitError, match=closed) as selecting:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    with pytest.raises(GenkitError, match=closed) as connecting:
        await host.connect('gamma', server(tmp_path, pid_name='gamma'))
    with pytest.raises(GenkitError, match=closed) as reconnecting:
        await host.reconnect('alpha')
    with pytest.raises(GenkitError, match=closed) as disconnecting:
        await host.disconnect('alpha')
    with pytest.raises(GenkitError, match=closed) as naming:
        host.tool_name('alpha', 'echo')
    for raised in (listing, selecting, connecting, reconnecting, disconnecting, naming):
        assert raised.value.status == 'FAILED_PRECONDITION'
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_failed_tool_call_re_lists_every_server(tmp_path: Path) -> None:
    """One host, one provider: a server that dies costs a re-list of all of them."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers={
            **servers(tmp_path, 'alpha', MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
            **servers(tmp_path, 'beta'),
        },
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        counter = await ai._registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai._registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        killed = server_process(tmp_path, 'beta')
        killed.kill()
        killed.wait(timeout=30)
        with pytest.raises(GenkitError) as raised:
            await beta_echo.run({'message': 'second'})
        assert raised.value.status == 'UNAVAILABLE'

        # The listing outlives its TTL, so re-listing here is beta's failure
        # invalidating the listing alpha shares with it.
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_server_connected_later_shares_the_same_listing(tmp_path: Path) -> None:
    """A server added after registration drops that one listing too when it fails."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers=servers(tmp_path, 'alpha', MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await host.connect('beta', server(tmp_path, pid_name='beta'))
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        counter = await ai._registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai._registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        killed = server_process(tmp_path, 'beta')
        killed.kill()
        killed.wait(timeout=30)
        with pytest.raises(GenkitError) as raised:
            await beta_echo.run({'message': 'second'})
        assert raised.value.status == 'UNAVAILABLE'

        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_disconnected_server_cannot_drop_the_listing(tmp_path: Path) -> None:
    """A server the host has dropped stops speaking for the ones it left behind."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers={
            **servers(tmp_path, 'alpha', MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
            **servers(tmp_path, 'beta'),
        },
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        counter = await ai._registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai._registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        await host.disconnect('beta')
        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'

        with pytest.raises(GenkitError) as raised:
            await beta_echo.run({'message': 'late'})
        assert raised.value.status == 'FAILED_PRECONDITION'

        await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['', 'my/host', 'my:host'])
async def test_a_host_name_no_selector_could_carry_is_refused(tmp_path: Path, name: str) -> None:
    """A host name is a provider name, which is looser than a tool name."""
    with pytest.raises(ValueError, match='cannot name an MCP provider'):
        McpHost(name, mcp_servers=servers(tmp_path, 'alpha'))
    assert server_processes() == []


@pytest.mark.parametrize('timeout', [0, -1])
def test_a_host_with_no_servers_yet_rejects_a_timeout_that_is_not_positive(timeout: int) -> None:
    """Servers connected later would inherit it, so it is refused before the first one."""
    with pytest.raises(ValueError, match='request_timeout_millis'):
        define_mcp_host(Genkit(), 'hosted', mcp_servers={}, request_timeout_millis=timeout)


@pytest.mark.asyncio
async def test_a_host_name_a_model_would_refuse_is_accepted(tmp_path: Path) -> None:
    """A host name reaches no model, so the tool name rules do not apply to it."""
    ai = Genkit()
    host = define_mcp_host(ai, 'my.host', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        assert host.name == 'my.host'
        expanded = await expand_wildcard_tools(ai._registry, ['my.host:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))


@pytest.mark.asyncio
@pytest.mark.parametrize('server_name', ['my server', 'a:b', 'a/b', '9lives', ''])
async def test_a_server_name_no_model_would_accept_is_refused(tmp_path: Path, server_name: str) -> None:
    """A server name becomes a tool prefix, and the host, not a provider, is what refuses it."""
    refused = f'{server_name!r} cannot prefix an MCP tool name'
    with pytest.raises(ValueError, match=re.escape(refused)):
        McpHost('hosted', mcp_servers={server_name: server(tmp_path)})

    host = McpHost('hosted', mcp_servers={})
    try:
        with pytest.raises(ValueError, match=re.escape(refused)):
            await host.connect(server_name, server(tmp_path))
        assert list(host._clients) == []
    finally:
        await host.close()
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_server_this_host_does_not_hold_is_refused(tmp_path: Path) -> None:
    """Every operation naming a server says so when the host holds no such name."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        with pytest.raises(ValueError, match='holds no server named'):
            host.tool_name('beta', 'echo')
        with pytest.raises(ValueError, match='holds no server named'):
            await host.disconnect('beta')
        with pytest.raises(ValueError, match='holds no server named'):
            await host.reconnect('beta')
        assert list(host._clients) == ['alpha']
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_an_unregistered_host_registers_nothing(tmp_path: Path) -> None:
    """A host built directly puts nothing in a registry, so no selector reaches its servers."""
    ai = Genkit()
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        assert host._provider is None
        assert await ai._registry.resolve_action_by_key('/dynamic-action-provider/hosted:tool/alpha_echo') is None
        assert await expand_wildcard_tools(ai._registry, ['hosted:tool/*']) == ['hosted:tool/*']
        assert server_processes() == []
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_an_mcp_servers_block_of_stdio_servers_is_accepted_as_it_is(tmp_path: Path) -> None:
    """A Claude Desktop or Cursor block, ``type`` marker included, is a valid mcp_servers value."""
    block = json.loads(json.dumps({'mcpServers': {'alpha': {'type': 'stdio', **server(tmp_path, pid_name='alpha')}}}))
    host = McpHost('hosted', mcp_servers=block['mcpServers'])
    try:
        assert tool_names(await host.get_active_tools()) == hosted('alpha')
    finally:
        await host.close()

    assert server_processes() == []


@pytest.mark.asyncio
async def test_an_entry_with_an_unknown_key_is_refused_without_echoing_it(tmp_path: Path) -> None:
    """A misspelt env must not launch the server without its secret, nor print the secret."""
    secret = 'sk-distinctive-secret-7f3a'
    loaded: dict[str, Any] = json.loads(json.dumps({'command': 'npx', 'enviroment': {'API_KEY': secret}}))
    with pytest.raises(ValueError, match="MCP host 'hosted': server 'alpha' is not a stdio entry") as raised:
        McpHost('hosted', mcp_servers={'alpha': loaded})
    assert 'enviroment' in str(raised.value)
    assert secret not in str(raised.value)
    assert isinstance(raised.value.__cause__, ValidationError)
    assert server_processes() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'entry',
    [
        {'url': 'https://mcp.example.com/sse?token=sk-distinctive-secret-7f3a'},
        {'type': 'sse', 'command': 'npx'},
        {'command': 'npx', 'disabled': True},
    ],
)
async def test_an_entry_which_is_not_stdio_is_refused_by_name(tmp_path: Path, entry: dict[str, Any]) -> None:
    """In a block of many servers, the error has to say which one is not stdio, without echoing it."""
    with pytest.raises(ValueError, match="MCP host 'hosted': server 'linear' is not a stdio entry") as raised:
        McpHost('hosted', mcp_servers={**servers(tmp_path, 'alpha'), 'linear': json.loads(json.dumps(entry))})
    assert 'sk-distinctive-secret-7f3a' not in str(raised.value)
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_server_breaking_the_protocol_fails_the_listing(tmp_path: Path) -> None:
    """Only a server that cannot be reached is skipped: a broken one is reported, not hidden."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers={**servers(tmp_path, 'alpha'), **servers(tmp_path, 'beta', MCP_FAKE_STUCK_CURSOR='1')},
    )
    try:
        with pytest.raises(GenkitError, match='repeated the tools/list cursor') as raised:
            await host.get_active_tools()
        assert raised.value.status == 'INTERNAL'
        with pytest.raises(GenkitError, match='repeated the tools/list cursor'):
            await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_server_error_answering_the_listing_fails_it(tmp_path: Path) -> None:
    """A server that answers tools/list with an error is up, so its error reaches the caller."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        mcp_servers={**servers(tmp_path, 'alpha'), **servers(tmp_path, 'beta', MCP_FAKE_APP_ERROR='1')},
    )
    try:
        with pytest.raises(GenkitError, match='application-level failure') as answered:
            await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert answered.value.status == 'UNKNOWN'
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_tools_of_one_server_sharing_a_genkit_name_fail_the_listing(tmp_path: Path) -> None:
    """A server whose own tools collide is not skipped: one of them would vanish silently."""
    host = McpHost(
        'hosted',
        mcp_servers={**servers(tmp_path, 'alpha'), **servers(tmp_path, 'beta', MCP_FAKE_COLLIDING_TOOLS='1')},
    )
    try:
        with pytest.raises(
            GenkitError, match=r"MCP tools 'look\.up' and 'look/up' both map to the Genkit tool name 'beta_look_up'"
        ) as refused:
            await host.get_active_tools()
        assert refused.value.status == 'INVALID_ARGUMENT'
    finally:
        await host.close()
