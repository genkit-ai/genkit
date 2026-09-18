# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for several MCP servers behind one dynamic action provider."""

import os
import sys
from pathlib import Path
from typing import Any

import psutil
import pytest
from genkit_mcp import (
    McpConnectionFailedError,
    McpHostServer,
    McpStdioServerConfig,
    create_mcp_host,
    define_mcp_host,
)

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import expand_wildcard_tools
from genkit._ai._testing import define_programmable_model
from genkit._core._action import Action, ActionKind
from genkit._core._error import GenkitError
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

ALL_TOOLS = ['echo', 'second', 'wait', 'list_requests', 'who_called', 'stall']

# A listing must not expire by the clock in these tests: every re-list they
# assert is one the host asked for.
LONG_TTL_MILLIS = 600_000


def config(tmp_path: Path, pid_name: str = 'server', **env: str) -> McpStdioServerConfig:
    """Launch the test server with the same interpreter as the test suite."""
    return McpStdioServerConfig(
        command=sys.executable,
        args=[str(Path(__file__).with_name('fake_server.py'))],
        env={
            **os.environ,
            'MCP_FAKE_PID_FILE': str(tmp_path / f'{pid_name}.pid'),
            **env,
        },
    )


def spec(name: str, tmp_path: Path, **env: str) -> McpHostServer:
    """Name one test server for a host to hold."""
    return McpHostServer(name=name, server=config(tmp_path, pid_name=name, **env))


def missing_server(tmp_path: Path) -> McpStdioServerConfig:
    """A server whose command does not exist, so it can never start."""
    return McpStdioServerConfig(command=str(tmp_path / 'no-such-server'))


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


def tool_names(tools: list[Action[Any, Any]]) -> list[str]:
    """Names of the listed tools, as a model is shown them."""
    return [tool.name for tool in tools]


def hosted(*servers: str) -> list[str]:
    """Every tool name a host holding these servers serves, in listing order."""
    return [f'{server}_{tool}' for server in servers for tool in ALL_TOOLS]


def text_response(text: str) -> ModelResponse:
    """Build a model response carrying one text part."""
    return ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        finish_reason=FinishReason.STOP,
    )


def tool_call_response(tool_name: str, input: dict[str, Any]) -> ModelResponse:
    """Build a model response requesting one tool call."""
    return ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name=tool_name, input=input, ref=tool_name)))],
        ),
        finish_reason=FinishReason.STOP,
    )


@pytest.mark.asyncio
async def test_every_server_is_listed_under_its_own_name(tmp_path: Path) -> None:
    """One host, two servers, and each tool named after the server serving it."""
    host = create_mcp_host('hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
    try:
        assert host.server_names == ['alpha', 'beta']
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha', 'beta')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_two_servers_cannot_share_a_name(tmp_path: Path) -> None:
    """A name namespaces tools, and a client knows nothing of its siblings."""
    with pytest.raises(ValueError, match='two servers named'):
        create_mcp_host('hosted', [spec('alpha', tmp_path), spec('alpha', tmp_path)])
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_server_which_cannot_start_leaves_the_others_listed(tmp_path: Path) -> None:
    """One unreachable server is skipped, not fatal: it must not blank the catalog."""
    host = create_mcp_host(
        'hosted',
        [
            McpHostServer(name='broken', server=missing_server(tmp_path)),
            spec('alpha', tmp_path),
        ],
    )
    try:
        tools = await host.get_active_tools()
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


@pytest.mark.asyncio
async def test_a_wildcard_selector_binds_tools_from_every_server(tmp_path: Path) -> None:
    """``<host>:tool/*`` reaches every server behind the one provider."""
    ai = Genkit()
    host = define_mcp_host(ai, 'hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
    try:
        expanded = await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha', 'beta'))


@pytest.mark.asyncio
async def test_generate_calls_a_tool_on_one_hosted_server(tmp_path: Path) -> None:
    """A model sees every hosted tool and reads back what one of them returned."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    host = define_mcp_host(ai, 'hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
    pm.responses = [
        tool_call_response('beta_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        response = await ai.generate(model='programmableModel', prompt='echo hello', tools=['hosted:tool/*'])
    finally:
        await host.close()

    assert response.text == 'done'
    assert pm.request_count == 2
    assert pm.last_request is not None
    assert sorted(tool.name for tool in pm.last_request.tools or []) == sorted(hosted('alpha', 'beta'))
    tool_message = pm.last_request.messages[-1]
    assert tool_message.role == Role.TOOL
    assert tool_message.content[0].root.tool_response is not None
    assert tool_message.content[0].root.tool_response.output == 'hello'


@pytest.mark.asyncio
async def test_tool_name_builds_a_selector_for_one_hosted_tool(tmp_path: Path) -> None:
    """A selector built from the helpers reaches one tool of one server."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    host = define_mcp_host(ai, 'hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
    pm.responses = [
        tool_call_response('beta_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        selector = f'{host.name}:tool/{host.tool_name("beta", "echo")}'
        response = await ai.generate(model='programmableModel', prompt='echo hello', tools=[selector])
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
    host = define_mcp_host(ai, 'hosted', [spec('alpha', tmp_path)], cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        before = await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        await host.connect('beta', config(tmp_path, pid_name='beta'))
        after = await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        assert host.server_names == ['alpha', 'beta']
    finally:
        await host.close()

    assert sorted(before) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert sorted(after) == sorted(f'/tool.v2/{name}' for name in hosted('alpha', 'beta'))


@pytest.mark.asyncio
async def test_connect_replaces_a_server_of_the_same_name(tmp_path: Path) -> None:
    """Reconfiguring a name stops the server it replaces, and the new one starts lazily."""
    host = create_mcp_host('hosted', [spec('alpha', tmp_path)])
    try:
        await host.get_active_tools()
        assert len(server_processes()) == 1

        await host.connect('alpha', config(tmp_path, pid_name='replacement'))
        assert server_processes() == []

        tools = await host.get_active_tools()
        assert len(server_processes()) == 1
        assert server_process(tmp_path, 'replacement').is_running()
        assert host.server_names == ['alpha']
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


@pytest.mark.asyncio
async def test_disconnect_removes_one_server_and_stops_it(tmp_path: Path) -> None:
    """The other servers keep running, and the listing loses only the one removed."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        [spec('alpha', tmp_path), spec('beta', tmp_path)],
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        assert len(server_processes()) == 2

        await host.disconnect('beta')

        assert host.server_names == ['alpha']
        assert [process.pid for process in server_processes()] == [server_process(tmp_path, 'alpha').pid]
        expanded = await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert server_processes() == []


@pytest.mark.asyncio
async def test_reconnect_replaces_one_server_process(tmp_path: Path) -> None:
    """Reconnecting restarts that server alone and keeps the tools it serves."""
    host = create_mcp_host('hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
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
    host = create_mcp_host('hosted', [McpHostServer(name='broken', server=missing_server(tmp_path))])
    try:
        with pytest.raises(McpConnectionFailedError, match='failed'):
            await host.reconnect('broken')
        assert host.server_names == ['broken']
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_close_releases_every_server_and_leaves_the_host_usable(tmp_path: Path) -> None:
    """Closing a host empties it rather than ending it: it can hold servers again."""
    host = create_mcp_host('hosted', [spec('alpha', tmp_path), spec('beta', tmp_path)])
    try:
        assert tool_names(await host.get_active_tools()) == hosted('alpha', 'beta')
        assert len(server_processes()) == 2

        await host.close()

        assert host.server_names == []
        assert await host.get_active_tools() == []
        assert server_processes() == []

        await host.connect('gamma', config(tmp_path, pid_name='gamma'))
        assert tool_names(await host.get_active_tools()) == hosted('gamma')
    finally:
        await host.close()

    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_failed_tool_call_re_lists_every_server(tmp_path: Path) -> None:
    """One host, one provider: a server that dies costs a re-list of all of them."""
    ai = Genkit()
    host = define_mcp_host(
        ai,
        'hosted',
        [
            spec('alpha', tmp_path, MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
            spec('beta', tmp_path),
        ],
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        counter = await ai.registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai.registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        killed = server_process(tmp_path, 'beta')
        killed.kill()
        killed.wait(timeout=30)
        with pytest.raises(GenkitError) as raised:
            await beta_echo.run({'message': 'second'})
        assert isinstance(raised.value.cause, McpConnectionFailedError)

        # The listing outlives its TTL, so re-listing here is beta's failure
        # invalidating the listing alpha shares with it.
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
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
        [spec('alpha', tmp_path, MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS)))],
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await host.connect('beta', config(tmp_path, pid_name='beta'))
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        counter = await ai.registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai.registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        killed = server_process(tmp_path, 'beta')
        killed.kill()
        killed.wait(timeout=30)
        with pytest.raises(GenkitError):
            await beta_echo.run({'message': 'second'})

        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
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
        [spec('alpha', tmp_path, MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))), spec('beta', tmp_path)],
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        counter = await ai.registry.resolve_action(ActionKind.TOOL, 'alpha_list_requests')
        beta_echo = await ai.registry.resolve_action(ActionKind.TOOL, 'beta_echo')
        assert counter is not None
        assert beta_echo is not None
        assert (await counter.run({})).response.output == '1'

        await host.disconnect('beta')
        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'

        with pytest.raises(GenkitError):
            await beta_echo.run({'message': 'late'})

        await expand_wildcard_tools(ai.registry, ['hosted:tool/*'])
        assert (await counter.run({})).response.output == '2'
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['', 'my/host', 'my:host'])
async def test_a_host_name_no_selector_could_carry_is_refused(tmp_path: Path, name: str) -> None:
    """A host name is a provider name, which is looser than a tool name."""
    with pytest.raises(ValueError, match='cannot name an MCP host'):
        create_mcp_host(name, [spec('alpha', tmp_path)])
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_host_name_a_model_would_refuse_is_accepted(tmp_path: Path) -> None:
    """A host name reaches no model, so the tool name rules do not apply to it."""
    ai = Genkit()
    host = define_mcp_host(ai, 'my.host', [spec('alpha', tmp_path)])
    try:
        assert host.name == 'my.host'
        expanded = await expand_wildcard_tools(ai.registry, ['my.host:tool/*'])
    finally:
        await host.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))


@pytest.mark.asyncio
async def test_a_server_name_no_model_would_accept_is_refused(tmp_path: Path) -> None:
    """A server name becomes a tool prefix, which a model provider has to accept."""
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        create_mcp_host('hosted', [McpHostServer(name='my server', server=config(tmp_path))])
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_supplied_prefix_replaces_the_server_name(tmp_path: Path) -> None:
    """A server names its tools after itself, unless the host gives it a prefix."""
    host = create_mcp_host(
        'hosted',
        [McpHostServer(name='shop', server=config(tmp_path), tool_prefix='bookshop')],
    )
    try:
        assert host.server_names == ['shop']
        assert host.tool_name('shop', 'echo') == 'bookshop_echo'
        assert tool_names(await host.get_active_tools()) == hosted('bookshop')
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_server_this_host_does_not_hold_is_refused(tmp_path: Path) -> None:
    """Every operation naming a server says so when the host holds no such name."""
    host = create_mcp_host('hosted', [spec('alpha', tmp_path)])
    try:
        with pytest.raises(ValueError, match='holds no server named'):
            host.tool_name('beta', 'echo')
        with pytest.raises(ValueError, match='holds no server named'):
            await host.disconnect('beta')
        with pytest.raises(ValueError, match='holds no server named'):
            await host.reconnect('beta')
        assert host.server_names == ['alpha']
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_created_host_registers_nothing(tmp_path: Path) -> None:
    """create_mcp_host puts nothing in a registry, so no selector reaches its servers."""
    ai = Genkit()
    host = create_mcp_host('hosted', [spec('alpha', tmp_path)])
    try:
        assert host.dynamic_action_provider is None
        assert await ai.registry.resolve_action_by_key('/dynamic-action-provider/hosted:tool/alpha_echo') is None
        assert await expand_wildcard_tools(ai.registry, ['hosted:tool/*']) == ['hosted:tool/*']
        assert server_processes() == []
    finally:
        await host.close()
