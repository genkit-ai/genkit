# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for several MCP servers behind one dynamic action provider."""

import json
import os
import sys
from pathlib import Path
from typing import Any

import psutil
import pytest
from genkit_mcp import McpClient, McpHost, define_mcp_host
from pydantic import ValidationError

from genkit import Genkit, GenkitError, Message, ModelResponse, Part
from genkit._ai._generate import expand_wildcard_tools
from genkit._core._action import Action, ActionKind
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


def server(tmp_path: Path, pid_name: str = 'server', **env: str) -> dict[str, Any]:
    """One mcpServers entry, or connect() keyword arguments, for the test server."""
    return {
        'command': sys.executable,
        'args': [str(Path(__file__).with_name('fake_server.py'))],
        'env': {
            **os.environ,
            'MCP_FAKE_PID_FILE': str(tmp_path / f'{pid_name}.pid'),
            **env,
        },
    }


def servers(tmp_path: Path, *names: str, **env: str) -> dict[str, dict[str, Any]]:
    """An mcpServers mapping holding one test server per name, each writing its own pid file."""
    return {name: server(tmp_path, pid_name=name, **env) for name in names}


def missing_server(tmp_path: Path) -> dict[str, Any]:
    """A server whose command does not exist, so it can never start."""
    return {'command': str(tmp_path / 'no-such-server')}


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
            ValueError, match="servers 'alpha_beta' and 'alpha' together: both serve a tool named 'alpha_beta_echo'"
        ):
            await host.get_active_tools()
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
        await host.connect('beta', **server(tmp_path, pid_name='beta'))
        after = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert list(host._clients) == ['alpha', 'beta']
    finally:
        await host.close()

    assert sorted(before) == sorted(f'/tool.v2/{name}' for name in hosted('alpha'))
    assert sorted(after) == sorted(f'/tool.v2/{name}' for name in hosted('alpha', 'beta'))


@pytest.mark.asyncio
async def test_connect_replaces_a_server_of_the_same_name(tmp_path: Path) -> None:
    """Reconfiguring a name stops the server it replaces, and the new one starts lazily."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha'))
    try:
        await host.get_active_tools()
        assert len(server_processes()) == 1

        await host.connect('alpha', **server(tmp_path, pid_name='replacement'))
        assert server_processes() == []

        tools = await host.get_active_tools()
        assert len(server_processes()) == 1
        assert server_process(tmp_path, 'replacement').is_running()
        assert list(host._clients) == ['alpha']
    finally:
        await host.close()

    assert tool_names(tools) == hosted('alpha')


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
                await host.connect('alpha', **server(tmp_path, pid_name='replacement'))
        assert list(host._clients) == ['alpha']
        assert server_processes() == []

        expanded = await expand_wildcard_tools(ai._registry, ['hosted:tool/*'])
        assert server_process(tmp_path, 'replacement').is_running()
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
async def test_close_releases_every_server_and_leaves_the_host_usable(tmp_path: Path) -> None:
    """Closing a host empties it rather than ending it: it can hold servers again."""
    host = McpHost('hosted', mcp_servers=servers(tmp_path, 'alpha', 'beta'))
    try:
        assert tool_names(await host.get_active_tools()) == hosted('alpha', 'beta')
        assert len(server_processes()) == 2

        await host.close()

        assert list(host._clients) == []
        assert await host.get_active_tools() == []
        assert server_processes() == []

        await host.connect('gamma', **server(tmp_path, pid_name='gamma'))
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
        await host.connect('beta', **server(tmp_path, pid_name='beta'))
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
async def test_a_server_name_no_model_would_accept_is_refused(tmp_path: Path) -> None:
    """A server name becomes a tool prefix, which a model provider has to accept."""
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        McpHost('hosted', mcp_servers={'my server': server(tmp_path)})
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
async def test_an_mcp_servers_block_passes_through_unchanged(tmp_path: Path) -> None:
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
    with pytest.raises(ValidationError) as raised:
        McpHost('hosted', mcp_servers={'alpha': loaded})
    assert 'enviroment' in str(raised.value)
    assert secret not in str(raised.value)
    assert server_processes() == []
