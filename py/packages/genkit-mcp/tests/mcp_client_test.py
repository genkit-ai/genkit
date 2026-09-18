# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the MCP client registered as a dynamic action provider."""

import asyncio
import os
import sys
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import psutil
import pytest
from genkit_mcp import (
    McpConnectionClosedError,
    McpConnectionFailedError,
    McpStdioServerConfig,
    create_mcp_client,
    define_mcp_client,
)

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import expand_wildcard_tools
from genkit._ai._testing import define_programmable_model
from genkit._core._action import ActionKind
from genkit._core._error import GenkitError
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

_T = TypeVar('_T')

ALL_TOOLS = ['echo', 'second', 'wait', 'list_requests', 'who_called', 'stall']

# A listing must not expire by the clock in these tests: every re-list they
# assert is one the client asked for.
LONG_TTL_MILLIS = 600_000


def config(tmp_path: Path, disabled: bool = False, **env: str) -> McpStdioServerConfig:
    """Launch the test server with the same interpreter as the test suite."""
    return McpStdioServerConfig(
        command=sys.executable,
        args=[str(Path(__file__).with_name('fake_server.py'))],
        env={
            **os.environ,
            'MCP_FAKE_PID_FILE': str(tmp_path / 'server.pid'),
            **env,
        },
        disabled=disabled,
    )


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


class BackgroundLoop:
    """A second event loop on its own thread, as the Dev UI reflection server runs."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    async def run(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Await a coroutine which runs on the background loop."""
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self.loop))

    def stop(self) -> None:
        """Stop the loop and join its thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=30)
        self.loop.close()


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
async def test_wildcard_selector_expands_to_every_tool(tmp_path: Path) -> None:
    """``<provider>:tool/*`` binds every server tool, which needs metadata['name']."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    try:
        expanded = await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert client.name == 'fake'
    assert sorted(expanded) == sorted(f'/tool.v2/fake_{tool}' for tool in ALL_TOOLS)


@pytest.mark.asyncio
async def test_resolve_action_by_key_runs_the_tool(tmp_path: Path) -> None:
    """A DAP-qualified key resolves to the tool, and running it returns the envelope."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    try:
        action = await ai.registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
        assert action is not None
        response = (await action.run({'message': 'hello'})).response
    finally:
        await client.close()

    assert response.output == 'hello'
    assert response.content is None


@pytest.mark.asyncio
async def test_generate_calls_a_tool_and_receives_its_result(tmp_path: Path) -> None:
    """A model can request an MCP tool through a selector and read what it returned."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    pm.responses = [
        tool_call_response('fake_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        response = await ai.generate(
            model='programmableModel',
            prompt='echo hello',
            tools=['fake:tool/*'],
        )
    finally:
        await client.close()

    assert response.text == 'done'
    assert pm.request_count == 2
    assert pm.last_request is not None
    assert sorted(tool.name for tool in pm.last_request.tools or []) == sorted(f'fake_{tool}' for tool in ALL_TOOLS)
    tool_message = pm.last_request.messages[-1]
    assert tool_message.role == Role.TOOL
    assert tool_message.content[0].root.tool_response is not None
    assert tool_message.content[0].root.tool_response.output == 'hello'


@pytest.mark.asyncio
async def test_disabled_server_has_no_tools_and_starts_no_process(tmp_path: Path) -> None:
    """``disabled`` advertises nothing and launches nothing."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path, disabled=True))
    try:
        expanded = await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert expanded == ['fake:tool/*']
    assert not (tmp_path / 'server.pid').exists()
    assert server_processes() == []


@pytest.mark.asyncio
@pytest.mark.parametrize(('override', 'prefix'), [(None, 'fake'), ('custom', 'custom')])
async def test_tool_names_carry_the_provider_name(
    tmp_path: Path,
    override: str | None,
    prefix: str,
) -> None:
    """Tool names carry the provider name, which the caller chose, unless overridden."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path), tool_prefix=override)
    try:
        expanded = await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert sorted(expanded) == sorted(f'/tool.v2/{prefix}_{tool}' for tool in ALL_TOOLS)


@pytest.mark.asyncio
async def test_a_dead_server_drops_the_listing_and_is_replaced(tmp_path: Path) -> None:
    """A failed call invalidates the listing, and the next one gets a new process."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        action = await ai.registry.resolve_action(ActionKind.TOOL, 'fake_echo')
        assert action is None
        await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
        action = await ai.registry.resolve_action(ActionKind.TOOL, 'fake_echo')
        assert action is not None
        assert (await action.run({'message': 'first'})).response.output == 'first'

        killed = server_processes()
        assert len(killed) == 1
        killed[0].kill()
        killed[0].wait(timeout=30)

        with pytest.raises(GenkitError) as raised:
            await action.run({'message': 'second'})
        assert isinstance(raised.value.cause, McpConnectionFailedError)

        # The listing outlives its TTL, so re-listing here is the invalidation,
        # and a process to list from is the replaced connection.
        expanded = await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
        assert sorted(expanded) == sorted(f'/tool.v2/fake_{tool}' for tool in ALL_TOOLS)
        replaced = server_processes()
        assert len(replaced) == 1
        assert replaced[0].pid != killed[0].pid
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_close_stops_the_server_on_every_loop_that_opened_one(tmp_path: Path) -> None:
    """Each loop gets its own server process, and close stops all of them."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    background = BackgroundLoop()
    try:
        action = await ai.registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
        assert action is not None
        assert (await action.run({'message': 'main'})).response.output == 'main'
        assert (await background.run(action.run({'message': 'background'}))).response.output == 'background'
        assert len(server_processes()) == 2

        await client.close()
        assert server_processes() == []
        await client.close()
    finally:
        background.stop()


@pytest.mark.asyncio
async def test_close_ignores_a_loop_which_already_finished(tmp_path: Path) -> None:
    """A finished loop cancelled its connection on the way out, taking the server with it."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    action = await ai.registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
    assert action is not None
    outputs: list[object] = []

    def use_own_loop() -> None:
        outputs.append(asyncio.run(action.run({'message': 'other'})).response.output)

    thread = threading.Thread(target=use_own_loop)
    thread.start()
    thread.join(timeout=30)

    assert outputs == ['other']
    # Only the main loop's connection is left: it was opened by the listing.
    assert len(server_processes()) == 1

    await client.close()
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_closed_client_starts_no_server_on_a_new_loop(tmp_path: Path) -> None:
    """Close is final: a loop which never had a connection does not get one."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    action = await ai.registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
    assert action is not None
    await client.close()

    errors: list[BaseException] = []

    def use_own_loop() -> None:
        try:
            asyncio.run(action.run({'message': 'late'}))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=use_own_loop)
    thread.start()
    thread.join(timeout=30)

    assert isinstance(errors[0], GenkitError)
    assert isinstance(errors[0].cause, McpConnectionClosedError)
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_closed_client_refuses_to_restart(tmp_path: Path) -> None:
    """Closing is the end of this client, not a pause."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    await client.get_active_tools()
    await client.close()

    with pytest.raises(McpConnectionClosedError, match="MCP client 'fake' is closed"):
        await client.restart()
    assert server_processes() == []


@pytest.mark.asyncio
async def test_restart_replaces_the_server_and_its_listing(tmp_path: Path) -> None:
    """Restart connects and lists on the calling loop before it returns."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        listed = [tool.name for tool in await client.get_active_tools()]
        before = server_processes()
        assert len(before) == 1

        await client.restart()

        after = server_processes()
        assert len(after) == 1
        assert after[0].pid != before[0].pid
        assert [tool.name for tool in await client.get_active_tools()] == listed
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_restart_reports_a_server_which_cannot_be_started(tmp_path: Path) -> None:
    """The caller of restart hears about a broken server, not the next generate."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', McpStdioServerConfig(command=str(tmp_path / 'no-such-server')))
    try:
        with pytest.raises(McpConnectionFailedError, match='failed'):
            await client.restart()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_names_are_known_before_anything_connects(tmp_path: Path) -> None:
    """Naming a tool costs nothing: the prefix is the caller's own."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    try:
        assert client.tool_prefix == 'fake'
        assert client.tool_name('echo') == 'fake_echo'
        assert not (tmp_path / 'server.pid').exists()
        assert server_processes() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_supplied_prefix_replaces_the_provider_name(tmp_path: Path) -> None:
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path), tool_prefix='bookshop')
    try:
        assert client.tool_prefix == 'bookshop'
        assert client.tool_name('echo') == 'bookshop_echo'
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('name', 'prefix'),
    [('my server', None), ('tools.everything', None), ('fake', 'mcp servers')],
)
async def test_a_prefix_no_model_would_accept_is_refused_at_definition(
    tmp_path: Path,
    name: str,
    prefix: str | None,
) -> None:
    """The prefix a tool name is built from has to be one a model accepts."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        define_mcp_client(ai, name, config(tmp_path), tool_prefix=prefix)
    assert server_processes() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['mcp-servers/everything', 'fake:everything', ''])
async def test_a_name_no_selector_could_carry_is_refused_at_definition(
    tmp_path: Path,
    name: str,
) -> None:
    """A tool_prefix cannot rescue a name, because the name is the selector."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot name an MCP provider'):
        define_mcp_client(ai, name, config(tmp_path), tool_prefix='bookshop')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_prefix_rescues_a_name_a_model_would_reject(tmp_path: Path) -> None:
    """A name is only a selector, so a prefix makes an awkward one usable."""
    ai = Genkit()
    client = define_mcp_client(ai, 'my bookshop', config(tmp_path), tool_prefix='bookshop')
    try:
        assert client.name == 'my bookshop'
        assert client.tool_name('echo') == 'bookshop_echo'
        assert [tool.name for tool in await client.get_active_tools()] == [f'bookshop_{tool}' for tool in ALL_TOOLS]
        assert await expand_wildcard_tools(ai.registry, ['my bookshop:tool/*']) == [
            f'/tool.v2/bookshop_{tool}' for tool in ALL_TOOLS
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_empty_prefix_is_not_silently_used(tmp_path: Path) -> None:
    """An empty prefix would name tools ``_echo``, so it is refused like any other."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        define_mcp_client(ai, 'fake', config(tmp_path), tool_prefix='')
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_disabled_server_names_tools_it_never_serves(tmp_path: Path) -> None:
    """A disabled server offers nothing, and naming still needs no connection."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path, disabled=True))
    try:
        assert client.tool_name('echo') == 'fake_echo'
        assert await client.get_active_tools() == []
        assert not (tmp_path / 'server.pid').exists()
        assert server_processes() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_get_active_tools_lists_what_the_selector_binds(tmp_path: Path) -> None:
    """The listing is the one ``<name>:tool/*`` binds, namespaced names and all."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    try:
        tools = await client.get_active_tools()
        expanded = await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert [tool.name for tool in tools] == [f'fake_{tool}' for tool in ALL_TOOLS]
    assert [tool.description for tool in tools][:2] == ['Echo input.', 'Another tool.']
    assert sorted(expanded) == sorted(f'/tool.v2/{tool.name}' for tool in tools)


@pytest.mark.asyncio
async def test_tool_name_builds_a_selector_generate_accepts(tmp_path: Path) -> None:
    """A selector built from the helpers reaches exactly that one tool."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    client = define_mcp_client(ai, 'fake', config(tmp_path))
    pm.responses = [
        tool_call_response('fake_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        selector = f'{client.name}:tool/{client.tool_name("echo")}'
        response = await ai.generate(model='programmableModel', prompt='echo hello', tools=[selector])
    finally:
        await client.close()

    assert selector == 'fake:tool/fake_echo'
    assert response.text == 'done'
    assert pm.last_request is not None
    assert [tool.name for tool in pm.last_request.tools or []] == ['fake_echo']


@pytest.mark.asyncio
async def test_a_created_client_registers_nothing(tmp_path: Path) -> None:
    """create_mcp_client puts nothing in a registry, so no selector reaches it."""
    ai = Genkit()
    client = create_mcp_client('fake', config(tmp_path))
    try:
        assert client.dynamic_action_provider is None
        assert await ai.registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo') is None
        assert await expand_wildcard_tools(ai.registry, ['fake:tool/*']) == ['fake:tool/*']
        assert server_processes() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_created_client_lists_tools_a_host_could_serve(tmp_path: Path) -> None:
    """An unregistered client still lists and runs its tools, which a host needs."""
    client = create_mcp_client('fake', config(tmp_path))
    try:
        tools = await client.get_active_tools()
        assert [tool.name for tool in tools] == [f'fake_{tool}' for tool in ALL_TOOLS]
        assert (await tools[0].run({'message': 'hello'})).response.output == 'hello'
    finally:
        await client.close()

    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_created_client_restarts_and_closes_without_a_provider(tmp_path: Path) -> None:
    """Restart and close do not need a provider, and close is still final."""
    client = create_mcp_client('fake', config(tmp_path))
    try:
        listed = [tool.name for tool in await client.get_active_tools()]
        before = server_processes()
        assert len(before) == 1

        await client.restart()

        after = server_processes()
        assert len(after) == 1
        assert after[0].pid != before[0].pid
        assert [tool.name for tool in await client.get_active_tools()] == listed
    finally:
        await client.close()

    assert client.dynamic_action_provider is None
    assert server_processes() == []
    with pytest.raises(McpConnectionClosedError, match="MCP client 'fake' is closed"):
        await client.restart()


@pytest.mark.asyncio
async def test_get_active_tools_asks_the_server_every_call(tmp_path: Path) -> None:
    """A registered client lists afresh, while its provider keeps serving one listing."""
    ai = Genkit()
    client = define_mcp_client(
        ai,
        'fake',
        config(tmp_path, MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
        await client.get_active_tools()
        counter = next(tool for tool in await client.get_active_tools() if tool.name == 'fake_list_requests')
        listings = (await counter.run({})).response.output

        await expand_wildcard_tools(ai.registry, ['fake:tool/*'])
        cached = (await counter.run({})).response.output
    finally:
        await client.close()

    assert listings == '3'
    assert cached == '3'
