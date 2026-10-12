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
from genkit_mcp import McpClient, define_mcp_client

from genkit import Genkit, GenkitError, Message, ModelResponse, Part
from genkit._ai._generate import expand_wildcard_tools
from genkit._core._action import ActionKind
from genkit._core._typing import (
    FinishReason,
    Role,
    ToolRequest,
)
from genkit.testing import define_scripted_model

_T = TypeVar('_T')

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
# assert is one the client asked for.
LONG_TTL_MILLIS = 600_000


def server(tmp_path: Path, **env: str) -> dict[str, Any]:
    """Keyword arguments that launch the test server with the suite's interpreter."""
    return {
        'command': sys.executable,
        'args': [str(Path(__file__).with_name('fake_server.py'))],
        'env': {
            **os.environ,
            'MCP_FAKE_PID_FILE': str(tmp_path / 'server.pid'),
            **env,
        },
    }


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
async def test_wildcard_selector_expands_to_every_tool(tmp_path: Path) -> None:
    """``<provider>:tool/*`` binds every server tool, which needs metadata['name']."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        expanded = await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert client.name == 'fake'
    assert sorted(expanded) == sorted(f'/tool.v2/fake_{tool}' for tool in ALL_TOOLS)


@pytest.mark.asyncio
async def test_resolve_action_by_key_runs_the_tool(tmp_path: Path) -> None:
    """A DAP-qualified key resolves to the tool, and running it returns the envelope."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
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
    pm, _ = define_scripted_model(ai)
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    pm.responses = [
        tool_call_response('fake_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        response = await ai.generate(
            model='scriptedModel',
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
    assert tool_message.content[0].tool_response is not None
    assert tool_message.content[0].tool_response.output == 'hello'


@pytest.mark.asyncio
async def test_tool_names_carry_the_provider_name(tmp_path: Path) -> None:
    """Tool names carry the provider name, which the caller chose."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        expanded = await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert sorted(expanded) == sorted(f'/tool.v2/fake_{tool}' for tool in ALL_TOOLS)


@pytest.mark.asyncio
async def test_a_dead_server_drops_the_listing_and_is_replaced(tmp_path: Path) -> None:
    """A failed call invalidates the listing, and the next one gets a new process."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
    try:
        action = await ai._registry.resolve_action(ActionKind.TOOL, 'fake_echo')
        assert action is None
        await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
        action = await ai._registry.resolve_action(ActionKind.TOOL, 'fake_echo')
        assert action is not None
        assert (await action.run({'message': 'first'})).response.output == 'first'

        killed = server_processes()
        assert len(killed) == 1
        killed[0].kill()
        killed[0].wait(timeout=30)

        with pytest.raises(GenkitError) as raised:
            await action.run({'message': 'second'})
        assert raised.value.status == 'UNAVAILABLE'

        # The listing outlives its TTL, so re-listing here is the invalidation,
        # and a process to list from is the replaced connection.
        expanded = await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
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
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    background = BackgroundLoop()
    try:
        action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
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
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
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
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
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
    assert errors[0].status == 'FAILED_PRECONDITION'
    assert server_processes() == []


@pytest.mark.asyncio
async def test_a_closed_client_refuses_to_restart(tmp_path: Path) -> None:
    """Closing is the end of this client, not a pause."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    await client.get_active_tools()
    await client.close()

    with pytest.raises(GenkitError, match="MCP client 'fake' is closed") as raised:
        await client.restart()
    assert raised.value.status == 'FAILED_PRECONDITION'
    assert server_processes() == []


@pytest.mark.asyncio
async def test_restart_replaces_the_server_and_its_listing(tmp_path: Path) -> None:
    """Restart connects and lists on the calling loop before it returns."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
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
    client = define_mcp_client(ai, 'fake', command=str(tmp_path / 'no-such-server'))
    try:
        with pytest.raises(GenkitError, match='failed') as raised:
            await client.restart()
        assert raised.value.status == 'UNAVAILABLE'
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_names_are_known_before_anything_connects(tmp_path: Path) -> None:
    """Naming a tool costs nothing: the prefix is the caller's own."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        assert client.tool_name('echo') == 'fake_echo'
        assert not (tmp_path / 'server.pid').exists()
        assert server_processes() == []
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['my server', 'tools.everything', '9lives', '-fake'])
async def test_a_name_no_model_would_accept_is_refused_at_definition(tmp_path: Path, name: str) -> None:
    """The name prefixes every tool name, so it has to be one a model accepts."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        define_mcp_client(ai, name, **server(tmp_path))
    assert server_processes() == []


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['mcp-servers/everything', 'fake:everything', ''])
async def test_a_name_no_selector_could_carry_is_refused_at_definition(
    tmp_path: Path,
    name: str,
) -> None:
    """The name is the selector, so it has to parse as one."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot name an MCP provider'):
        define_mcp_client(ai, name, **server(tmp_path))
    assert server_processes() == []


@pytest.mark.asyncio
async def test_get_active_tools_lists_what_the_selector_binds(tmp_path: Path) -> None:
    """The listing is the one ``<name>:tool/*`` binds, namespaced names and all."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        tools = await client.get_active_tools()
        expanded = await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
    finally:
        await client.close()

    assert [tool.name for tool in tools] == [f'fake_{tool}' for tool in ALL_TOOLS]
    assert [tool.description for tool in tools][:2] == ['Echo input.', 'Another tool.']
    assert sorted(expanded) == sorted(f'/tool.v2/{tool.name}' for tool in tools)


@pytest.mark.asyncio
async def test_tool_name_builds_a_selector_generate_accepts(tmp_path: Path) -> None:
    """A selector built from the helpers reaches exactly that one tool."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    pm.responses = [
        tool_call_response('fake_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        selector = f'{client.name}:tool/{client.tool_name("echo")}'
        response = await ai.generate(model='scriptedModel', prompt='echo hello', tools=[selector])
    finally:
        await client.close()

    assert selector == 'fake:tool/fake_echo'
    assert response.text == 'done'
    assert pm.last_request is not None
    assert [tool.name for tool in pm.last_request.tools or []] == ['fake_echo']


@pytest.mark.asyncio
async def test_an_unregistered_client_registers_nothing(tmp_path: Path) -> None:
    """A client built directly puts nothing in a registry, so no selector reaches it."""
    ai = Genkit()
    client = McpClient('fake', **server(tmp_path))
    try:
        assert client._provider is None
        assert await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo') is None
        assert await expand_wildcard_tools(ai._registry, ['fake:tool/*']) == ['fake:tool/*']
        assert server_processes() == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unregistered_client_lists_tools_a_host_could_serve(tmp_path: Path) -> None:
    """An unregistered client still lists and runs its tools, which a host needs."""
    client = McpClient('fake', **server(tmp_path))
    try:
        tools = await client.get_active_tools()
        assert [tool.name for tool in tools] == [f'fake_{tool}' for tool in ALL_TOOLS]
        assert (await tools[0].run({'message': 'hello'})).response.output == 'hello'
    finally:
        await client.close()

    assert server_processes() == []


@pytest.mark.asyncio
async def test_an_unregistered_client_restarts_and_closes_without_a_provider(tmp_path: Path) -> None:
    """Restart and close do not need a provider, and close is still final."""
    client = McpClient('fake', **server(tmp_path))
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

    assert client._provider is None
    assert server_processes() == []
    with pytest.raises(GenkitError, match="MCP client 'fake' is closed") as raised:
        await client.restart()
    assert raised.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_get_active_tools_asks_the_server_every_call(tmp_path: Path) -> None:
    """A registered client lists afresh, while its provider keeps serving one listing."""
    ai = Genkit()
    client = define_mcp_client(
        ai,
        'fake',
        **server(tmp_path, MCP_FAKE_PAGE_SIZE=str(len(ALL_TOOLS))),
        cache_ttl_millis=LONG_TTL_MILLIS,
    )
    try:
        await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
        await client.get_active_tools()
        counter = next(tool for tool in await client.get_active_tools() if tool.name == 'fake_list_requests')
        listings = (await counter.run({})).response.output

        await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
        cached = (await counter.run({})).response.output
    finally:
        await client.close()

    assert listings == '3'
    assert cached == '3'


@pytest.mark.asyncio
async def test_tools_sharing_a_genkit_name_fail_the_listing(tmp_path: Path) -> None:
    """The provider keeps one action per name, so a collision would hide a tool."""
    ai = Genkit()
    client = define_mcp_client(ai, 'bookshop', **server(tmp_path, MCP_FAKE_COLLIDING_TOOLS='1'))
    collision = r"MCP tools 'look\.up' and 'look/up' both map to the Genkit tool name 'bookshop_look_up'"
    try:
        with pytest.raises(ValueError, match=collision):
            await client.get_active_tools()
        with pytest.raises(ValueError, match=collision):
            await expand_wildcard_tools(ai._registry, ['bookshop:tool/*'])
    finally:
        await client.close()
