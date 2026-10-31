# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the MCP client registered as a dynamic action provider."""

import asyncio
import gc
import os
import sys
import threading
import time
import weakref
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import genkit_mcp._client
import psutil
import pytest
from genkit_mcp import McpClient, define_mcp_client
from genkit_mcp._config import McpStdioServerConfig
from genkit_mcp._connection import McpConnection, _serving

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


def server_processes(tmp_path: Path) -> list[psutil.Process]:
    """Every fake MCP server this test started that is still running."""
    pid_file = str(tmp_path / 'server.pid')
    running: list[psutil.Process] = []
    for child in psutil.Process().children(recursive=True):
        try:
            if any('fake_server.py' in argument for argument in child.cmdline()) and (
                child.environ().get('MCP_FAKE_PID_FILE') == pid_file
            ):
                running.append(child)
        except psutil.NoSuchProcess:  # pragma: no cover - the child exited mid-scan
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

    def pause(self) -> None:
        """Stop the loop without closing it, as a loop driven one run at a time sits between runs."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=30)

    def resume(self) -> None:
        """Run a paused loop again on a new thread. Does nothing while it runs."""
        if self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

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
async def test_generate_accepts_the_listed_tools(tmp_path: Path) -> None:
    """The listing is made of tool handles, so it goes into ``generate`` as it is, provider or not."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    client = McpClient('fake', **server(tmp_path))
    pm.responses = [
        tool_call_response('fake_echo', {'message': 'hello'}),
        text_response('done'),
    ]

    try:
        tools = await client.get_active_tools()
        response = await ai.generate(model='scriptedModel', prompt='echo hello', tools=tools)
    finally:
        await client.close()

    assert response.text == 'done'
    assert pm.last_request is not None
    assert sorted(tool.name for tool in pm.last_request.tools or []) == sorted(f'fake_{tool}' for tool in ALL_TOOLS)
    tool_message = pm.last_request.messages[-1]
    assert tool_message.content[0].tool_response is not None
    assert tool_message.content[0].tool_response.output == 'hello'
    sent = next(tool for tool in pm.last_request.tools or [] if tool.name == 'fake_bad_output')
    assert sent.output_schema is None
    assert tools[0].output_schema is None
    assert tools[-1].name == 'fake_bad_output'
    assert tools[-1].output_schema == {
        'type': 'object',
        'properties': {'value': {'type': 'integer'}},
        'required': ['value'],
    }


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

        killed = server_processes(tmp_path)
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
        replaced = server_processes(tmp_path)
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
        assert len(server_processes(tmp_path)) == 2

        await client.close()
        assert server_processes(tmp_path) == []
    finally:
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_close_ignores_a_loop_which_already_finished(tmp_path: Path) -> None:
    """A finished loop cancelled its connection on the way out, taking the server with it."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
        assert action is not None
        outputs: list[object] = []

        def use_own_loop() -> None:
            outputs.append(asyncio.run(action.run({'message': 'other'})).response.output)

        thread = threading.Thread(target=use_own_loop)
        thread.start()
        thread.join(timeout=30)

        assert not thread.is_alive()
        assert outputs == ['other']
        # Only the main loop's connection is left: it was opened by the listing.
        assert len(server_processes(tmp_path)) == 1

        await client.close()
        assert server_processes(tmp_path) == []
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.filterwarnings('error::pytest.PytestUnraisableExceptionWarning')
async def test_a_loop_which_stops_before_closing_its_connection_closes_it_when_run_again(tmp_path: Path) -> None:
    """Close does not wait on a loop which stopped, and that loop stops its server once it runs again."""
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    release = threading.Event()
    try:
        tools = await client.get_active_tools()
        assert (await background.run(tools[0]({'message': 'background'}))).output == 'background'
        background.loop.call_soon_threadsafe(release.wait)
        background.loop.call_soon_threadsafe(background.loop.stop)
        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        release.set()

        await asyncio.wait_for(closing, 10)
        stranded = server_processes(tmp_path)
        assert len(stranded) == 1

        background.resume()
        _, alive = await asyncio.to_thread(psutil.wait_procs, stranded, 15)
        assert alive == []
    finally:
        release.set()
        background.resume()
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_a_cancelled_close_still_stops_every_server(tmp_path: Path) -> None:
    """Each loop's shutdown runs on its own, so a caller which gives up strands no server."""
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    try:
        tools = await client.get_active_tools()
        assert (await background.run(tools[0]({'message': 'background'}))).output == 'background'
        running = server_processes(tmp_path)
        assert len(running) == 2

        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        _, alive = await asyncio.to_thread(psutil.wait_procs, running, 15)
        assert alive == []
    finally:
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_a_second_close_waits_for_the_first(tmp_path: Path) -> None:
    """Close returns once the server has exited, even when another close started the shutdown."""
    client = McpClient('fake', **server(tmp_path))
    await client.get_active_tools()

    first = asyncio.create_task(client.close())
    try:
        await asyncio.sleep(0)
        assert not first.done()
        await client.close()

        assert server_processes(tmp_path) == []
    finally:
        await first


@pytest.mark.asyncio
async def test_close_returns_when_another_loop_cancels_its_shutdown(tmp_path: Path) -> None:
    """asyncio.run cancels every task on its way out, the shutdown among them, and that is no failure of close."""
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    release = threading.Event()

    def cancel_everything() -> None:
        for task in asyncio.all_tasks(background.loop):
            task.cancel()

    try:
        tools = await client.get_active_tools()
        assert (await background.run(tools[0]({'message': 'background'}))).output == 'background'
        background.loop.call_soon_threadsafe(release.wait)
        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)
        background.loop.call_soon_threadsafe(cancel_everything)
        release.set()

        await asyncio.wait_for(closing, 10)
        _, alive = await asyncio.to_thread(psutil.wait_procs, server_processes(tmp_path), 15)
        assert alive == []
    finally:
        release.set()
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_restart_stops_a_paused_loops_server_when_that_loop_runs_again(tmp_path: Path) -> None:
    """A loop driven one run at a time is not running when another loop restarts, and its old server is not leaked."""
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    try:
        tools = await client.get_active_tools()
        assert (await background.run(tools[0]({'message': 'background'}))).output == 'background'
        replaced = server_processes(tmp_path)
        assert len(replaced) == 2
        background.pause()

        await client.restart()
        background.resume()
        assert (await background.run(tools[0]({'message': 'again'}))).output == 'again'

        _, alive = await asyncio.to_thread(psutil.wait_procs, replaced, 15)
        assert alive == []
        assert len(server_processes(tmp_path)) == 2
    finally:
        background.resume()
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_a_close_refused_mid_request_leaves_the_server_for_the_next_close(tmp_path: Path) -> None:
    """A connection cannot be closed from inside its own request, and stays where a later close reaches it."""
    client = McpClient('fake', **server(tmp_path))
    await client.get_active_tools()
    running = server_processes(tmp_path)
    assert len(running) == 1
    (serving,) = client._connections.values()

    token = _serving.set(serving)
    try:
        with pytest.raises(GenkitError, match='cannot be closed from inside') as raised:
            await client.close()
        assert raised.value.status == 'FAILED_PRECONDITION'
    finally:
        _serving.reset(token)
    assert server_processes(tmp_path) == running

    await client.close()
    assert server_processes(tmp_path) == []


@pytest.mark.asyncio
async def test_a_finished_loops_connection_is_dropped(tmp_path: Path) -> None:
    """A connection holds its loop, so the client lets go of those whose loop has closed."""
    client = McpClient('fake', **server(tmp_path))
    finished: list[weakref.ref[asyncio.AbstractEventLoop]] = []
    try:
        tools = await client.get_active_tools()

        async def call_and_note_the_loop() -> None:
            finished.append(weakref.ref(asyncio.get_running_loop()))
            await tools[0]({'message': 'other'})

        def use_own_loop() -> None:
            asyncio.run(call_and_note_the_loop())

        for _ in range(3):
            thread = threading.Thread(target=use_own_loop)
            thread.start()
            thread.join(timeout=30)
            assert not thread.is_alive()

        assert (await tools[0]({'message': 'main'})).output == 'main'
        gc.collect()
        assert len(finished) == 3
        assert [loop for loop in finished if loop() is not None] == []
    finally:
        await client.close()


def test_a_discarded_genkit_instance_takes_its_client_with_it(tmp_path: Path) -> None:
    """The record of which client registered a provider does not keep either alive."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    assert client._provider is not None
    collected_client = weakref.ref(client)
    collected_provider = weakref.ref(client._provider.action)

    del ai, client
    gc.collect()

    assert collected_client() is None
    assert collected_provider() is None


@pytest.mark.asyncio
async def test_close_stops_waiting_on_a_loop_too_busy_to_close_its_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked loop holds close up for the timeout only, and closes the connection once it is free."""
    monkeypatch.setattr(genkit_mcp._client, '_CROSS_LOOP_CLOSE_TIMEOUT_SECONDS', 0.5)
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    try:
        tools = await client.get_active_tools()
        assert (await background.run(tools[0]({'message': 'background'}))).output == 'background'
        background.loop.call_soon_threadsafe(time.sleep, 3)

        started = time.monotonic()
        await asyncio.wait_for(client.close(), 10)
        assert time.monotonic() - started < 2.5

        _, alive = psutil.wait_procs(server_processes(tmp_path), timeout=10)
        assert alive == []
    finally:
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_a_closed_client_starts_no_server_on_a_new_loop(tmp_path: Path) -> None:
    """Close is final: a loop which never had a connection does not get one."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
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

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], GenkitError)
        assert errors[0].status == 'FAILED_PRECONDITION'
        assert server_processes(tmp_path) == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_closed_clients_tools_fail_as_closed_except_by_key(tmp_path: Path) -> None:
    """Generate and listing report the closed client, while key resolution finds nothing."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    client = define_mcp_client(ai, 'fake', **server(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
    await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
    await client.close()

    with pytest.raises(GenkitError) as raised:
        await ai.generate(model='scriptedModel', prompt='echo hello', tools=['fake:tool/*'])
    assert raised.value.status == 'FAILED_PRECONDITION'
    with pytest.raises(GenkitError) as raised:
        await client.get_active_tools()
    assert raised.value.status == 'FAILED_PRECONDITION'
    assert await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo') is None
    assert pm.request_count == 0


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
    assert server_processes(tmp_path) == []


@pytest.mark.asyncio
async def test_restart_replaces_the_server_and_its_listing(tmp_path: Path) -> None:
    """Restart connects and lists on the calling loop before it returns."""
    ai = Genkit()
    client = define_mcp_client(ai, 'fake', **server(tmp_path), cache_ttl_millis=LONG_TTL_MILLIS)
    key = '/dynamic-action-provider/fake:tool/fake_echo'
    try:
        listed = [tool.name for tool in await client.get_active_tools()]
        cached = await ai._registry.resolve_action_by_key(key)
        assert cached is not None
        before = server_processes(tmp_path)
        assert len(before) == 1

        await client.restart()

        after = server_processes(tmp_path)
        assert len(after) == 1
        assert after[0].pid != before[0].pid
        assert [tool.name for tool in await client.get_active_tools()] == listed
        relisted = await ai._registry.resolve_action_by_key(key)
        assert relisted is not None
        assert relisted is not cached
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_call_cut_off_by_restart_fails_as_unavailable(tmp_path: Path) -> None:
    """The client is still live after a restart, so the cut-off call is one to retry, not a closed client."""
    client = McpClient('fake', **server(tmp_path))
    try:
        tools = {tool.name: tool for tool in await client.get_active_tools()}
        stalled = asyncio.create_task(tools['fake_stall']({}))
        await tools['fake_await_stall']({})

        await client.restart()

        with pytest.raises(GenkitError, match="MCP client 'fake' restarted") as raised:
            await asyncio.wait_for(stalled, 15)
        assert raised.value.status == 'UNAVAILABLE'
        assert (await tools['fake_echo']({'message': 'retried'})).output == 'retried'
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_a_call_on_another_loop_cut_off_by_restart_fails_as_unavailable(tmp_path: Path) -> None:
    """A restart on one loop reaches a call in flight on another, such as the Dev UI's."""
    client = McpClient('fake', **server(tmp_path))
    background = BackgroundLoop()
    try:
        tools = {tool.name: tool for tool in await client.get_active_tools()}
        stalled = asyncio.wrap_future(asyncio.run_coroutine_threadsafe(tools['fake_stall']({}), background.loop))
        await background.run(tools['fake_await_stall']({}))

        await client.restart()

        with pytest.raises(GenkitError, match="MCP client 'fake' restarted") as raised:
            await asyncio.wait_for(stalled, 15)
        assert raised.value.status == 'UNAVAILABLE'
        assert (await background.run(tools['fake_echo']({'message': 'retried'}))).output == 'retried'
    finally:
        await client.close()
        background.stop()


@pytest.mark.asyncio
async def test_a_call_cut_off_by_close_fails_as_closed(tmp_path: Path) -> None:
    """Close is final, so a call it cuts off reports the closed client."""
    client = McpClient('fake', **server(tmp_path))
    try:
        tools = {tool.name: tool for tool in await client.get_active_tools()}
        stalled = asyncio.create_task(tools['fake_stall']({}))
        await tools['fake_await_stall']({})
    finally:
        await client.close()

    with pytest.raises(GenkitError) as raised:
        await stalled
    assert raised.value.status == 'FAILED_PRECONDITION'


@pytest.mark.asyncio
async def test_restart_reports_a_server_which_cannot_be_started(tmp_path: Path) -> None:
    """The caller of restart hears about a broken server, not the next generate."""
    ai = Genkit()
    client = define_mcp_client(ai, 'bookshop', command=str(tmp_path / 'no-such-server'))
    try:
        with pytest.raises(GenkitError, match="MCP connection to 'bookshop' .* failed") as raised:
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
        assert server_processes(tmp_path) == []
    finally:
        await client.close()


@pytest.mark.parametrize('name', ['my server', 'tools.everything', '9lives', '-fake'])
def test_a_name_no_model_would_accept_is_refused_at_definition(tmp_path: Path, name: str) -> None:
    """The name prefixes every tool name, so it has to be one a model accepts."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot prefix an MCP tool name'):
        define_mcp_client(ai, name, **server(tmp_path))
    assert server_processes(tmp_path) == []


@pytest.mark.parametrize('name', ['mcp-servers/everything', 'fake:everything', ''])
def test_a_name_no_selector_could_carry_is_refused_at_definition(
    tmp_path: Path,
    name: str,
) -> None:
    """The name is the selector, so it has to parse as one."""
    ai = Genkit()
    with pytest.raises(ValueError, match='cannot name an MCP provider'):
        define_mcp_client(ai, name, **server(tmp_path))
    assert server_processes(tmp_path) == []


@pytest.mark.asyncio
async def test_a_name_a_live_client_holds_is_refused_at_definition(tmp_path: Path) -> None:
    """A second provider under the same name would silently orphan the first client."""
    ai = Genkit()
    first = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        with pytest.raises(ValueError, match="'fake' is already registered"):
            define_mcp_client(ai, 'fake', **server(tmp_path))
        registered = ai._registry.registered_action(ActionKind.DYNAMIC_ACTION_PROVIDER, 'fake')
        assert first._provider is not None
        assert registered is first._provider.action
    finally:
        await first.close()


def test_a_name_another_provider_holds_is_refused_at_definition(tmp_path: Path) -> None:
    """The name is taken whoever registered the provider."""
    ai = Genkit()

    async def list_nothing() -> dict[str, list[Any]]:
        return {}

    ai.define_dynamic_action_provider('fake', list_nothing)
    with pytest.raises(ValueError, match="'fake' is already registered"):
        define_mcp_client(ai, 'fake', **server(tmp_path))


@pytest.mark.asyncio
async def test_a_closed_clients_name_can_be_defined_again(tmp_path: Path) -> None:
    """A new client is how a closed one's server is used again, under the same selectors."""
    ai = Genkit()
    first = define_mcp_client(ai, 'fake', **server(tmp_path))
    await first.close()
    second = define_mcp_client(ai, 'fake', **server(tmp_path))
    try:
        action = await ai._registry.resolve_action_by_key('/dynamic-action-provider/fake:tool/fake_echo')
        assert action is not None
        assert (await action.run({'message': 'again'})).response.output == 'again'
    finally:
        await second.close()


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
        assert server_processes(tmp_path) == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_unregistered_client_lists_and_runs_its_tools(tmp_path: Path) -> None:
    """An unregistered client still lists and runs its tools, with no provider involved."""
    client = McpClient('fake', **server(tmp_path))
    try:
        tools = await client.get_active_tools()
        assert [tool.name for tool in tools] == [f'fake_{tool}' for tool in ALL_TOOLS]
        assert (await tools[0]({'message': 'hello'})).output == 'hello'
    finally:
        await client.close()

    assert server_processes(tmp_path) == []


@pytest.mark.asyncio
async def test_an_unregistered_client_restarts_and_closes_without_a_provider(tmp_path: Path) -> None:
    """Restart and close do not need a provider, and close is still final."""
    client = McpClient('fake', **server(tmp_path))
    try:
        listed = [tool.name for tool in await client.get_active_tools()]
        before = server_processes(tmp_path)
        assert len(before) == 1

        await client.restart()

        after = server_processes(tmp_path)
        assert len(after) == 1
        assert after[0].pid != before[0].pid
        assert [tool.name for tool in await client.get_active_tools()] == listed
    finally:
        await client.close()

    assert client._provider is None
    assert server_processes(tmp_path) == []
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
        listings = (await counter({})).output

        await expand_wildcard_tools(ai._registry, ['fake:tool/*'])
        cached = (await counter({})).output
    finally:
        await client.close()

    assert listings == '3'
    assert cached == '3'


@pytest.mark.asyncio
async def test_tools_sharing_a_genkit_name_are_left_out(tmp_path: Path) -> None:
    """Either colliding tool could answer a call meant for the other, so neither is listed or bound."""
    ai = Genkit()
    client = define_mcp_client(ai, 'bookshop', **server(tmp_path, MCP_FAKE_COLLIDING_TOOLS='1'))
    try:
        assert await client.get_active_tools() == []
        expanded = await expand_wildcard_tools(ai._registry, ['bookshop:tool/*'])
        assert [name for name in expanded if name.startswith('/tool')] == []
        assert (
            await ai._registry.resolve_action_by_key('/dynamic-action-provider/bookshop:tool/bookshop_look_up') is None
        )
    finally:
        await client.close()


@pytest.mark.parametrize('timeout', [0, -1])
def test_define_mcp_client_rejects_a_timeout_that_is_not_positive(timeout: int) -> None:
    """A zero timeout would fail every request at once instead of meaning no timeout."""
    with pytest.raises(ValueError, match='request_timeout_millis'):
        define_mcp_client(Genkit(), 'everything', command='npx', request_timeout_millis=timeout)


@pytest.mark.parametrize('timeout', [0, -1])
def test_connection_rejects_a_timeout_that_is_not_positive(timeout: int) -> None:
    """The connection guards the timeout too, for callers that build one directly."""
    with pytest.raises(ValueError, match='request_timeout_millis'):
        McpConnection('fake', McpStdioServerConfig(command='npx'), request_timeout_millis=timeout)
