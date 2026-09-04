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

"""Tests for DAP-backed tool resolution in the generate loop."""

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse
from genkit._ai._generate import expand_wildcard_tools, resolve_tool
from genkit._ai._testing import define_programmable_model
from genkit._core._action import Action, ActionKind
from genkit._core._dap import DapValue, define_dynamic_action_provider
from genkit._core._error import GenkitError
from genkit._core._registry import Registry
from genkit._core._typing import (
    FinishReason,
    Part,
    Role,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(role=Role.MODEL, content=[Part(root=TextPart(text=text))]),
        finish_reason=FinishReason.STOP,
    )


def _tool_call_response(tool_name: str, input: dict) -> ModelResponse:
    return ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name=tool_name, input=input, ref=tool_name)))],
        ),
        finish_reason=FinishReason.STOP,
    )


# ---------------------------------------------------------------------------
# expand_wildcard_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_star_includes_every_tool_from_that_provider() -> None:
    """tools=['mcp:tool/*'] includes every tool from that provider as /tool.v2/<name>."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    t1 = registry.register_action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})
    t2 = registry.register_action(name='ping', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'ping'})

    async def dap_fn() -> DapValue:
        return {'tool': [t1, t2]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    result = await expand_wildcard_tools(registry, ['mcp:tool/*'])
    assert sorted(result) == [
        '/tool.v2/echo',
        '/tool.v2/ping',
    ]
    assert set(registry._entries.get(ActionKind.TOOL, {})) == {'echo', 'ping'}


@pytest.mark.asyncio
async def test_mcp_tool_echo_becomes_tool_v2_echo() -> None:
    """tools=['mcp:tool/echo'] becomes /tool.v2/echo on the registry generate handed us."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    # The provider is a catalog row. Its tools are not — people pick them
    # with a selector, and generate binds them on the child it passes in.
    before = await registry.list_actions()
    assert '/dynamic-action-provider/mcp' in before
    assert '/dynamic-action-provider/mcp:tool/echo' not in before
    assert '/tool.v2/echo' not in before

    expanded = await expand_wildcard_tools(registry, ['mcp:tool/echo'])
    assert expanded == ['/tool.v2/echo']
    assert registry._entries.get(ActionKind.TOOL, {}).get('echo') is echo
    catalog = await registry.list_actions()
    assert catalog['/tool.v2/echo'].name == 'echo'
    assert '/dynamic-action-provider/mcp:tool/echo' not in catalog
    assert '/dynamic-action-provider/mcp' in catalog


@pytest.mark.asyncio
async def test_dap_wildcard_registers_on_passed_registry() -> None:
    """``mcp:tool/*`` binds each tool onto the registry generate handed us."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})
    ping = Action(name='ping', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'ping'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo, ping]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)
    child = registry.new_child()

    result = await expand_wildcard_tools(child, ['mcp:tool/*'])
    assert sorted(result) == [
        '/tool.v2/echo',
        '/tool.v2/ping',
    ]
    assert child._entries.get(ActionKind.TOOL, {}).get('echo') is echo
    assert child._entries.get(ActionKind.TOOL, {}).get('ping') is ping
    assert 'echo' not in registry._entries.get(ActionKind.TOOL, {})
    assert 'ping' not in registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_mcp_tool_echo_does_not_appear_on_the_app_catalog() -> None:
    """mcp:tool/echo is not a row on the app catalog. /tool.v2/echo lives on the generate child."""
    parent = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(parent, 'mcp', dap_fn)
    child = parent.new_child()

    expanded = await expand_wildcard_tools(child, ['mcp:tool/echo'])
    assert expanded == ['/tool.v2/echo']
    assert child._entries.get(ActionKind.TOOL, {}).get('echo') is echo
    assert 'echo' not in parent._entries.get(ActionKind.TOOL, {})

    parent_catalog = await parent.list_actions()
    assert '/dynamic-action-provider/mcp' in parent_catalog
    assert '/dynamic-action-provider/mcp:tool/echo' not in parent_catalog
    assert '/tool.v2/echo' not in parent_catalog

    child_catalog = await child.list_actions()
    assert child_catalog['/tool.v2/echo'].name == 'echo'


@pytest.mark.asyncio
async def test_mcp_tool_v2_echo_is_not_a_tools_argument() -> None:
    """tools=['mcp:tool.v2/echo'] is not a tools argument. Expand leaves it."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    for selector in ('mcp:tool.v2/echo', 'mcp:tool.v2/*'):
        expanded = await expand_wildcard_tools(registry, [selector])
        assert expanded == [selector]
        assert 'echo' not in registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_mcp_tool_prefix_star_includes_only_matching_names() -> None:
    """tools=['mcp:tool/get_*'] includes only matching names from that provider."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    t1 = registry.register_action(
        name='get_weather', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'get_weather'}
    )
    t2 = registry.register_action(name='get_time', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'get_time'})
    t3 = registry.register_action(name='set_alarm', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'set_alarm'})

    async def dap_fn() -> DapValue:
        return {'tool': [t1, t2, t3]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    result = await expand_wildcard_tools(registry, ['mcp:tool/get_*'])
    assert sorted(result) == [
        '/tool.v2/get_time',
        '/tool.v2/get_weather',
    ]


@pytest.mark.asyncio
async def test_resolve_tool_catalog_key_and_bare_name() -> None:
    """After a local register, both /tool.v2/name and the short name resolve."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = registry.register_action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    by_key = await resolve_tool(registry, '/tool.v2/echo')
    by_name = await resolve_tool(registry, 'echo')
    assert by_key is echo
    assert by_name is echo


@pytest.mark.asyncio
async def test_resolve_tool_rejects_mcp_tool_echo() -> None:
    """resolve_tool('mcp:tool/echo') fails. Expand is the only bind path."""
    registry = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(registry, 'mcp', dap_fn)

    for ref in (
        'mcp:tool/echo',
        'mcp:tool.v2/echo',
        '/dynamic-action-provider/mcp:tool/echo',
        '/tool/echo',
    ):
        with pytest.raises(GenkitError, match=f'Unable to resolve tool {ref}'):
            await resolve_tool(registry, ref)

    assert 'echo' not in registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_after_generate_echo_resolves_as_tool_v2_echo_not_mcp_tool_echo() -> None:
    """After expand, echo resolves as /tool.v2/echo or echo, not as mcp:tool/echo."""
    parent = Registry()

    async def tool_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    define_dynamic_action_provider(parent, 'mcp', dap_fn)
    child = parent.new_child()

    expanded = await expand_wildcard_tools(child, ['mcp:tool/echo'])
    assert expanded == ['/tool.v2/echo']

    assert (await resolve_tool(child, '/tool.v2/echo')) is echo
    assert (await resolve_tool(child, 'echo')) is echo

    with pytest.raises(GenkitError, match='Unable to resolve tool mcp:tool/echo'):
        await resolve_tool(child, 'mcp:tool/echo')
    with pytest.raises(GenkitError, match='Unable to resolve tool /tool.v2/echo'):
        await resolve_tool(parent, '/tool.v2/echo')


@pytest.mark.asyncio
async def test_non_wildcard_names_pass_through() -> None:
    """Non-wildcard names are returned unchanged."""
    registry = Registry()
    result = await expand_wildcard_tools(registry, ['my_tool', 'other_tool'])
    assert result == ['my_tool', 'other_tool']


# ---------------------------------------------------------------------------
# DAP tools resolved inside generate loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_mcp_tool_echo_runs_the_dap_tool() -> None:
    """generate(tools=['mcp:tool/echo']) runs the DAP tool that was never register_action'd."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    call_log: list[str] = []

    class EchoInput(BaseModel):
        text: str

    async def echo_fn(inp: EchoInput) -> str:
        call_log.append(inp.text)
        return f'echoed:{inp.text}'

    # Detached Action — only returned from the DAP; not registered on the root registry.
    echo_action = Action(
        name='echo',
        kind=ActionKind.TOOL,
        fn=echo_fn,
        metadata={'name': 'echo'},
    )

    async def dap_fn() -> DapValue:
        return {'tool': [echo_action]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    # Precondition: `echo` is not a normal root TOOL registration (only in the DAP).
    assert 'echo' not in ai.registry._entries.get(ActionKind.TOOL, {})

    pm.responses = [
        _tool_call_response('echo', {'text': 'hello'}),
        _text_response('done'),
    ]

    response = await ai.generate(
        model='programmableModel',
        prompt='use echo',
        tools=['mcp:tool/echo'],
    )

    assert response.text == 'done'
    assert call_log == ['hello']
    assert pm.last_request is not None
    assert pm.last_request.tools
    # Raw DAP Action — no originalOutputSchema key — still advertises the handler return.
    assert pm.last_request.tools[0].output_schema == {'type': 'string'}
    # Postcondition: resolving/running the tool via DAP still does not
    # persist `echo` under the root registry as a static tool (same check as above).
    assert 'echo' not in ai.registry._entries.get(ActionKind.TOOL, {})
    root_catalog = await ai.registry.list_actions()
    assert '/dynamic-action-provider/mcp' in root_catalog
    assert '/dynamic-action-provider/mcp:tool/echo' not in root_catalog
    assert '/tool.v2/echo' not in root_catalog


@pytest.mark.asyncio
async def test_generate_mcp_tool_does_not_leave_tool_v2_on_the_app() -> None:
    """After generate(tools=['mcp:tool/echo']), the app catalog still has no /tool.v2/echo."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    class Inp(BaseModel):
        x: str

    async def tool_fn(inp: Inp) -> str:
        return inp.x

    # Create an Action directly — NOT registered in root via register_action
    dap_only_action = Action(name='dap_only_tool', kind=ActionKind.TOOL, fn=tool_fn, metadata={'name': 'dap_only_tool'})

    async def dap_fn() -> DapValue:
        return {'tool': [dap_only_action]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    pm.responses = [_text_response('no tools called')]

    await ai.generate(
        model='programmableModel',
        prompt='hi',
        tools=['mcp:tool/dap_only_tool'],
    )

    # Root registry should NOT have dap_only_tool cached — it was never registered there
    root_tools = ai.registry._entries.get(ActionKind.TOOL, {})
    assert 'dap_only_tool' not in root_tools
    root_catalog = await ai.registry.list_actions()
    assert '/dynamic-action-provider/mcp' in root_catalog
    assert '/dynamic-action-provider/mcp:tool/dap_only_tool' not in root_catalog
    assert '/tool.v2/dap_only_tool' not in root_catalog


@pytest.mark.asyncio
async def test_generate_mcp_tool_star_can_run_a_tool_from_that_provider() -> None:
    """generate(tools=['mcp:tool/*']) can run a tool from that provider."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    call_log: list[str] = []

    class InpA(BaseModel):
        x: str

    class InpB(BaseModel):
        x: str

    async def tool_a_fn(inp: InpA) -> str:
        call_log.append(f'a:{inp.x}')
        return f'a:{inp.x}'

    async def tool_b_fn(inp: InpB) -> str:
        call_log.append(f'b:{inp.x}')
        return f'b:{inp.x}'

    tool_a = ai.registry.register_action(name='tool_a', kind=ActionKind.TOOL, fn=tool_a_fn, metadata={'name': 'tool_a'})
    tool_b = ai.registry.register_action(name='tool_b', kind=ActionKind.TOOL, fn=tool_b_fn, metadata={'name': 'tool_b'})

    async def dap_fn() -> DapValue:
        return {'tool': [tool_a, tool_b]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    pm.responses = [
        _tool_call_response('tool_a', {'x': 'hi'}),
        _text_response('finished'),
    ]

    response = await ai.generate(
        model='programmableModel',
        prompt='use a tool',
        tools=['mcp:tool/*'],
    )

    assert response.text == 'finished'
    assert call_log == ['a:hi']


@pytest.mark.asyncio
async def test_generate_mcp2_tool_star_runs_mcp2_when_both_have_echo() -> None:
    """generate(tools=['mcp2:tool/*']) runs mcp2's echo when mcp1 also has echo."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    call_log: list[str] = []

    class Inp(BaseModel):
        x: str

    async def echo1_fn(inp: Inp) -> str:
        call_log.append('mcp1')
        return 'echo 1'

    async def echo2_fn(inp: Inp) -> str:
        call_log.append('mcp2')
        return 'echo 2'

    # Detached Actions (not registered in root registry directly)
    echo1_action = Action(name='echo', kind=ActionKind.TOOL, fn=echo1_fn, metadata={'name': 'echo'})
    echo2_action = Action(name='echo', kind=ActionKind.TOOL, fn=echo2_fn, metadata={'name': 'echo'})

    async def dap1_fn() -> DapValue:
        return {'tool': [echo1_action]}

    async def dap2_fn() -> DapValue:
        return {'tool': [echo2_action]}

    # Register mcp1 first. If resolution falls back to an unqualified lookup, mcp1 will "win".
    ai.define_dynamic_action_provider('mcp1', dap1_fn)
    ai.define_dynamic_action_provider('mcp2', dap2_fn)

    # The model calls the 'echo' tool
    pm.responses = [
        _tool_call_response('echo', {'x': 'hello'}),
        _text_response('finished'),
    ]

    response = await ai.generate(
        model='programmableModel',
        prompt='use echo',
        # Crucially, we explicitly request tools from mcp2 ONLY
        tools=['mcp2:tool/*'],
    )

    assert response.text == 'finished'

    # If the bug is present, this will fail because it will fall back to the unqualified
    # global loop and find mcp1's 'echo' tool instead.
    assert call_log == ['mcp2']


@pytest.mark.asyncio
async def test_generate_mcp_tool_echo_runs_mcp_echo_not_the_local_echo() -> None:
    """generate(tools=['mcp:tool/echo']) runs the DAP echo, not a local echo of the same name."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    call_log: list[str] = []

    class Inp(BaseModel):
        x: str

    @ai.tool(name='echo')
    async def local_echo(inp: Inp) -> str:
        call_log.append('local')
        return 'local'

    async def mcp_echo_fn(inp: Inp) -> str:
        call_log.append('mcp')
        return 'mcp'

    mcp_echo = Action(name='echo', kind=ActionKind.TOOL, fn=mcp_echo_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [mcp_echo]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    pm.responses = [
        _tool_call_response('echo', {'x': 'hello'}),
        _text_response('done'),
    ]

    response = await ai.generate(
        model='programmableModel',
        prompt='use echo',
        tools=['mcp:tool/echo'],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'done'
    assert call_log == ['mcp']
    assert pm.request_count == 2


@pytest.mark.asyncio
async def test_unknown_mcp_provider_fails_before_the_model() -> None:
    """generate(tools=['mcp:nope/echo']) fails before the model is called."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [_text_response('should not run')]

    with pytest.raises(GenkitError) as ei:
        await ai.generate(
            model='programmableModel',
            prompt='hi',
            tools=['mcp:nope/echo'],
        )

    assert ei.value.status == 'NOT_FOUND'
    assert 'Unable to resolve tool mcp:nope/echo' in ei.value.original_message
    assert pm.request_count == 0
    assert pm.last_request is None


@pytest.mark.asyncio
async def test_unknown_mcp_tool_fails_before_the_model() -> None:
    """generate(tools=['mcp:tool/ghost']) fails before the model is called."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [_text_response('should not run')]

    async def echo_fn(x: str) -> str:
        return x

    echo = Action(name='echo', kind=ActionKind.TOOL, fn=echo_fn, metadata={'name': 'echo'})

    async def dap_fn() -> DapValue:
        return {'tool': [echo]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    with pytest.raises(GenkitError) as ei:
        await ai.generate(
            model='programmableModel',
            prompt='hi',
            tools=['mcp:tool/ghost'],
        )

    assert ei.value.status == 'NOT_FOUND'
    assert 'Unable to resolve tool mcp:tool/ghost' in ei.value.original_message
    assert pm.request_count == 0
    assert pm.last_request is None


@pytest.mark.asyncio
async def test_generate_mcp_star_and_local_tool_a_same_name_raises() -> None:
    """tools=['mcp:tool/*', 'toolA'] when both are named toolA raises before the model."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [_text_response('should not run')]

    @ai.tool(name='toolA')
    async def local_tool_a() -> str:
        return 'local'

    async def dap_tool_a_fn() -> str:
        return 'mcp'

    dap_tool_a = Action(name='toolA', kind=ActionKind.TOOL, fn=dap_tool_a_fn, metadata={'name': 'toolA'})

    async def dap_fn() -> DapValue:
        return {'tool': [dap_tool_a]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    with pytest.raises(GenkitError) as ei:
        await ai.generate(
            model='programmableModel',
            prompt='hi',
            tools=['mcp:tool/*', 'toolA'],
        )

    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'Cannot provide two tools with the same name' in ei.value.original_message
    assert pm.request_count == 0
    assert pm.last_request is None


@pytest.mark.asyncio
async def test_generate_local_tool_a_then_mcp_star_same_name_raises() -> None:
    """tools=['toolA', 'mcp:tool/*'] when both are named toolA raises before the model."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.responses = [_text_response('should not run')]

    @ai.tool(name='toolA')
    async def local_tool_a() -> str:
        return 'local'

    async def dap_tool_a_fn() -> str:
        return 'mcp'

    dap_tool_a = Action(name='toolA', kind=ActionKind.TOOL, fn=dap_tool_a_fn, metadata={'name': 'toolA'})

    async def dap_fn() -> DapValue:
        return {'tool': [dap_tool_a]}

    ai.define_dynamic_action_provider('mcp', dap_fn)

    with pytest.raises(GenkitError) as ei:
        await ai.generate(
            model='programmableModel',
            prompt='hi',
            tools=['toolA', 'mcp:tool/*'],
        )

    assert ei.value.status == 'INVALID_ARGUMENT'
    assert 'Cannot provide two tools with the same name' in ei.value.original_message
    assert pm.request_count == 0
    assert pm.last_request is None
