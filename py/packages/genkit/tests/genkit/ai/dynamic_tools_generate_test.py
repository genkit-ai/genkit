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
from genkit._ai._generate import expand_wildcard_tools
from genkit._ai._testing import define_programmable_model
from genkit._core._action import Action, ActionKind
from genkit._core._dap import DapValue, define_dynamic_action_provider
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
async def test_expand_wildcard_all() -> None:
    """'provider:tool/*' expands to all tools from the DAP."""
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
        '/dynamic-action-provider/mcp:tool/echo',
        '/dynamic-action-provider/mcp:tool/ping',
    ]


@pytest.mark.asyncio
async def test_expand_wildcard_prefix() -> None:
    """'provider:tool/prefix*' expands only matching tools."""
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
        '/dynamic-action-provider/mcp:tool/get_time',
        '/dynamic-action-provider/mcp:tool/get_weather',
    ]


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
async def test_dap_tool_resolved_in_generate() -> None:
    """generate resolves and runs a tool that is only advertised via a DAP (never register_action)."""
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
    # Postcondition: resolving/running the tool via DAP still does not
    # persist `echo` under the root registry as a static tool (same check as above).
    assert 'echo' not in ai.registry._entries.get(ActionKind.TOOL, {})


@pytest.mark.asyncio
async def test_dap_tools_do_not_pollute_root_registry() -> None:
    """After generate, DAP-resolved tools are not cached in the root registry."""
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


@pytest.mark.asyncio
async def test_wildcard_tools_in_generate() -> None:
    """Wildcard tool pattern is expanded before generate resolves tools."""
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
async def test_wildcard_tools_avoids_shadowing_conflict() -> None:
    """Explicit wildcard provider paths should not be shadowed by earlier providers."""
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
