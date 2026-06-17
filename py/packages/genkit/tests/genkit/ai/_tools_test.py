# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool restart builder and run_tool_after_restart."""

import pytest

from genkit import ActionKind, Genkit
from genkit._ai._generate import run_tool_after_restart
from genkit._ai._tools import (
    Interrupt,
    ToolRunContext,
    _tool_original_input,
    _tool_resumed_metadata,
    respond_to_interrupt,
    restart_tool,
)
from genkit._core._error import GenkitError
from genkit._core._typing import ToolRequest, ToolRequestPart, ToolResponsePart


def test_restart_sets_resumed_metadata_and_preserves_interrupt() -> None:
    """``restart_tool``: copy interrupt metadata, set ``resumed``; ``interrupt`` stays on the restart TRP."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': {'reason': 'hold'}},
    )
    out = restart_tool(interrupt_trp, resumed_metadata={'k': 'v'})
    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('resumed') == {'k': 'v'}
    assert out.metadata.get('interrupt') == {'reason': 'hold'}
    assert out.tool_request.input == {'amount': 10}


def test_restart_replace_input_sets_replaced_input() -> None:
    """Restart with new input sets ``replacedInput`` to prior input and updates ``tool_request.input``."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': True},
    )
    out = restart_tool(interrupt_trp, resumed_metadata={'by': 'u'}, replace_input={'amount': 99})
    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('replacedInput') == {'amount': 10}
    assert out.tool_request.input == {'amount': 99}
    assert out.metadata.get('resumed') == {'by': 'u'}
    assert out.metadata.get('interrupt') is True


def test_restart_resumed_defaults_to_true() -> None:
    """When ``resumed_metadata=None``, restart TRP sets ``metadata.resumed`` to True."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={}),
        metadata={'interrupt': True},
    )
    out = restart_tool(interrupt_trp, resumed_metadata=None)
    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('resumed') is True
    assert out.metadata.get('interrupt') is True


@pytest.mark.asyncio
async def test_run_tool_after_restart_resumed_true_maps_to_empty_dict_in_context() -> None:
    """``run_tool_after_restart``: ``metadata.resumed is True`` → ``ToolRunContext.resumed_metadata`` is ``{}``."""
    ai = Genkit()
    captured: list[tuple[dict | None, object | None]] = []

    @ai.tool(name='t2')
    async def t2(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        captured.append((ctx.resumed_metadata, ctx.original_input))
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t2')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t2', ref='x', input={'q': 1}),
        metadata={'resumed': True},
    )
    await run_tool_after_restart(action, restart_trp)
    assert len(captured) == 1
    assert captured[0][0] == {}
    assert captured[0][1] is None


@pytest.mark.asyncio
async def test_run_tool_after_restart_resumed_dict() -> None:
    """Restart TRP with ``metadata.resumed`` dict is passed through to ``ToolRunContext.resumed_metadata``."""
    ai = Genkit()
    captured: list[dict | None] = []

    @ai.tool(name='t2')
    async def t2(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        captured.append(ctx.resumed_metadata)
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t2')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t2', ref='x', input={}),
        metadata={'resumed': {'by': 'x'}},
    )
    await run_tool_after_restart(action, restart_trp)
    assert captured == [{'by': 'x'}]


@pytest.mark.asyncio
async def test_run_tool_after_restart_replaced_input() -> None:
    """``replacedInput`` on TRP sets tool input from current request and ``original_input`` from prior."""
    ai = Genkit()
    captured: list[tuple[object, object | None]] = []

    @ai.tool(name='t2')
    async def t2(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        captured.append((inp, ctx.original_input))
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t2')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t2', ref='x', input={'new': True}),
        metadata={'resumed': True, 'replacedInput': {'old': True}},
    )
    await run_tool_after_restart(action, restart_trp)
    assert len(captured) == 1
    assert captured[0][0] == {'new': True}
    assert captured[0][1] == {'old': True}


@pytest.mark.asyncio
async def test_run_tool_after_restart_resets_contextvars() -> None:
    """After ``run_tool_after_restart`` returns, resume ContextVars are cleared (no leak between runs)."""
    ai = Genkit()

    @ai.tool(name='t2')
    async def t2(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t2')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t2', ref='x', input={}),
        metadata={'resumed': True},
    )
    await run_tool_after_restart(action, restart_trp)
    assert _tool_resumed_metadata.get() is None
    assert _tool_original_input.get() is None


@pytest.mark.asyncio
async def test_run_tool_after_restart_nested_interrupt_raises() -> None:
    """Tool raising ``Interrupt`` during a restart run raises ``GenkitError`` (nested interrupt unsupported)."""
    ai = Genkit()

    @ai.tool(name='t2')
    async def t2(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        raise Interrupt()

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t2')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t2', ref='x', input={}),
        metadata={'resumed': True},
    )
    with pytest.raises(GenkitError) as ei:
        await run_tool_after_restart(action, restart_trp)
    assert ei.value.status == 'FAILED_PRECONDITION'
    assert 'interrupted again' in ei.value.original_message.lower()


def test_respond_to_interrupt_wire_format_basic() -> None:
    """respond_to_interrupt produces a ToolResponsePart with matching ref/name and interruptResponse metadata."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='ask_user', ref='ref-abc', input={'question': 'ok?'}),
        metadata={'interrupt': {'reason': 'needs_approval'}},
    )

    result = respond_to_interrupt('yes', interrupt=interrupt_trp)

    assert isinstance(result, ToolResponsePart)
    assert result.tool_response.name == 'ask_user'
    assert result.tool_response.ref == 'ref-abc'
    assert result.tool_response.output == 'yes'
    assert result.metadata is not None
    assert result.metadata.get('interruptResponse') is True


def test_respond_to_interrupt_wire_format_with_metadata() -> None:
    """respond_to_interrupt attaches custom metadata under interruptResponse key."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='confirm', ref='ref-xyz', input={}),
        metadata={'interrupt': True},
    )

    result = respond_to_interrupt({'approved': True}, interrupt=interrupt_trp, metadata={'by': 'admin'})

    assert result.tool_response.ref == 'ref-xyz'
    assert result.tool_response.output == {'approved': True}
    assert result.metadata is not None
    assert result.metadata.get('interruptResponse') == {'by': 'admin'}


def test_restart_tool_does_not_require_tool_reference() -> None:
    """``restart_tool`` works from an interrupt alone — no ``Tool`` needed.

    Middleware-contributed tools (``read_file`` from a filesystem middleware,
    anything gated by a ``ToolApproval`` middleware) never give the caller
    a ``Tool`` reference; they just appear in ``response.interrupts``. The
    helper has to be callable from the interrupt by itself.
    """
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='middleware_tool', ref='r1', input={'p': 1}),
        metadata={'interrupt': True},
    )

    out = restart_tool(interrupt_trp, resumed_metadata={'toolApproved': True})

    assert out.tool_request.name == 'middleware_tool'
    assert out.tool_request.input == {'p': 1}
    assert out.metadata is not None
    assert out.metadata.get('resumed') == {'toolApproved': True}


def test_restart_preserves_ref_on_wire() -> None:
    """``restart_tool`` preserves the original tool_request.ref so the resumed TRP can be correlated."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='corr-id-1', input={'amount': 50}),
        metadata={'interrupt': True},
    )
    out = restart_tool(interrupt_trp)

    assert out.tool_request.ref == 'corr-id-1'


@pytest.mark.asyncio
async def test_run_tool_after_restart_response_preserves_ref() -> None:
    """run_tool_after_restart produces a ToolResponsePart whose ref matches the restart TRP's ref."""
    ai = Genkit()

    @ai.tool(name='t_ref')
    async def t_ref(inp: dict) -> str:  # noqa: ARG001
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t_ref')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t_ref', ref='wire-ref-99', input={}),
        metadata={'resumed': True},
    )
    part = await run_tool_after_restart(action, restart_trp)
    assert part.tool_response.ref == 'wire-ref-99'


@pytest.mark.asyncio
async def test_run_tool_after_restart_response_preserves_ref_and_uses_new_input() -> None:
    """``run_tool_after_restart`` returns a ToolResponsePart whose ref matches the restart TRP;
    ``tool_request.input`` is what ``tool.run`` receives, and ``metadata.replacedInput`` is
    ``ToolRunContext.original_input`` (prior interrupted input).
    """
    ai = Genkit()
    received_inputs: list[dict] = []
    original_inputs: list[object | None] = []

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolRunContext) -> str:
        received_inputs.append(dict(inp))
        original_inputs.append(ctx.original_input)
        if not inp.get('confirmed'):
            raise Interrupt({'reason': 'needs_approval'})
        return f'transferred {inp.get("amount")}'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='transfer')
    assert action is not None

    prior = {'amount': 100, 'confirmed': False}
    # Simulate a restart TRP: original input had confirmed=False, new input has confirmed=True.
    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='transfer', ref='ref-42', input={'amount': 100, 'confirmed': True}),
        metadata={'resumed': True, 'replacedInput': prior},
    )
    result = await run_tool_after_restart(action, restart_trp)

    # Ref is preserved from the restart TRP.
    assert result.tool_response.ref == 'ref-42'
    assert result.tool_response.name == 'transfer'
    # Primary arg is current tool_request.input; replacedInput is surfaced as original_input.
    assert received_inputs == [{'amount': 100, 'confirmed': True}]
    assert original_inputs == [prior]
    assert result.tool_response.output == 'transferred 100'
