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


@pytest.mark.asyncio
async def test_restart_sets_resumed_metadata_and_preserves_interrupt() -> None:
    """``tool.restart``: copy interrupt metadata, set ``resumed``; ``interrupt`` stays on the restart TRP."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': {'reason': 'hold'}},
    )
    out = pay.restart(None, interrupt=interrupt_trp, resumed_metadata={'k': 'v'})
    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('resumed') == {'k': 'v'}
    assert out.metadata.get('interrupt') == {'reason': 'hold'}
    assert out.tool_request.input == {'amount': 10}


@pytest.mark.asyncio
async def test_restart_replace_input_sets_replaced_input() -> None:
    """Restart with new input sets ``replacedInput`` to prior input and updates ``tool_request.input``."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': True},
    )
    out = pay.restart({'amount': 99}, interrupt=interrupt_trp, resumed_metadata={'by': 'u'})
    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('replacedInput') == {'amount': 10}
    assert out.tool_request.input == {'amount': 99}
    assert out.metadata.get('resumed') == {'by': 'u'}
    assert out.metadata.get('interrupt') is True


@pytest.mark.asyncio
async def test_restart_resumed_defaults_to_true() -> None:
    """When ``resumed_metadata=None``, restart TRP sets ``metadata.resumed`` to True."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={}),
        metadata={'interrupt': True},
    )
    out = pay.restart(None, interrupt=interrupt_trp, resumed_metadata=None)
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


@pytest.mark.asyncio
async def test_restart_tool_matches_method_no_replace_input() -> None:
    """``restart_tool`` is equivalent to calling ``tool.restart`` (no replace_input)."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': {'reason': 'hold'}},
    )

    via_helper = restart_tool(tool=pay, interrupt=interrupt_trp, resumed_metadata={'k': 'v'})
    via_method = pay.restart(None, interrupt=interrupt_trp, resumed_metadata={'k': 'v'})

    assert via_helper.model_dump() == via_method.model_dump()


@pytest.mark.asyncio
async def test_restart_tool_with_replace_input() -> None:
    """``restart_tool`` forwards ``replace_input`` and matches ``tool.restart``."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': True},
    )

    out = restart_tool({'amount': 99}, tool=pay, interrupt=interrupt_trp, resumed_metadata={'by': 'u'})

    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('replacedInput') == {'amount': 10}
    assert out.tool_request.input == {'amount': 99}
    assert out.metadata.get('resumed') == {'by': 'u'}
    assert out.metadata.get('interrupt') is True


@pytest.mark.asyncio
async def test_restart_tool_resumed_defaults_to_true() -> None:
    """``restart_tool`` with ``resumed_metadata=None`` sets ``metadata.resumed`` to True."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={}),
        metadata={'interrupt': True},
    )

    out = restart_tool(tool=pay, interrupt=interrupt_trp)

    assert isinstance(out, ToolRequestPart)
    assert out.metadata is not None
    assert out.metadata.get('resumed') is True
    assert out.tool_request.ref == 'r1'


@pytest.mark.asyncio
async def test_restart_tool_rejects_mismatched_tool() -> None:
    """``restart_tool`` propagates the ``Tool.restart`` name-mismatch guard."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='other', ref='r1', input={}),
        metadata={'interrupt': True},
    )

    with pytest.raises(ValueError, match="Interrupt is for tool 'other', not 'pay'"):
        restart_tool(tool=pay, interrupt=interrupt_trp)


def test_restart_preserves_ref_on_wire() -> None:
    """restart() preserves the original tool_request.ref so the resumed TRP can be correlated."""
    ai = Genkit()

    @ai.tool(name='pay')
    async def pay(inp: dict) -> str:  # noqa: ARG001
        return 'ok'

    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='corr-id-1', input={'amount': 50}),
        metadata={'interrupt': True},
    )
    out = pay.restart(None, interrupt=interrupt_trp)

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
