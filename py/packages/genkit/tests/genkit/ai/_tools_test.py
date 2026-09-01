# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for tool restart builder and run_tool_after_restart."""

import pytest

from genkit import ActionKind, Genkit
from genkit._ai._tools import (
    Interrupt,
    ToolRunContext,
    _tool_original_input,
    _tool_resumed_metadata,
    respond_to_interrupt,
    restart_interrupt_error,
    restart_tool,
    run_tool_after_restart,
)
from genkit._core._error import GenkitError
from genkit._core._middleware import GenerateMiddlewareContext
from genkit._core._typing import ToolRequest, ToolRequestPart, ToolResponsePart


async def _echo_tool(x: object) -> object:
    return x


def test_restart_sets_resumed_metadata_and_preserves_interrupt() -> None:
    """``restart_tool``: copy interrupt metadata, set ``resumed``; ``interrupt`` stays on the restart TRP."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='r1', input={'amount': 10}),
        metadata={'interrupt': {'reason': 'hold'}},
    )
    out = restart_tool(interrupt=interrupt_trp, resumed_metadata={'k': 'v'})
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
    out = restart_tool(replace_input={'amount': 99}, interrupt=interrupt_trp, resumed_metadata={'by': 'u'})
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
    out = restart_tool(interrupt=interrupt_trp, resumed_metadata=None)
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
    await run_tool_after_restart(tool=action, restart_trp=restart_trp)
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
    await run_tool_after_restart(tool=action, restart_trp=restart_trp)
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
    await run_tool_after_restart(tool=action, restart_trp=restart_trp)
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
    await run_tool_after_restart(tool=action, restart_trp=restart_trp)
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
        await run_tool_after_restart(tool=action, restart_trp=restart_trp)
    assert ei.value.status == 'FAILED_PRECONDITION'
    assert 'interrupted again' in ei.value.original_message.lower()
    assert isinstance(ei.value.cause, Interrupt)


def test_restart_interrupt_error_accepts_string_metadata() -> None:
    """Plain-string Interrupt metadata must not crash; use it as the reason."""
    intr = Interrupt('plain string reason')  # type: ignore[arg-type]
    err = restart_interrupt_error(intr)
    assert err.status == 'FAILED_PRECONDITION'
    assert err.original_message == 'Tool interrupted again during restart: plain string reason'


@pytest.mark.asyncio
async def test_run_tool_after_restart_nested_interrupt_includes_reason() -> None:
    """Nested restart Interrupt with ``metadata.message`` is surfaced in the GenkitError text."""
    ai = Genkit()

    @ai.tool(name='t3')
    async def t3(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        raise Interrupt({'message': 'Tool not in approved list: t3'})

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='t3')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='t3', ref='x', input={}),
        metadata={'resumed': True},
    )
    with pytest.raises(GenkitError) as ei:
        await run_tool_after_restart(tool=action, restart_trp=restart_trp)
    assert ei.value.status == 'FAILED_PRECONDITION'
    assert ei.value.original_message == ('Tool interrupted again during restart: Tool not in approved list: t3')
    assert isinstance(ei.value.cause, Interrupt)


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


def test_restart_tool_directly() -> None:
    """``restart_tool`` works directly without a ``Tool`` reference."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='middleware_tool', ref='r1', input={'p': 1}),
        metadata={'interrupt': True},
    )
    out = restart_tool(interrupt=interrupt_trp, resumed_metadata={'tool_approved': True})

    assert out.tool_request.name == 'middleware_tool'
    assert out.tool_request.input == {'p': 1}
    assert out.metadata is not None
    assert out.metadata.get('resumed') == {'tool_approved': True}


def test_restart_preserves_ref_on_wire() -> None:
    """``restart_tool`` preserves the original tool_request.ref so the resumed TRP can be correlated."""
    interrupt_trp = ToolRequestPart(
        tool_request=ToolRequest(name='pay', ref='corr-id-1', input={'amount': 50}),
        metadata={'interrupt': True},
    )
    out = restart_tool(interrupt=interrupt_trp)

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
    part = await run_tool_after_restart(tool=action, restart_trp=restart_trp)
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
    result = await run_tool_after_restart(tool=action, restart_trp=restart_trp)

    # Ref is preserved from the restart TRP.
    assert result.tool_response.ref == 'ref-42'
    assert result.tool_response.name == 'transfer'
    # Primary arg is current tool_request.input; replacedInput is surfaced as original_input.
    assert received_inputs == [{'amount': 100, 'confirmed': True}]
    assert original_inputs == [prior]
    assert result.tool_response.output == 'transferred 100'


@pytest.mark.asyncio
async def test_run_tool_after_restart_pipes_generate_context() -> None:
    """``run_tool_after_restart(..., ctx=ctx)`` pipes custom_context into ``ToolRunContext.context``."""
    ai = Genkit()
    seen: list[dict[str, object]] = []

    @ai.tool(name='ctx_restart_tool')
    async def ctx_restart_tool(inp: dict, ctx: ToolRunContext) -> str:  # noqa: ARG001
        seen.append(dict(ctx.context))
        return 'resumed_ok'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='ctx_restart_tool')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='ctx_restart_tool', ref='r1', input={}),
        metadata={'resumed': True},
    )
    mw_ctx = GenerateMiddlewareContext(ai, custom_context={'auth_role': 'admin'})
    await run_tool_after_restart(tool=action, restart_trp=restart_trp, ctx=mw_ctx)

    assert seen == [{'auth_role': 'admin'}]


@pytest.mark.asyncio
async def test_tool_with_only_tool_run_context_param_is_valid() -> None:
    """Tool with ``ToolRunContext`` as its sole parameter defines and executes.

    Regression for #4492: the parameter must not be treated as the tool's input
    type (previously crashed with ``PydanticSchemaGenerationError``), and the
    tool must receive a ``ToolRunContext`` instead of the raw input at runtime.
    """
    ai = Genkit()
    seen: list[ToolRunContext] = []

    @ai.tool(name='ctx_only')
    async def get_user(ctx: ToolRunContext) -> str:
        seen.append(ctx)
        raw_user = ctx.context.get('user')
        user_id = 0
        if isinstance(raw_user, dict):
            user_id = int(raw_user.get('id', 0))
        if user_id == 42:
            return 'User is Arthur Dent, an intergalactic traveler.'
        return 'User is Guest.'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='ctx_only')
    assert action is not None
    # No input schema is inferred for a context-only parameter.
    assert action.input_schema == {}
    assert action.input_type is None

    resp = await action.run(input={}, context={'user': {'id': 42}})
    assert resp.response == 'User is Arthur Dent, an intergalactic traveler.'
    assert len(seen) == 1
    assert isinstance(seen[0], ToolRunContext)
    assert seen[0].context == {'user': {'id': 42}}
    assert seen[0].resumed_metadata is None


@pytest.mark.asyncio
async def test_run_tool_after_restart_with_ctx_only_tool() -> None:
    """Context-only tools get resume metadata: ``metadata.resumed`` surfaces as ``resumed_metadata``."""
    ai = Genkit()
    captured: list[tuple[dict | None, object | None]] = []

    @ai.tool(name='ctx_only_restart')
    async def check(ctx: ToolRunContext) -> str:
        captured.append((ctx.resumed_metadata, ctx.original_input))
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='ctx_only_restart')
    assert action is not None

    restart_trp = ToolRequestPart(
        tool_request=ToolRequest(name='ctx_only_restart', ref='x', input={}),
        metadata={'resumed': True, 'replacedInput': {'old': True}},
    )
    await run_tool_after_restart(tool=action, restart_trp=restart_trp)

    assert len(captured) == 1
    assert captured[0][0] == {}
    assert captured[0][1] == {'old': True}


@pytest.mark.asyncio
async def test_tool_with_only_tool_run_context_param_with_default() -> None:
    """A context-only tool whose param has a default still gets a ``ToolRunContext``.

    Regression for the case where ``_first_arg_optional`` is True: without
    explicit input the dispatch must pass the context, not call the tool with
    no arguments (previously hit ``IndexError`` in the tool wrapper).
    """
    ai = Genkit()
    seen: list[ToolRunContext] = []

    @ai.tool(name='ctx_only_default')
    async def get_user(ctx: ToolRunContext = None) -> str:  # noqa: ARG001
        if ctx is not None:
            seen.append(ctx)
        return 'done'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='ctx_only_default')
    assert action is not None

    resp = await action.run(input=None, context={'user': {'id': 42}})
    assert resp.response == 'done'
    assert len(seen) == 1
    assert isinstance(seen[0], ToolRunContext)
    assert seen[0].context == {'user': {'id': 42}}


@pytest.mark.asyncio
async def test_tool_with_input_and_context_params_unaffected() -> None:
    """``(input, ctx)`` tools keep receiving the input and a ``ToolRunContext``."""
    ai = Genkit()
    seen: list[tuple[dict, ToolRunContext]] = []

    @ai.tool(name='inp_ctx')
    async def echo(inp: dict, ctx: ToolRunContext) -> str:
        seen.append((inp, ctx))
        return 'echoed'

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='inp_ctx')
    assert action is not None
    assert action.input_schema.get('type') == 'object'

    resp = await action.run(input={'msg': 'hi'}, context={'auth': 'role'})
    assert resp.response == 'echoed'
    assert len(seen) == 1
    assert seen[0][0] == {'msg': 'hi'}
    assert isinstance(seen[0][1], ToolRunContext)
    assert seen[0][1].context == {'auth': 'role'}


@pytest.mark.asyncio
async def test_tool_with_single_input_param_unaffected() -> None:
    """A tool with a single input parameter keeps receiving the raw input."""
    ai = Genkit()

    @ai.tool(name='echo_input')
    async def echo_input(x: dict) -> dict:
        return x

    action = await ai.registry.resolve_action(kind=ActionKind.TOOL, name='echo_input')
    assert action is not None

    resp = await action.run({'a': 1})
    assert resp.response == {'a': 1}
