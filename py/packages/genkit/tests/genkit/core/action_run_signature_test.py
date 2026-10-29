# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Action.run and Action.stream take the input positionally and everything else by keyword."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest

from genkit import FlowContext
from genkit._core._action import Action, ActionKind, BidiAction, get_current_context


async def _plate(dish: str, ctx: FlowContext[str]) -> str:
    ctx.send_chunk(f'plating {dish}')
    return f'{dish} served'


def _kitchen() -> Action[str, str, str]:
    return Action(ActionKind.CUSTOM, 'plate', _plate)


@pytest.mark.asyncio
async def test_run_with_positional_on_chunk_raises_type_error() -> None:
    """`run(input, on_chunk)` raises TypeError; on_chunk is keyword-only."""
    chunks: list[str] = []

    with pytest.raises(TypeError, match='positional argument'):
        await _kitchen().run('salmon', chunks.append)  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]  # pyrefly: ignore[bad-argument-count]


def test_stream_with_positional_context_raises_type_error() -> None:
    """`stream(input, context)` raises TypeError; context is keyword-only."""
    with pytest.raises(TypeError, match='positional argument'):
        _kitchen().stream('salmon', {'table': 4})  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]  # pyrefly: ignore[bad-argument-count]


async def _table_echo(init: str, inputs: AsyncIterator[str], send_chunk: Callable[[str], None]) -> dict[str, Any]:
    async for dish in inputs:
        send_chunk(dish)
    return {'init': init, 'context': get_current_context()}


@pytest.mark.asyncio
async def test_stream_bidi_with_positional_context_raises_type_error() -> None:
    """`stream_bidi(init, context)` raises TypeError; options after init are keyword-only, like run/stream."""
    session: BidiAction[str, dict[str, Any], str, str] = BidiAction(ActionKind.CUSTOM, 'table', _table_echo)

    with pytest.raises(TypeError, match='positional argument'):
        await session.stream_bidi('table-4', {'table': 4})  # pyright: ignore[reportCallIssue]  # ty: ignore[too-many-positional-arguments]  # pyrefly: ignore[bad-argument-count]


@pytest.mark.asyncio
async def test_stream_bidi_with_keyword_context_reaches_the_session() -> None:
    """`stream_bidi(init, context=...)` runs the session with that request context."""
    session: BidiAction[str, dict[str, Any], str, str] = BidiAction(ActionKind.CUSTOM, 'table', _table_echo)

    conn = await session.stream_bidi('table-4', context={'table': 4})
    await conn.send('soup')
    await conn.close()

    assert [chunk async for chunk in conn.receive()] == ['soup']
    assert await conn.output() == {'init': 'table-4', 'context': {'table': 4}}


@pytest.mark.asyncio
async def test_run_rejects_public_abort_signal_kwarg() -> None:
    """`run(abort_signal=...)` raises TypeError; cancellation uses standard asyncio task cancellation."""
    abort = asyncio.Event()

    with pytest.raises(TypeError, match='abort_signal'):
        await _kitchen().run('salmon', abort_signal=abort)  # pyright: ignore[reportCallIssue]  # ty: ignore[unknown-argument]  # pyrefly: ignore[unexpected-keyword]


def test_stream_rejects_public_abort_signal_kwarg() -> None:
    """`stream(abort_signal=...)` raises TypeError; cancellation uses standard asyncio stream/task cancellation."""
    abort = asyncio.Event()

    with pytest.raises(TypeError, match='abort_signal'):
        _kitchen().stream('salmon', abort_signal=abort)  # pyright: ignore[reportCallIssue]  # ty: ignore[unknown-argument]  # pyrefly: ignore[unexpected-keyword]


@pytest.mark.asyncio
async def test_private_stream_abort_signal_reaches_the_action() -> None:
    """`_stream(abort_signal=ev)` hands `ev` to the action as `ctx.abort_signal`, so setting it stops the run."""

    async def simmer(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('simmering')
        await ctx.abort_signal.wait()
        return f'{dish} pulled off the heat'

    abort = asyncio.Event()
    result = Action(ActionKind.CUSTOM, 'simmer', simmer)._stream('stock', abort_signal=abort)

    first = await anext(aiter(result.stream))
    abort.set()

    assert first == 'simmering'
    assert await asyncio.wait_for(result.response, timeout=1) == 'stock pulled off the heat'


@pytest.mark.asyncio
async def test_private_run_reports_trace_start_before_the_action_body() -> None:
    """`_run(on_trace_start=...)` gets the trace and span ids before the fn runs, which the Dev UI needs for cancel."""
    started: list[tuple[str, str]] = []
    seen_at_body: list[int] = []

    async def on_trace_start(trace_id: str, span_id: str) -> None:
        started.append((trace_id, span_id))

    async def plate(dish: str) -> str:
        seen_at_body.append(len(started))
        return f'{dish} served'

    result = await Action(ActionKind.CUSTOM, 'plate', plate)._run('salmon', on_trace_start=on_trace_start)

    assert seen_at_body == [1]
    assert started == [(result.trace_id, result.span_id)]
    assert result.response == 'salmon served'
