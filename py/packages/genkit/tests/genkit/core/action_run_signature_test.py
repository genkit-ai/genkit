# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Action.run and Action.stream take the input positionally and everything else by keyword."""

import asyncio

import pytest

from genkit import FlowContext
from genkit._core._action import Action, ActionKind


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


@pytest.mark.asyncio
async def test_run_with_keywords_streams_chunks_and_returns() -> None:
    """`run(input, on_chunk=..., context=...)` streams to on_chunk and returns the result."""
    chunks: list[str] = []

    result = await _kitchen().run('salmon', on_chunk=chunks.append, context={'table': 4})

    assert chunks == ['plating salmon']
    assert result.response == 'salmon served'


@pytest.mark.asyncio
async def test_stream_abort_signal_reaches_the_action() -> None:
    """`stream(abort_signal=ev)` hands `ev` to the action as `ctx.abort_signal`, so setting it stops the run."""

    async def simmer(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('simmering')
        await ctx.abort_signal.wait()
        return f'{dish} pulled off the heat'

    abort = asyncio.Event()
    result = Action(ActionKind.CUSTOM, 'simmer', simmer).stream('stock', abort_signal=abort)

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
