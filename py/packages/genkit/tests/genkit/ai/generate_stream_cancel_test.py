# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Stopping `ai.generate_stream` / `prompt.stream` early stops the model call."""

import asyncio
import contextlib
import sys
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest

from genkit import (
    FinishReason,
    FlowContext,
    Genkit,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    Role,
    ToolContext,
)
from genkit._core._action import ActionRunContext
from genkit._core._model import ModelRequest
from genkit.exp.agent import AgentContext
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import ModelContext, ToolRequest

T = TypeVar('T')


class SlowModel:
    """Sends 'one ', then waits for `release`, then sends 'two '."""

    def __init__(self, *, release: bool = False, fail_midway: bool = False) -> None:
        self.release = asyncio.Event()
        if release:
            self.release.set()
        self.fail_midway = fail_midway
        self.cancelled = asyncio.Event()
        self.calls = 0

    async def __call__(self, request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        self.calls += 1
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('one ')]))
        if self.fail_midway:
            raise ValueError('model blew up')
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('two ')]))
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('one two')]),
        )


def setup(**kwargs: bool) -> tuple[Genkit, SlowModel]:
    ai = Genkit()
    model = SlowModel(**kwargs)

    async def slow(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return await model(request, ctx)

    ai.define_model(name='slow', fn=slow)
    return ai, model


async def settle(aw: Awaitable[T]) -> T:
    # guards against a hang turning into a stuck test run; never the thing that passes a test.
    return await asyncio.wait_for(aw, timeout=5)


@pytest.mark.asyncio
async def test_contexts_do_not_expose_abort_signal() -> None:
    """Action, flow, tool, model, agent, and middleware contexts omit abort_signal."""
    ai = Genkit()
    action_ctx: ActionRunContext[str] = ActionRunContext()
    flow_ctx: FlowContext[str] = FlowContext()
    tool_ctx: ToolContext[str] = ToolContext(action_ctx)
    agent_ctx: AgentContext = ActionRunContext()
    mw_ctx = MiddlewareContext(ai=ai)

    for ctx_obj in (action_ctx, flow_ctx, tool_ctx, agent_ctx, mw_ctx):
        assert not hasattr(ctx_obj, 'abort_signal')

    seen_has_attr: list[bool] = []

    @ai.tool()
    async def check_pantry(item: str, ctx: ToolContext) -> str:
        seen_has_attr.append(hasattr(ctx, 'abort_signal'))
        return f'stocked:{item}'

    class InspectMiddleware(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            seen_has_attr.append(hasattr(ctx, 'abort_signal'))
            return await next_fn(params, ctx)

    turns = 0

    async def model_fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        nonlocal turns
        turns += 1
        seen_has_attr.append(hasattr(ctx, 'abort_signal'))
        if turns == 1:
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='check_pantry', ref='1', input='flour'))],
                ),
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ready')]),
        )

    ai.define_model(name='inspector', fn=model_fn)

    @ai.flow()
    async def run_flow(prompt_text: str, ctx: FlowContext) -> str:
        seen_has_attr.append(hasattr(ctx, 'abort_signal'))
        resp = await ai.generate(
            model='inspector',
            prompt=prompt_text,
            tools=['check_pantry'],
            use=[InspectMiddleware()],
        )
        return resp.text

    assert await run_flow('bake') == 'ready'
    assert seen_has_attr == [False, False, False, False, False, False]


@pytest.mark.asyncio
async def test_generate_stream_immediate_aclose_before_iterating_returns_aborted_without_calling_model() -> None:
    """Immediate `await stream.stream.aclose()` on tick 0 resolves `.response` to ABORTED with zero model calls."""
    ai, model = setup(release=True)

    stream = ai.generate_stream(model='slow', prompt='hi')
    await settle(stream.stream.aclose())
    response = await settle(stream.response)

    assert model.calls == 0
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.messages[0].text == 'hi'
    assert response.text == ''


@pytest.mark.asyncio
async def test_generate_stream_break_after_first_chunk_cancels_model_and_response_is_aborted() -> None:
    """`break` after the first chunk of `ai.generate_stream(...)` cancels the model; `.response` is ABORTED."""
    ai, model = setup()

    stream = ai.generate_stream(model='slow', prompt='hi')
    seen = []
    async for chunk in stream:
        seen.append(chunk.text)
        break
    response = await settle(stream.response)

    assert seen == ['one ']
    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.text == ''


@pytest.mark.asyncio
async def test_generate_stream_consumer_task_cancelled_cancels_model_and_response_is_aborted() -> None:
    """Cancelling the task iterating `ai.generate_stream(...)` cancels the model; `.response` is ABORTED."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')
    first_chunk = asyncio.Event()

    async def read() -> None:
        async for _ in stream:
            first_chunk.set()

    reader = asyncio.create_task(read())
    await settle(first_chunk.wait())
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    response = await settle(stream.response)

    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.text == ''


@pytest.mark.asyncio
async def test_generate_stream_iterated_to_end_returns_completed_response() -> None:
    """Reading `ai.generate_stream(...)` to the end yields every chunk; `.response` is the finished reply."""
    ai, model = setup(release=True)

    stream = ai.generate_stream(model='slow', prompt='hi')
    seen = [chunk.text async for chunk in stream]
    response = await settle(stream.response)

    assert seen == ['one ', 'two ']
    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_generate_stream_reading_one_chunk_via_stream_then_iterating_gets_the_rest() -> None:
    """`await s.stream.__anext__()` then `async for c in s` gets 'one ' then 'two ' and the finished reply."""
    ai, model = setup(release=True)

    s = ai.generate_stream(model='slow', prompt='hi')
    first = await settle(s.stream.__anext__())
    rest = [chunk.text async for chunk in s]
    response = await settle(s.response)

    assert first.text == 'one '
    assert rest == ['two ']
    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_generate_stream_model_fails_midway_ends_loop_and_response_is_failed() -> None:
    """A model that raises after one chunk ends the `async for` normally; `.response` is FAILED."""
    ai, _ = setup(fail_midway=True)

    stream = ai.generate_stream(model='slow', prompt='hi')
    seen = [chunk.text async for chunk in stream]
    response = await settle(stream.response)

    assert seen == ['one ']
    assert response.finish_reason == FinishReason.FAILED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.text == ''


@pytest.mark.asyncio
async def test_generate_stream_await_response_without_iterating_runs_to_completion() -> None:
    """Awaiting `.response` of `ai.generate_stream(...)` without reading chunks returns the finished reply."""
    ai, model = setup(release=True)

    stream = ai.generate_stream(model='slow', prompt='hi')
    response = await settle(stream.response)

    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_prompt_stream_break_after_first_chunk_cancels_model_and_response_is_aborted() -> None:
    """`break` after the first chunk of `prompt.stream()` cancels the model; `.response` is ABORTED."""
    ai, model = setup()
    prompt = ai.define_prompt(model='slow', prompt='hi')

    stream = prompt.stream()
    seen = []
    async for chunk in stream:
        seen.append(chunk.text)
        break
    response = await settle(stream.response)

    assert seen == ['one ']
    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.text == ''


@pytest.mark.asyncio
async def test_prompt_stream_iterated_to_end_returns_completed_response() -> None:
    """Reading `prompt.stream()` to the end yields every chunk; `.response` is the finished reply."""
    ai, model = setup(release=True)
    prompt = ai.define_prompt(model='slow', prompt='hi')

    stream = prompt.stream()
    seen = [chunk.text async for chunk in stream]
    response = await settle(stream.response)

    assert seen == ['one ', 'two ']
    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_generate_stream_consumer_exception_cancels_model_and_response_is_aborted() -> None:
    """An unhandled exception in consumer loop body drops the iterator and cancels the model."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')

    with pytest.raises(RuntimeError, match='consumer crash'):
        async for _ in stream:
            raise RuntimeError('consumer crash')

    response = await settle(stream.response)

    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED


@pytest.mark.asyncio
async def test_generate_stream_timeout_cancels_model_and_response_is_aborted() -> None:
    """A `wait_for` deadline on the reader cancels it and cancels the model."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')

    async def read_slowly() -> None:
        async for _ in stream:
            await asyncio.sleep(1)

    # asyncio.TimeoutError, not TimeoutError: they're only the same class on 3.11+.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(read_slowly(), timeout=0.02)

    response = await settle(stream.response)

    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED


@pytest.mark.asyncio
async def test_generate_stream_break_on_turn_two_cancels_model_and_keeps_closed_tool_round() -> None:
    """Breaking on Turn 2 cancels the Turn 2 model call across hop() and preserves [USER, MODEL, TOOL]."""
    ai = Genkit()
    turn2_cancelled = asyncio.Event()
    model_calls = 0

    @ai.tool()
    async def check_pantry(item: str) -> str:
        return f'have {item}'

    async def chef(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('checking ')]))
            return ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part(tool_request=ToolRequest(name='check_pantry', ref='1', input='flour'))],
                ),
            )
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('baking ')]))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            turn2_cancelled.set()
            raise
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )

    ai.define_model(name='chef', fn=chef)

    stream = ai.generate_stream(model='chef', prompt='bake bread', tools=['check_pantry'])
    seen_texts: list[str] = []
    async for chunk in stream:
        seen_texts.append(chunk.text)
        if chunk.text == 'baking ':
            break
    response = await settle(stream.response)

    assert 'baking ' in seen_texts
    assert model_calls == 2
    assert turn2_cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED
    assert response.message is None
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]


@pytest.mark.asyncio
async def test_generate_stream_break_during_tool_cancels_tool_and_waits_for_finally_cleanup() -> None:
    """`break` while a tool runs cancels the tool, waits for `finally:` cleanup on `aclose()`, and boxes ABORTED."""
    ai = Genkit()
    tool_started = asyncio.Event()
    tool_cancelled = asyncio.Event()
    cleanup_finished = False

    @ai.tool()
    async def check_pantry(item: str) -> str:
        """Blocks until cancelled and performs async cleanup in finally."""
        nonlocal cleanup_finished
        tool_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            tool_cancelled.set()
            raise
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished = True
        return f'never:{item}'

    async def chef(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('checking ')]))
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='check_pantry', ref='1', input='flour'))],
            ),
        )

    ai.define_model(name='chef', fn=chef)

    stream = ai.generate_stream(model='chef', prompt='bake bread', tools=['check_pantry'])
    async with contextlib.aclosing(stream.stream) as chunks:
        async for _ in chunks:
            await settle(tool_started.wait())
            break

    assert tool_cancelled.is_set()
    assert cleanup_finished is True
    response = await settle(stream.response)
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    current = asyncio.current_task()
    assert current is not None
    if sys.version_info >= (3, 11):
        assert current.cancelling() == 0
