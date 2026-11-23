# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Stopping `ai.generate_stream` / `prompt.stream` early stops the model call."""

import asyncio
import contextlib
import warnings
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
from genkit._core._compat import cancelling
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
    model_ctx: ModelContext = ModelContext()
    agent_ctx: AgentContext = AgentContext()
    mw_ctx = MiddlewareContext(ai=ai)

    for ctx_obj in (action_ctx, flow_ctx, tool_ctx, model_ctx, agent_ctx, mw_ctx):
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
async def test_generate_stream_async_with_break_cancels_model_and_response_is_aborted() -> None:
    """Exiting `async with ai.generate_stream(...):` early cancels the model; `.response` is ABORTED."""
    ai, model = setup()

    seen = []
    async with ai.generate_stream(model='slow', prompt='hi') as stream:
        async for chunk in stream:
            seen.append(chunk.text)
            break
    response = await settle(stream.response)

    assert seen == ['one ']
    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER]
    assert response.text == ''
    assert isinstance(stream.response, asyncio.Task)
    assert cancelling(stream.response) == 0


@pytest.mark.asyncio
async def test_generate_stream_bare_break_without_async_with_keeps_model_running() -> None:
    """Bare `break` without `async with` or `aclose()` leaves the background model task running to completion."""
    ai, model = setup(release=True)

    stream = ai.generate_stream(model='slow', prompt='hi')
    seen = []
    async for chunk in stream:
        seen.append(chunk.text)
        break
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    response = await settle(stream.response)

    assert seen == ['one ']
    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_generate_stream_consumer_task_cancelled_under_async_with_cancels_model_and_response_is_aborted() -> None:
    """Cancelling the task inside `async with ai.generate_stream(...):` cancels the model; `.response` is ABORTED."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')
    first_chunk = asyncio.Event()

    async def read() -> None:
        async with stream:
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
async def test_generate_stream_anext_aiter_keeps_model_running_on_both_handle_and_stream() -> None:
    """`aiter(s)` and `aiter(s.stream)` both return `s.stream`, so `anext(aiter(...))` never cancels."""
    ai, model = setup(release=True)

    s = ai.generate_stream(model='slow', prompt='hi')
    assert aiter(s) is s.stream
    assert aiter(s.stream) is s.stream
    first = await settle(anext(aiter(s)))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    rest = [chunk.text async for chunk in s]
    response = await settle(s.response)

    assert first.text == 'one '
    assert rest == ['two ']
    assert not model.cancelled.is_set()
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'one two'


@pytest.mark.asyncio
async def test_generate_stream_async_with_read_to_end_completes_normally() -> None:
    """Exiting `async with ai.generate_stream(...)` after reading to the end preserves FinishReason.STOP."""
    ai, model = setup(release=True)

    async with ai.generate_stream(model='slow', prompt='hi') as s:
        seen = [chunk.text async for chunk in s]
    response = await settle(s.response)

    assert seen == ['one ', 'two ']
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
async def test_prompt_stream_async_with_break_cancels_model_and_response_is_aborted() -> None:
    """Exiting `async with prompt.stream():` early cancels the model; `.response` is ABORTED."""
    ai, model = setup()
    prompt = ai.define_prompt(model='slow', prompt='hi')

    seen = []
    async with prompt.stream() as stream:
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
async def test_generate_stream_consumer_exception_under_async_with_cancels_model_and_response_is_aborted() -> None:
    """An unhandled exception inside `async with ai.generate_stream(...):` cancels the model."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')

    with pytest.raises(RuntimeError, match='consumer crash'):
        async with stream:
            async for _ in stream:
                raise RuntimeError('consumer crash')

    response = await settle(stream.response)

    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED


@pytest.mark.asyncio
async def test_generate_stream_timeout_under_async_with_cancels_model_and_response_is_aborted() -> None:
    """A `wait_for` deadline on a task inside `async with ai.generate_stream(...):` cancels the model."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')

    async def read_slowly() -> None:
        async with stream:
            async for _ in stream:
                await asyncio.sleep(1)

    # asyncio.TimeoutError, not TimeoutError: they're only the same class on 3.11+.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(read_slowly(), timeout=0.02)

    response = await settle(stream.response)

    assert model.cancelled.is_set()
    assert response.finish_reason == FinishReason.ABORTED


@pytest.mark.asyncio
async def test_generate_stream_async_with_break_on_turn_two_cancels_model_and_keeps_closed_tool_round() -> None:
    """Exiting `async with` on Turn 2 cancels the Turn 2 model call across hop() and preserves [USER, MODEL, TOOL]."""
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

    seen_texts: list[str] = []
    async with ai.generate_stream(model='chef', prompt='bake bread', tools=['check_pantry']) as stream:
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
    assert isinstance(stream.response, asyncio.Task)
    assert cancelling(stream.response) == 0


@pytest.mark.asyncio
async def test_generate_stream_anext_timeout_does_not_cancel_model() -> None:
    """Cancelling a single `stream.stream.__anext__()` (SSE heartbeat tick) leaves the model call running."""
    ai, model = setup()
    stream = ai.generate_stream(model='slow', prompt='hi')

    first = await settle(stream.stream.__anext__())
    assert first.text == 'one '
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(stream.stream.__anext__(), timeout=0.01)

    assert not model.cancelled.is_set()
    model.release.set()
    second = await settle(stream.stream.__anext__())
    assert second.text == 'two '
    response = await settle(stream.response)
    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'one two'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['aclosing', 'async_with'])
async def test_generate_stream_aclosing_and_async_with_preserve_awaiting_model_finally_cleanup(mode: str) -> None:
    """Breaking under `aclosing(stream.stream)` or `async with ai.generate_stream(...)` lets model `finally:` finish."""
    ai = Genkit()
    cleanup_finished = False

    async def slow_model(_request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        nonlocal cleanup_finished
        try:
            ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('one ')]))
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.02)
            cleanup_finished = True
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('never')]),
        )

    ai.define_model(name='slow_model', fn=slow_model)

    stream = ai.generate_stream(model='slow_model', prompt='hi')
    if mode == 'aclosing':
        async with contextlib.aclosing(stream.stream) as chunks:
            async for _ in chunks:
                break
    else:
        async with stream as chunks:
            async for _ in chunks:
                break

    assert cleanup_finished is True
    response = await settle(stream.response)
    assert response.finish_reason == FinishReason.ABORTED
    assert isinstance(stream.response, asyncio.Task)
    assert cancelling(stream.response) == 0


@pytest.mark.asyncio
async def test_generate_cancel_same_tick_as_child_failure_preserves_cancelled_error() -> None:
    """When `hop()`'s child fails in the same tick the parent task is cancelled, `CancelledError` wins over retry."""
    ai = Genkit()
    calls = 0
    gen_task: asyncio.Task[ModelResponse] | None = None

    class RetryOnceMiddleware(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            except ValueError:
                return await next_fn(params, ctx)

    async def races_cancel(_request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0)
            assert gen_task is not None
            # Schedule parent cancellation in the exact tick child_task finishes with ValueError.
            gen_task.cancel()
            raise ValueError('transient failure')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('retried ok')]),
        )

    ai.define_model(name='races_cancel', fn=races_cancel)

    gen_task = asyncio.create_task(
        ai.generate(model='races_cancel', prompt='hi', use=[RetryOnceMiddleware()]),
    )
    with pytest.raises(asyncio.CancelledError):
        await settle(gen_task)

    assert calls == 1


@pytest.mark.asyncio
async def test_generate_cancel_same_tick_as_child_failure_across_stacked_middleware_preserves_cancelled_error() -> None:
    """Same-tick parent cancel + model error across stacked middleware hops unwinds as `CancelledError` on 3.10+."""
    ai = Genkit()
    calls = 0
    gen_task: asyncio.Task[ModelResponse] | None = None

    class PassThroughMiddleware(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(params, ctx)

    class RetryOnceMiddleware(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            try:
                return await next_fn(params, ctx)
            except ValueError:
                return await next_fn(params, ctx)

    async def races_cancel(_request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0)
            assert gen_task is not None
            gen_task.cancel()
            raise ValueError('transient failure')
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('retried ok')]),
        )

    ai.define_model(name='races_stacked_cancel', fn=races_cancel)

    gen_task = asyncio.create_task(
        ai.generate(
            model='races_stacked_cancel',
            prompt='hi',
            use=[
                PassThroughMiddleware(),
                PassThroughMiddleware(),
                PassThroughMiddleware(),
                RetryOnceMiddleware(),
                PassThroughMiddleware(),
            ],
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        await settle(gen_task)

    assert calls == 1


@pytest.mark.asyncio
async def test_generate_cancel_before_hop_body_enters_emits_no_runtime_warning() -> None:
    """Cancelling `ai.generate(...)` before `hop()` enters `body` closes the coroutine without `RuntimeWarning`."""
    ai, model = setup(release=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        task = asyncio.create_task(ai.generate(model='slow', prompt='hi'))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await settle(task)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert model.calls <= 1
