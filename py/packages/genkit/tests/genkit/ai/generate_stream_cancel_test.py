# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Stopping `ai.generate_stream` / `prompt.stream` early stops the model call."""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from genkit import ActionRunContext, FinishReason, Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit._core._model import ModelRequest

T = TypeVar('T')


class SlowModel:
    """Sends 'one ', then waits for `release` (ignoring the abort signal), then sends 'two '."""

    def __init__(self, *, release: bool = False, fail_midway: bool = False) -> None:
        self.release = asyncio.Event()
        if release:
            self.release.set()
        self.fail_midway = fail_midway
        self.cancelled = asyncio.Event()
        self.abort_set_when_cancelled: bool | None = None

    async def __call__(self, request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text('one ')]))
        if self.fail_midway:
            raise ValueError('model blew up')
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.abort_set_when_cancelled = ctx.abort_signal.is_set()
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

    async def slow(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        return await model(request, ctx)

    ai.define_model(name='slow', fn=slow)
    return ai, model


async def settle(aw: Awaitable[T]) -> T:
    # guards against a hang turning into a stuck test run; never the thing that passes a test.
    return await asyncio.wait_for(aw, timeout=5)


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
async def test_generate_stream_model_sees_abort_signal_set_when_caller_breaks() -> None:
    """After a `break`, the model's `ctx.abort_signal` is already set when its await is cancelled."""
    ai, model = setup()

    stream = ai.generate_stream(model='slow', prompt='hi')
    async for _ in stream:
        break
    await settle(stream.response)

    assert model.abort_set_when_cancelled is True


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
    assert model.abort_set_when_cancelled is True
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
