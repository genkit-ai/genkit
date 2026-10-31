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

"""Tests for async cleanup and exception retrieval when tool execution is cancelled."""

from __future__ import annotations

import asyncio
import contextlib
import gc
from collections.abc import AsyncIterator
from typing import Any

import pytest

from genkit import FinishReason, Genkit, GenkitError, Message, ModelResponse, Part, Role
from genkit._ai._generate import execute_tool_request, resolve_tool
from genkit._ai._tools import Interrupt
from genkit._core._middleware import MiddlewareContext
from genkit.model import ToolRequest
from genkit.testing import define_scripted_model


def _model_calls_tool(*, name: str, ref: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=name, input={}, ref=ref))],
        ),
    )


@pytest.mark.asyncio
async def test_execute_tool_request_cancel_awaits_async_context_manager_cleanup() -> None:
    """Cancelling execute_tool_request awaits the tool's async context manager teardown before unwinding."""
    ai = Genkit()
    lease_events: list[str] = []
    acquired = asyncio.Event()

    @contextlib.asynccontextmanager
    async def inventory_lease() -> AsyncIterator[None]:
        lease_events.append('acquired')
        acquired.set()
        try:
            yield
        finally:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            lease_events.append('released')

    @ai.tool(name='hold_inventory')
    async def hold_inventory() -> str:
        async with inventory_lease():
            await asyncio.sleep(10)
        return 'held'

    action = await resolve_tool(ai._registry, 'hold_inventory')
    trp = Part.from_tool_request(name='hold_inventory', ref='r1', input={})
    ctx = MiddlewareContext(ai)

    task = asyncio.create_task(execute_tool_request(tool=action, tool_request_part=trp, ctx=ctx))
    await acquired.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert lease_events == ['acquired', 'released']
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.asyncio
async def test_generate_cancel_awaits_tool_async_finally_cleanup_before_unwinding() -> None:
    """Cancelling ai.generate mid-tool waits for the tool's async finally block to complete before returning."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    seat_locked = asyncio.Event()
    seat_released = False

    @contextlib.asynccontextmanager
    async def seat_hold() -> AsyncIterator[None]:
        nonlocal seat_released
        seat_locked.set()
        try:
            yield
        finally:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            seat_released = True

    @ai.tool(name='lock_seat')
    async def lock_seat() -> str:
        async with seat_hold():
            await asyncio.sleep(10)
        return '12A'

    pm.responses = [_model_calls_tool(name='lock_seat', ref='r1')]

    gen_task = asyncio.create_task(ai.generate(prompt='Reserve seat 12A', tools=['lock_seat']))
    await seat_locked.wait()
    gen_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await gen_task

    assert seat_released is True


@pytest.mark.asyncio
async def test_execute_tool_request_cancel_retrieves_teardown_exception_without_loop_error() -> None:
    """An exception raised inside a cancelled tool's finally block is retrieved and never leaked to the event loop."""
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        ai = Genkit()
        started = asyncio.Event()
        teardown_ran = False

        @ai.tool(name='void_authorization')
        async def void_authorization() -> str:
            nonlocal teardown_ran
            started.set()
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(0)
                teardown_ran = True
                raise RuntimeError('gateway socket closed during teardown')
            return 'voided'

        action = await resolve_tool(ai._registry, 'void_authorization')
        trp = Part.from_tool_request(name='void_authorization', ref='r1', input={})
        ctx = MiddlewareContext(ai)

        task: asyncio.Task[Any] | None = asyncio.create_task(
            execute_tool_request(tool=action, tool_request_part=trp, ctx=ctx)
        )
        await started.wait()
        assert task is not None
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert teardown_ran is True
        assert asyncio.all_tasks() == {asyncio.current_task()}

        task = None
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert reported == []


@pytest.mark.asyncio
async def test_generate_cancel_with_tool_raising_in_finally_leaves_zero_unretrieved_task_exceptions() -> None:
    """Cancelling ai.generate when a tool raises in finally surfaces CancelledError and leaks zero task exceptions."""
    loop = asyncio.get_running_loop()
    reported: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reported.append(context))
    try:
        ai = Genkit(model='scriptedModel')
        pm, _ = define_scripted_model(ai)
        escrow_opened = asyncio.Event()
        escrow_cleanup_attempted = False

        @ai.tool(name='release_escrow')
        async def release_escrow() -> str:
            nonlocal escrow_cleanup_attempted
            escrow_opened.set()
            try:
                await asyncio.sleep(10)
            finally:
                await asyncio.sleep(0)
                escrow_cleanup_attempted = True
                raise RuntimeError('escrow connection reset during rollback')
            return 'released'

        pm.responses = [_model_calls_tool(name='release_escrow', ref='r1')]

        gen_task: asyncio.Task[Any] | None = asyncio.create_task(
            ai.generate(prompt='Cancel order #42', tools=['release_escrow'])
        )
        await escrow_opened.wait()
        assert gen_task is not None
        gen_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await gen_task

        assert escrow_cleanup_attempted is True

        gen_task = None
        gc.collect()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert reported == []


@pytest.mark.asyncio
async def test_execute_tool_request_preserves_non_cancel_errors_and_interrupts_when_abort_signal_set() -> None:
    """A normal Exception or Interrupt from the tool body is not rewritten into Task aborted if abort_signal is set."""
    ai = Genkit()
    abort_signal = asyncio.Event()

    @ai.tool(name='charge_card')
    async def charge_card() -> str:
        abort_signal.set()
        raise ValueError('card declined')

    abort_signal_intr = asyncio.Event()

    @ai.tool(name='wire_funds')
    async def wire_funds() -> str:
        abort_signal_intr.set()
        raise Interrupt({'confirm': 'wire_transfer'})

    charge_action = await resolve_tool(ai._registry, 'charge_card')
    charge_trp = Part.from_tool_request(name='charge_card', ref='r1', input={})
    ctx = MiddlewareContext(ai, abort_signal=abort_signal)

    with pytest.raises(ValueError, match='card declined'):
        await execute_tool_request(tool=charge_action, tool_request_part=charge_trp, ctx=ctx)

    wire_action = await resolve_tool(ai._registry, 'wire_funds')
    wire_trp = Part.from_tool_request(name='wire_funds', ref='r2', input={})
    intr_ctx = MiddlewareContext(ai, abort_signal=abort_signal_intr)

    with pytest.raises(Interrupt) as intr_info:
        await execute_tool_request(tool=wire_action, tool_request_part=wire_trp, ctx=intr_ctx)
    assert intr_info.value.metadata == {'confirm': 'wire_transfer'}

    abort_live = asyncio.Event()
    hold_started = asyncio.Event()
    hold_cleaned = False

    @ai.tool(name='hold_funds')
    async def hold_funds() -> str:
        nonlocal hold_cleaned
        hold_started.set()
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            hold_cleaned = True
        return 'held'

    hold_action = await resolve_tool(ai._registry, 'hold_funds')
    hold_trp = Part.from_tool_request(name='hold_funds', ref='r3', input={})
    hold_ctx = MiddlewareContext(ai, abort_signal=abort_live)

    abort_task = asyncio.create_task(execute_tool_request(tool=hold_action, tool_request_part=hold_trp, ctx=hold_ctx))
    await hold_started.wait()
    abort_live.set()

    with pytest.raises(GenkitError) as abort_err:
        await abort_task
    assert abort_err.value.status == 'CANCELLED'
    assert abort_err.value.message == 'Task aborted'
    assert hold_cleaned is True
    assert asyncio.all_tasks() == {asyncio.current_task()}


@pytest.mark.asyncio
async def test_execute_tool_request_happy_path_awaits_cancelled_watcher_task() -> None:
    """Happy-path tool execution cancels and awaits watcher_task before returning."""
    ai = Genkit()

    @ai.tool(name='fetch_menu')
    async def fetch_menu() -> str:
        return 'Smoked Salmon Tartine'

    action = await resolve_tool(ai._registry, 'fetch_menu')
    trp = Part.from_tool_request(name='fetch_menu', ref='r1', input={})
    ctx = MiddlewareContext(ai)

    result = await execute_tool_request(tool=action, tool_request_part=trp, ctx=ctx)

    assert result.output == 'Smoked Salmon Tartine'
    assert asyncio.all_tasks() == {asyncio.current_task()}
