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

from __future__ import annotations

import asyncio

import pytest

from genkit._ai._agents._base import AgentRuntime, SessionRunner, _agent_input_has_payload
from genkit._ai._agents._session import InMemorySessionStore, Session
from genkit._ai._aio import Genkit
from genkit._ai._generate import generate_action
from genkit._ai._testing import define_programmable_model
from genkit._ai._tools import ToolRunContext
from genkit._core._action import ActionRunContext
from genkit._core._channel import CloseableQueue
from genkit._core._error import GenkitError
from genkit._core._model import GenerateActionOptions, Message, ModelResponse
from genkit._core._typing import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    MessageData,
    ModelResponseChunk,
    Part,
    Role,
    SessionState,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)


async def _wait_for_snapshot_status(
    store: InMemorySessionStore,
    snapshot_id: str,
    status: SnapshotStatus,
    *,
    timeout_s: float = 3.0,
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        snap = await store.get_snapshot(snapshot_id=snapshot_id)
        if snap is not None and snap.status == status:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f'snapshot {snapshot_id!r} never reached status {status!r}')


def _runtime(session: Session, store: InMemorySessionStore | None) -> tuple[AgentRuntime, CloseableQueue]:
    out_queue = CloseableQueue()
    rt = AgentRuntime(
        name='detachAudit',
        session=session,
        parent_snapshot=None,
        store=store,
        snapshot_callback=None,
        client_transform=None,
        session_outputs=out_queue,
    )
    return rt, out_queue


_NO_ABORT = asyncio.Event()


@pytest.mark.asyncio
async def test_agent_input_has_payload() -> None:
    assert _agent_input_has_payload(
        AgentInput(message=MessageData(role=Role.USER, content=[Part(TextPart(text='x'))]), detach=True),
    )
    assert not _agent_input_has_payload(AgentInput(detach=True))


@pytest.mark.asyncio
async def test_detach_forwards_message_payload_in_same_input() -> None:
    store = InMemorySessionStore()
    session = Session(SessionState(messages=[]))
    rt, _ = _runtime(session, store)
    await rt.sess.seed_last_good_state()

    seen_inputs: list[AgentInput] = []

    async def agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> None:
            seen_inputs.append(inp)
            return None

        await sess.run(handle_turn)
        return await sess.result()

    in_queue = CloseableQueue()
    await in_queue.put(
        AgentInput(
            message=MessageData(role=Role.USER, content=[Part(TextPart(text='appended message'))]),
            detach=True,
        )
    )
    in_queue.close()

    out = await rt.run(agent_fn, in_queue)

    assert out.finish_reason == AgentFinishReason.DETACHED
    assert out.snapshot_id is not None
    assert len(seen_inputs) == 1
    assert seen_inputs[0].message is not None
    assert seen_inputs[0].message.content[0].root.text == 'appended message'

    msgs = await session.get_messages()
    assert len(msgs) == 1
    assert msgs[0].content[0].root.text == 'appended message'

    await _wait_for_snapshot_status(store, out.snapshot_id, SnapshotStatus.COMPLETED)
    snap = await store.get_snapshot(snapshot_id=out.snapshot_id)
    assert snap is not None
    assert snap.state is not None
    assert snap.state.messages is not None
    assert len(snap.state.messages) == 1


@pytest.mark.asyncio
async def test_detach_mid_turn_finalizes_snapshot_when_work_completes() -> None:
    store = InMemorySessionStore()
    session = Session(SessionState(messages=[]))
    rt, out_queue = _runtime(session, store)
    await rt.sess.seed_last_good_state()

    release = asyncio.Event()
    chunks: list[AgentStreamChunk] = []

    async def agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput) -> None:
            ctx.send_chunk(
                AgentStreamChunk(
                    model_chunk=ModelResponseChunk(role=Role.MODEL, content=[Part(TextPart(text='working'))])
                )
            )
            await release.wait()

        await sess.run(handle_turn)
        return await sess.result()

    in_queue = CloseableQueue()
    await in_queue.put(AgentInput(message=MessageData(role=Role.USER, content=[Part(TextPart(text='slow'))])))
    await in_queue.put(AgentInput(detach=True))
    in_queue.close()

    out = await rt.run(agent_fn, in_queue)
    assert out.finish_reason == AgentFinishReason.DETACHED
    assert out.snapshot_id is not None

    snap_pending = await store.get_snapshot(snapshot_id=out.snapshot_id)
    assert snap_pending is not None
    assert snap_pending.status == SnapshotStatus.PENDING

    while not out_queue.empty():
        chunks.append(out_queue.get_nowait())

    release.set()
    await _wait_for_snapshot_status(store, out.snapshot_id, SnapshotStatus.COMPLETED)

    snap_done = await store.get_snapshot(snapshot_id=out.snapshot_id)
    assert snap_done is not None
    assert snap_done.finish_reason is None or snap_done.status == SnapshotStatus.COMPLETED
    assert snap_done.state is not None
    assert snap_done.state.messages is not None
    assert len(snap_done.state.messages) == 1

    # No chunks after detach (wire quiet).
    await asyncio.sleep(0.05)
    while not out_queue.empty():
        chunks.append(out_queue.get_nowait())
    assert all(c.turn_end is None for c in chunks)


@pytest.mark.asyncio
async def test_detach_without_store_raises() -> None:
    session = Session(SessionState(messages=[]))
    rt, _ = _runtime(session, None)
    await rt.sess.seed_last_good_state()

    async def agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput) -> None:
            await ctx.abort_signal.wait()

        await sess.run(handle_turn)
        return await sess.result()

    in_queue = CloseableQueue()
    await in_queue.put(AgentInput(message=MessageData(role=Role.USER, content=[Part(TextPart(text='x'))])))
    await in_queue.put(AgentInput(detach=True))
    in_queue.close()

    with pytest.raises(ValueError, match='detach requires a session store'):
        await rt.run(agent_fn, in_queue)


@pytest.mark.asyncio
async def test_abort_snapshot_stops_detached_work() -> None:
    store = InMemorySessionStore()
    session = Session(SessionState(messages=[]))
    rt, _ = _runtime(session, store)
    await rt.sess.seed_last_good_state()

    aborted = asyncio.Event()

    async def agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(_inp: AgentInput) -> None:
            for _ in range(100):
                if ctx.abort_signal.is_set():
                    aborted.set()
                    return
                await asyncio.sleep(0.02)

        await sess.run(handle_turn)
        return await sess.result()

    in_queue = CloseableQueue()
    await in_queue.put(AgentInput(message=MessageData(role=Role.USER, content=[Part(TextPart(text='long'))])))
    await in_queue.put(AgentInput(detach=True))
    in_queue.close()

    out = await rt.run(agent_fn, in_queue)
    assert out.snapshot_id is not None

    prev = await store.abort_snapshot(out.snapshot_id)
    assert prev == SnapshotStatus.ABORTED

    await _wait_for_snapshot_status(store, out.snapshot_id, SnapshotStatus.ABORTED, timeout_s=2.0)
    await asyncio.wait_for(aborted.wait(), timeout=2.0)

    snap = await store.get_snapshot(snapshot_id=out.snapshot_id)
    assert snap is not None
    assert snap.status == SnapshotStatus.ABORTED


@pytest.mark.asyncio
async def test_generate_tool_respects_abort_signal() -> None:
    """Tools invoked during generate see the same abort_signal as the agent runtime."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    abort_signal = asyncio.Event()
    tool_saw_abort = asyncio.Event()

    @ai.tool(name='slowWork')
    async def slow_work(_: dict, ctx: ToolRunContext) -> dict:
        try:
            for _i in range(200):
                if ctx.abort_signal.is_set():
                    tool_saw_abort.set()
                    raise GenkitError(status='ABORTED', message='Task aborted')
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            if ctx.abort_signal.is_set():
                tool_saw_abort.set()
            raise
        return {'done': True}

    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='slowWork', input={}, ref='r1')))],
            ),
        )
    )

    async def run_generate() -> None:
        with pytest.raises(GenkitError) as exc_info:
            await generate_action(
                ai.registry,
                GenerateActionOptions(
                    model='programmableModel',
                    messages=[Message(role=Role.USER, content=[Part(TextPart(text='go'))])],
                    tools=['slowWork'],
                ),
                abort_signal=abort_signal,
            )
        assert exc_info.value.status == 'ABORTED'

    task = asyncio.create_task(run_generate())
    await asyncio.sleep(0.05)
    abort_signal.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert tool_saw_abort.is_set()
