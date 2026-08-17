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
from datetime import datetime, timedelta, timezone

import pytest

from genkit._ai._agents._base import define_custom_agent
from genkit._ai._agents._client import AgentError
from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore
from genkit._ai._agents._snapshot import is_heartbeat_expired, resolve_snapshot
from genkit._ai._agents._types import TurnContext, TurnResult
from genkit._core._action import ActionKind, ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._registry import Registry
from genkit._core._typing import (
    AgentInput,
    AgentResult,
    MessageData,
    Part,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
)
from genkit.agent import AgentFinishReason


def input_text(inp: AgentInput) -> str:
    """Concatenate the text parts of a turn's input message."""
    message = inp.message
    if message is None:
        return ''
    return ''.join(
        root.text
        for part in (message.content or [])
        if isinstance((root := getattr(part, 'root', part)), TextPart) and root.text
    )


@pytest.mark.asyncio
async def test_resolve_snapshot_applies_client_transform() -> None:
    store = InMemorySessionStore()

    snap = SessionSnapshot(
        snapshot_id='s1',
        session_id='sess',
        created_at=datetime.now(timezone.utc).isoformat(),
        status=SnapshotStatus.COMPLETED,
        state=SessionState(
            session_id='sess',
            custom={'public': 'ok', 'secret': 'hidden'},
        ),
    )
    saved = await store.save_snapshot(snap.snapshot_id, lambda _: snap)
    assert saved is not None

    def redact(state: SessionState) -> SessionState:
        custom = state.custom if isinstance(state.custom, dict) else {}
        return state.model_copy(update={'custom': {'public': custom.get('public')}})

    result = await resolve_snapshot(store=store, snapshot_id=saved.snapshot_id, state_transform=redact)
    assert result is not None
    assert result.state is not None
    assert result.state.custom == {'public': 'ok'}
    assert 'secret' not in (result.state.custom or {})


@pytest.mark.asyncio
async def test_resolve_snapshot_by_session_returns_failed_leaf() -> None:
    """Inspect by sessionId returns the stored leaf, even when it isn't resumable.

    A failed leaf has to stay visible so you can see why the last turn died.
    Resume walks back separately.
    """
    store = InMemorySessionStore()
    completed = SessionSnapshot(
        snapshot_id='snap-ok',
        session_id='sess',
        created_at='2026-01-01T00:00:00Z',
        status=SnapshotStatus.COMPLETED,
        state=SessionState(session_id='sess'),
    )
    failed = SessionSnapshot(
        snapshot_id='snap-fail',
        session_id='sess',
        parent_id='snap-ok',
        created_at='2026-01-01T00:01:00Z',
        status=SnapshotStatus.FAILED,
        state=SessionState(session_id='sess'),
    )
    await store.save_snapshot(completed.snapshot_id, lambda _: completed)
    await store.save_snapshot(failed.snapshot_id, lambda _: failed)

    inspected = await resolve_snapshot(store=store, session_id='sess')
    assert inspected is not None
    assert inspected.snapshot_id == 'snap-fail'
    assert inspected.status == SnapshotStatus.FAILED


def test_is_heartbeat_expired_pending_with_stale_heartbeat() -> None:
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    snap = SessionSnapshot(
        snapshot_id='s1',
        created_at=old,
        status=SnapshotStatus.PENDING,
        heartbeat_at=old,
        state=SessionState(session_id='sess'),
    )
    assert is_heartbeat_expired(snap)


@pytest.mark.asyncio
async def test_define_custom_agent_registers_snapshot_and_abort_actions() -> None:
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
            await session_runner.add_messages([MessageData(role='model', content=[Part(root=TextPart(text='hi'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'snapTest', fn, store=store)

    snapshot_action = registry._entries[ActionKind.AGENT_SNAPSHOT]['snapTest']  # noqa: SLF001
    abort_action = registry._entries[ActionKind.AGENT_ABORT]['snapTest']  # noqa: SLF001
    assert snapshot_action is not None
    assert abort_action is not None

    chat = agent.chat()
    turn = chat.send_stream('hello')
    async for _ in turn.stream:
        pass
    out = await turn.response
    assert out.snapshot_id

    via_method = await agent.get_snapshot_data(snapshot_id=out.snapshot_id)
    assert via_method is not None
    assert via_method.snapshot_id == out.snapshot_id

    via_action = await snapshot_action.run({'snapshotId': out.snapshot_id})
    assert via_action.response is not None
    assert via_action.response.snapshot_id == out.snapshot_id


@pytest.mark.asyncio
async def test_snapshot_action_raises_not_found_for_missing_snapshot() -> None:
    """A poll for a snapshot that isn't in the store surfaces NOT_FOUND, not a null."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        return await session_runner.result()

    define_custom_agent(registry, 'missingSnapTest', fn, store=store)
    snapshot_action = registry._entries[ActionKind.AGENT_SNAPSHOT]['missingSnapTest']  # noqa: SLF001

    with pytest.raises(GenkitError) as exc:
        await snapshot_action.run({'snapshotId': 'non-existent-id'})
    assert exc.value.status == 'NOT_FOUND'
    assert 'non-existent-id' in str(exc.value)


@pytest.mark.asyncio
async def test_get_snapshot_data_returns_none_for_missing_snapshot() -> None:
    """The method returns None on a miss; the action wraps that as NOT_FOUND."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        return await session_runner.result()

    agent = define_custom_agent(registry, 'missingMethodSnapTest', fn, store=store)
    assert await agent.get_snapshot_data(snapshot_id='non-existent-id') is None


@pytest.mark.asyncio
async def test_custom_agent_turn_that_raises_resolves_as_failed() -> None:
    """A turn that raises settles FAILED, keeps the resume handle on the last good
    parent, and rolls the optimistic prompt back instead of crashing the chat."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
            text = input_text(inp)
            if 'fail' in text.lower():
                raise GenkitError(status='INTERNAL', message='boom')
            await session_runner.add_messages([MessageData(role='model', content=[Part(root=TextPart(text='ok'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'flakyTest', fn, store=store)
    chat = agent.chat()

    out_ok = await chat.send('hello')
    assert out_ok.finish_reason == AgentFinishReason.STOP
    last_good_parent = chat.snapshot_id
    history_before_failure = list(chat.messages)

    with pytest.raises(AgentError) as exc_info:
        await chat.send('please fail now')
    assert exc_info.value.status == 'INTERNAL'
    assert exc_info.value.message == 'boom'
    # The failed turn is a dead end: the resume handle stays on the last good
    # parent and the unanswered prompt is dropped from the running view.
    assert exc_info.value.snapshot_id == last_good_parent
    assert chat.snapshot_id == last_good_parent
    assert chat.messages == history_before_failure


@pytest.mark.asyncio
async def test_chat_points_at_detached_snapshot_so_send_needs_completed_or_reload() -> None:
    """After detach the chat resumes the pending snapshot.

    A send while it is still pending (or after abort) is rejected; resume by
    session_id walks back to the last completed turn.
    """
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
            text = input_text(inp)
            if 'slow' in text.lower():
                await asyncio.sleep(1.0)  # keep the turn pending long enough to abort it
            await session_runner.add_messages([MessageData(role='model', content=[Part(root=TextPart(text='ok'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'detachAbortTest', fn, store=store)
    chat = agent.chat()

    await chat.send('hello')
    session_id = chat.session_id
    history_before_detach = list(chat.messages)

    task = await chat.detach('slow background work')
    assert chat.messages != history_before_detach  # optimistic prompt pushed
    # Resume handle tracks the pending detached snapshot.
    assert chat.snapshot_id == task.snapshot_id
    assert chat._resume_snapshot_id == task.snapshot_id  # noqa: SLF001

    with pytest.raises(AgentError, match='not resumable'):
        await chat.send('too soon')

    status = await task.abort()
    # abort() returns the previous status: pending while the turn was running.
    assert status == SnapshotStatus.PENDING
    # The prompt stays — it was still asked. The resume id still names the
    # aborted snapshot, so a bare send keeps failing until we reload.
    assert chat.messages != history_before_detach
    with pytest.raises(AgentError, match='not resumable'):
        await chat.send('still stranded')

    chat = agent.chat(session_id=session_id)
    out = await chat.send('are you there?')
    assert out.finish_reason == AgentFinishReason.STOP
    assert chat.snapshot_id not in (None, task.snapshot_id)


@pytest.mark.asyncio
async def test_load_chat_by_session_hydrates_aborted_leaf() -> None:
    """load_chat is inspect: a session whose newest snapshot is aborted lands
    on that leaf. send() is rejected; chat(session_id=) walks back to resume."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
            text = input_text(inp)
            if 'slow' in text.lower():
                await asyncio.sleep(1.0)
            await session_runner.add_messages([MessageData(role='model', content=[Part(root=TextPart(text='ok'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'loadAfterAbortTest', fn, store=store)
    chat = agent.chat()
    await chat.send('hello')
    session_id = chat.session_id

    task = await chat.detach('slow background work')
    assert await task.abort() == SnapshotStatus.PENDING  # previous status
    await asyncio.sleep(1.1)  # let the aborted background turn unwind

    inspected = await agent.get_snapshot(session_id=session_id)
    assert inspected is not None
    assert inspected.snapshot_id == task.snapshot_id
    assert inspected.status == SnapshotStatus.ABORTED

    reloaded = await agent.load_chat(session_id=session_id)
    assert reloaded.snapshot_id == task.snapshot_id
    with pytest.raises(AgentError, match='not resumable'):
        await reloaded.send('still there?')

    resumed = agent.chat(session_id=session_id)
    out = await resumed.send('still there?')
    assert out.finish_reason == AgentFinishReason.STOP
    assert resumed.snapshot_id not in (None, task.snapshot_id)
