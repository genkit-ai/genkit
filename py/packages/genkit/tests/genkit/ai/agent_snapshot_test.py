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
from genkit._ai._agents._snapshot import is_heartbeat_expired, resolve_snapshot
from genkit._ai._agents._types import TurnResult
from genkit._ai._agents.session_stores.inmemory_store import InMemorySessionStore
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
    saved = await store.save_snapshot(None, lambda _: snap)
    assert saved is not None

    def redact(state: SessionState) -> SessionState:
        custom = state.custom if isinstance(state.custom, dict) else {}
        return state.model_copy(update={'custom': {'public': custom.get('public')}})

    result = await resolve_snapshot(store=store, snapshot_id=saved.snapshot_id, state_transform=redact)
    assert result is not None
    assert result.state is not None
    assert result.state.custom == {'public': 'ok'}
    assert 'secret' not in (result.state.custom or {})


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

    async def fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='hi'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'snapTest', fn, store=store)

    snapshot_action = registry._entries[ActionKind.AGENT_SNAPSHOT]['snapTest']  # noqa: SLF001
    abort_action = registry._entries[ActionKind.AGENT_ABORT]['snapTest']  # noqa: SLF001
    assert snapshot_action is not None
    assert abort_action is not None

    chat = agent.chat()
    turn = chat.send('hello')
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

    async def fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        return await session_runner.result()

    define_custom_agent(registry, 'missingSnapTest', fn, store=store)
    snapshot_action = registry._entries[ActionKind.AGENT_SNAPSHOT]['missingSnapTest']  # noqa: SLF001

    with pytest.raises(GenkitError) as exc:
        await snapshot_action.run({'snapshotId': 'non-existent-id'})
    assert exc.value.status == 'NOT_FOUND'
    assert 'non-existent-id' in str(exc.value)


@pytest.mark.asyncio
async def test_custom_agent_turn_that_raises_resolves_as_failed() -> None:
    """A turn that raises settles FAILED, keeps the resume handle on the last good
    parent, and rolls the optimistic prompt back instead of crashing the chat."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            text = input_text(inp)
            if 'fail' in text.lower():
                raise GenkitError(status='INTERNAL', message='boom')
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'flakyTest', fn, store=store)
    chat = agent.chat()

    out_ok = await chat.send('hello').response
    assert out_ok.finish_reason == AgentFinishReason.STOP
    last_good_parent = chat.snapshot_id
    history_before_failure = list(chat.messages)

    with pytest.raises(AgentError) as exc_info:
        await chat.send('please fail now').response
    assert exc_info.value.status == 'INTERNAL'
    assert exc_info.value.message == 'boom'
    # The failed turn is a dead end: the resume handle stays on the last good
    # parent and the unanswered prompt is dropped from the running view.
    assert exc_info.value.snapshot_id == last_good_parent
    assert chat.snapshot_id == last_good_parent
    assert chat.messages == history_before_failure


@pytest.mark.asyncio
async def test_chat_resumes_from_last_good_after_detached_turn_is_aborted() -> None:
    """Aborting a detached turn must not strand the chat: the next send() resumes
    from the last completed turn instead of the dead, aborted snapshot."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            text = input_text(inp)
            if 'slow' in text.lower():
                await asyncio.sleep(1.0)  # keep the turn pending long enough to abort it
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'detachAbortTest', fn, store=store)
    chat = agent.chat()

    await chat.send('hello').response
    last_good_parent = chat.snapshot_id
    history_before_detach = list(chat.messages)

    task = await chat.detach('slow background work')
    assert chat.messages != history_before_detach  # optimistic prompt pushed
    # The current snapshot is the in-flight detached one (so abort()/get_snapshot
    # act on it), but the resume handle stays on the last completed turn.
    assert chat.snapshot_id == task.snapshot_id
    assert chat._snapshot_id != chat._resume_snapshot_id  # noqa: SLF001

    status = await task.abort()
    assert status == SnapshotStatus.ABORTED
    # Aborting rolls back the optimistic prompt the chat was holding for the
    # killed turn, so the running view returns to the last completed turn.
    assert chat.messages == history_before_detach

    # The chat is still usable: the next turn branches off the last good parent
    # rather than the aborted snapshot, which is not resumable.
    out = await chat.send('are you there?').response
    assert out.finish_reason == AgentFinishReason.STOP
    assert chat._resume_snapshot_id == chat.snapshot_id  # noqa: SLF001
    assert chat.snapshot_id not in (None, last_good_parent, task.snapshot_id)


@pytest.mark.asyncio
async def test_load_chat_by_session_skips_aborted_leaf_to_last_resumable() -> None:
    """Reloading a session whose newest snapshot is an aborted detached turn lands
    on the last completed turn, not the dead leaf — so the reloaded chat resumes."""
    registry = Registry()
    store = InMemorySessionStore()

    async def fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            text = input_text(inp)
            if 'slow' in text.lower():
                await asyncio.sleep(1.0)
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'loadAfterAbortTest', fn, store=store)
    chat = agent.chat()
    await chat.send('hello').response
    session_id = chat.session_id
    last_good_parent = chat.snapshot_id

    task = await chat.detach('slow background work')
    assert await task.abort() == SnapshotStatus.ABORTED
    await asyncio.sleep(1.1)  # let the aborted background turn unwind

    reloaded = await agent.load_chat(session_id=session_id)
    # Landed on the last completed turn, not the aborted leaf.
    assert reloaded.snapshot_id == last_good_parent
    out = await reloaded.send('still there?').response
    assert out.finish_reason == AgentFinishReason.STOP
