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

"""Pre-turn init failures: AgentInitError throws; recoverable becomes finish_reason failed."""

from __future__ import annotations

import pytest

from genkit._ai._agents._base import define_custom_agent
from genkit._ai._agents._client import AgentError
from genkit._ai._agents._runtime import AgentInitError, SessionRunner
from genkit._core._action import ActionRunContext
from genkit._core._registry import Registry
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentResult,
    MessageData,
    Part,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
)
from genkit.agent import InMemorySessionStore, TurnContext, TurnResult


async def echo_fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
        text = ''
        if inp.message and inp.message.content:
            root = inp.message.content[0].root
            text = getattr(root, 'text', '') or ''
        await session_runner.add_messages([
            MessageData(role='model', content=[Part(root=TextPart(text=f'Echo: {text}'))])
        ])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await session_runner.run(handle_turn)
    return await session_runner.result()


@pytest.mark.asyncio
async def test_missing_snapshot_resolves_as_failed_agent_output() -> None:
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'missingSnap', echo_fn, store=store)

    conn = await agent.stream_bidi(AgentInit(snapshot_id='does-not-exist'))
    out = await conn.output()

    assert out.finish_reason == AgentFinishReason.FAILED
    assert out.error is not None
    assert out.error.status == 'NOT_FOUND'
    assert 'does-not-exist' in (out.error.message or '')
    # Recoverable pre-turn failure must not write a snapshot.
    assert await store.get_snapshot(snapshot_id='does-not-exist') is None


@pytest.mark.asyncio
async def test_non_resumable_snapshot_resolves_as_failed_agent_output() -> None:
    registry = Registry()
    store = InMemorySessionStore()
    failed = SessionSnapshot(
        snapshot_id='snap-failed',
        session_id='sess-1',
        created_at='2026-06-18T12:00:00Z',
        status=SnapshotStatus.FAILED,
        state=SessionState(session_id='sess-1', messages=[], artifacts=[]),
    )
    saved = await store.save_snapshot(failed.snapshot_id, lambda existing: failed)
    assert saved is not None

    agent = define_custom_agent(registry, 'badStatus', echo_fn, store=store)
    conn = await agent.stream_bidi(AgentInit(snapshot_id=saved.snapshot_id))
    out = await conn.output()

    assert out.finish_reason == AgentFinishReason.FAILED
    assert out.error is not None
    assert out.error.status == 'INVALID_ARGUMENT'
    assert 'not resumable' in (out.error.message or '')


@pytest.mark.asyncio
async def test_state_on_server_managed_agent_raises_agent_init_error() -> None:
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'serverOnly', echo_fn, store=store)

    conn = await agent.stream_bidi(AgentInit(state=SessionState(custom={'x': 1})))
    with pytest.raises(AgentInitError) as exc:
        await conn.output()

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert "Cannot send 'state'" in str(exc.value)


def test_chat_rejects_state_on_server_managed_agent() -> None:
    """App-facing chat() refuses a state seed the same way the wire path does."""
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'serverChatSeed', echo_fn, store=store)

    with pytest.raises(AgentInitError) as exc:
        agent.chat(state={'x': 1})

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert "Cannot send 'state'" in str(exc.value)


def test_chat_rejects_messages_on_server_managed_agent() -> None:
    """Bundled messages seed must be named as 'messages', not blamed as 'state'."""
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'serverChatMessages', echo_fn, store=store)

    with pytest.raises(AgentInitError) as exc:
        agent.chat(messages=[MessageData(role='user', content=[Part(root=TextPart(text='hi'))])])

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert "Cannot send 'messages'" in str(exc.value)
    assert "Cannot send 'state'" not in str(exc.value)


def test_chat_rejects_messages_mixed_with_snapshot_id() -> None:
    """messages= + snapshot_id= is AgentInitError naming 'messages', not 'state'."""
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'serverChatMix', echo_fn, store=store)

    with pytest.raises(AgentInitError) as exc:
        agent.chat(
            messages=[MessageData(role='user', content=[Part(root=TextPart(text='hi'))])],
            snapshot_id='snap-1',
        )

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert "Cannot send 'messages'" in str(exc.value)
    assert "Cannot send 'state'" not in str(exc.value)


@pytest.mark.asyncio
async def test_snapshot_id_on_client_managed_agent_raises_agent_init_error() -> None:
    registry = Registry()
    agent = define_custom_agent(registry, 'clientOnly', echo_fn, store=None)

    conn = await agent.stream_bidi(AgentInit(snapshot_id='snap-1'))
    with pytest.raises(AgentInitError) as exc:
        await conn.output()

    assert exc.value.status == 'FAILED_PRECONDITION'
    assert 'no store configured' in str(exc.value)


@pytest.mark.asyncio
async def test_chat_surfaces_missing_snapshot_as_agent_error() -> None:
    """App-facing chat API wraps the failed AgentOutput into AgentError."""
    registry = Registry()
    store = InMemorySessionStore()
    agent = define_custom_agent(registry, 'missingSnapChat', echo_fn, store=store)

    with pytest.raises(AgentError) as exc:
        await agent.chat(snapshot_id='gone').send('hi')

    assert exc.value.status == 'NOT_FOUND'
