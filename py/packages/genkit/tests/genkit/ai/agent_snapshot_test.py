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

from datetime import datetime, timedelta, timezone

import pytest

from genkit._ai._agents._base import define_custom_agent
from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session_stores._latest_state import InMemoryLatestStateStore
from genkit._ai._agents._snapshot import is_heartbeat_expired, resolve_snapshot
from genkit._ai._agents._types import TurnResult
from genkit._core._action import ActionKind, ActionRunContext
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


@pytest.mark.asyncio
async def test_resolve_snapshot_applies_client_transform() -> None:
    store = InMemoryLatestStateStore()

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

    result = await resolve_snapshot(store, snapshot_id=saved.snapshot_id, client_transform={'state': redact})
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
    store = InMemoryLatestStateStore()

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
    async for _ in turn:
        pass
    out = await turn
    assert out.snapshot_id

    via_method = await agent.get_snapshot_data(snapshot_id=out.snapshot_id)
    assert via_method is not None
    assert via_method.snapshot_id == out.snapshot_id

    via_action = await snapshot_action.run({'snapshotId': out.snapshot_id})
    assert via_action.response is not None
    assert via_action.response.snapshot_id == out.snapshot_id
