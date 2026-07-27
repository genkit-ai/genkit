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

"""Reserved snapshot ids + TurnContext: handler knows the id before the turn ends."""

from __future__ import annotations

from pathlib import Path

import pytest

from genkit._ai._agents._base import define_custom_agent
from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session import reserve_snapshot_id
from genkit._ai._agents._types import TurnContext, TurnResult
from genkit._core._action import ActionRunContext
from genkit._core._registry import Registry
from genkit._core._typing import (
    AgentInput,
    AgentResult,
    MessageData,
    Part,
    TextPart,
)
from genkit.agent import AgentFinishReason, InMemorySessionStore


def test_reserve_snapshot_id_is_unique_uuid() -> None:
    a = reserve_snapshot_id()
    b = reserve_snapshot_id()
    assert a != b
    assert len(a) == 36


@pytest.mark.asyncio
async def test_handler_receives_reserved_id_reused_on_persisted_snapshot() -> None:
    registry = Registry()
    store = InMemorySessionStore()
    seen: dict[str, object] = {}

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(_: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
            seen['snapshot_id'] = turn_ctx.snapshot_id
            seen['parent_snapshot_id'] = turn_ctx.parent_snapshot_id
            seen['turn_index'] = turn_ctx.turn_index
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'reserveTest', fn, store=store)
    out = await agent.chat().send('hi').response

    assert seen['snapshot_id']
    assert seen['parent_snapshot_id'] is None
    assert seen['turn_index'] == 0
    assert out.snapshot_id == seen['snapshot_id']
    saved = await store.get_snapshot(snapshot_id=str(seen['snapshot_id']))
    assert saved is not None
    assert saved.snapshot_id == seen['snapshot_id']


@pytest.mark.asyncio
async def test_second_turn_parent_is_first_turn_snapshot() -> None:
    registry = Registry()
    store = InMemorySessionStore()
    snapshot_ids: list[str] = []
    parent_ids: list[str | None] = []

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(_: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
            assert turn_ctx.snapshot_id is not None
            snapshot_ids.append(turn_ctx.snapshot_id)
            parent_ids.append(turn_ctx.parent_snapshot_id)
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'parentTest', fn, store=store)
    chat = agent.chat()
    await chat.send('one').response
    await chat.send('two').response

    assert len(snapshot_ids) == 2
    assert parent_ids[0] is None
    assert parent_ids[1] == snapshot_ids[0]


@pytest.mark.asyncio
async def test_no_store_means_no_reserved_snapshot_id() -> None:
    registry = Registry()
    seen: dict[str, object] = {'snapshot_id': 'sentinel'}

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(_: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
            seen['snapshot_id'] = turn_ctx.snapshot_id
            await session_runner.add_messages(MessageData(role='model', content=[Part(root=TextPart(text='ok'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'clientManaged', fn, store=None)
    await agent.chat().send('hi').response
    assert seen['snapshot_id'] is None


@pytest.mark.asyncio
async def test_handler_can_name_external_dir_after_reserved_id(tmp_path: Path) -> None:
    """The product reason for reserved ids: bind external resources before save."""
    registry = Registry()
    store = InMemorySessionStore()
    workspace_root = tmp_path / 'workspaces'

    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(_: AgentInput, turn_ctx: TurnContext) -> TurnResult | None:
            assert turn_ctx.snapshot_id is not None
            work = workspace_root / turn_ctx.snapshot_id
            work.mkdir(parents=True)
            (work / 'notes.txt').write_text('drafted during the turn\n', encoding='utf-8')
            await session_runner.add_messages(
                MessageData(
                    role='model',
                    content=[Part(root=TextPart(text=f'wrote {work / "notes.txt"}'))],
                )
            )
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    agent = define_custom_agent(registry, 'workspaceAgent', fn, store=store)
    out = await agent.chat().send('start').response
    assert out.snapshot_id is not None
    notes = workspace_root / out.snapshot_id / 'notes.txt'
    assert notes.is_file()
    assert notes.read_text(encoding='utf-8') == 'drafted during the turn\n'
