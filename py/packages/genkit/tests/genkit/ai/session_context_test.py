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

import pytest

from genkit import Genkit
from genkit._ai._agents._runtime import AgentRuntime, SessionRunner
from genkit._ai._agents._session import Session, get_current_session, run_with_session
from genkit._core._action import ActionRunContext
from genkit._core._channel import CloseableQueue
from genkit._core._typing import AgentInput, AgentResult, SessionState
from genkit.middleware import GenerateMiddlewareContext


@pytest.mark.asyncio
async def test_get_current_session_outside_bind() -> None:
    assert get_current_session() is None


@pytest.mark.asyncio
async def test_middleware_context_session_field() -> None:
    ai = Genkit()
    ctx = GenerateMiddlewareContext(ai=ai)
    assert ctx.ai.current_session() is None

    session = Session(SessionState(custom={'bound': True}))

    async def check() -> None:
        assert ctx.ai.current_session() is session

    await run_with_session(session=session, coro=check())


@pytest.mark.asyncio
async def test_run_with_session_binds_and_clears() -> None:
    session = Session(SessionState(custom={'count': 0}))

    async def inner() -> Session | None:
        bound = get_current_session()
        assert bound is session
        return bound

    result = await run_with_session(session=session, coro=inner())
    assert result is session
    assert get_current_session() is None


@pytest.mark.asyncio
async def test_run_with_session_nested_bind() -> None:
    outer = Session(SessionState(custom={'label': 'outer'}))
    inner = Session(SessionState(custom={'label': 'inner'}))

    async def nested() -> str:
        assert get_current_session() is inner
        cur = get_current_session()
        assert cur is not None
        custom = await cur.get_custom()
        assert isinstance(custom, dict)
        return custom['label']

    async def outer_fn() -> tuple[dict[str, str] | None, str]:
        assert get_current_session() is outer
        label = await run_with_session(session=inner, coro=nested())
        assert get_current_session() is outer
        custom = await outer.get_custom()
        assert isinstance(custom, dict)
        return custom, label

    custom, nested_label = await run_with_session(session=outer, coro=outer_fn())
    assert custom == {'label': 'outer'}
    assert nested_label == 'inner'


@pytest.mark.asyncio
async def test_agent_runtime_binds_session_during_handler() -> None:
    """AgentRuntime.run wraps the agent fn in run_with_session."""
    out_queue = CloseableQueue()
    session = Session(SessionState(custom={'seen': False}))
    rt = AgentRuntime(
        name='test',
        session=session,
        parent_snapshot=None,
        store=None,
        state_transform=None,
        chunk_transform=None,
        emit_chunk=out_queue.put_nowait,
    )
    seen: list[Session | None] = []

    async def agent_fn(session_runner: SessionRunner, ctx: ActionRunContext) -> AgentResult:
        seen.append(get_current_session())

        async def handle_turn(_inp: AgentInput) -> None:
            seen.append(get_current_session())
            cur = get_current_session()
            assert cur is not None
            await cur.update_custom(lambda c: {**(c or {}), 'seen': True})

        await session_runner.run(handle_turn)
        return await session_runner.result()

    in_queue = CloseableQueue()
    in_queue.put_nowait(AgentInput())
    in_queue.close()

    await rt.run(fn=agent_fn, client_inputs=in_queue)

    assert seen == [session, session]
    assert (await session.get_custom()) == {'seen': True}
