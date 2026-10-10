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

"""A failed agent turn returns AgentResponse.error instead of raising."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Part
from genkit._ai._agents._base import Agent, define_custom_agent
from genkit._ai._agents._client import AgentClient, AgentError
from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session_stores._inmemory_store import InMemorySessionStore
from genkit._ai._agents._types import TurnResult
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError, GenkitRuntimeError, RuntimeErrorReason
from genkit._core._model import AgentInit, AgentInput, AgentOutput, AgentResult, Message, SessionState
from genkit._core._registry import Registry
from genkit._core._typing import (
    AgentFinishReason,
    SnapshotStatus,
    ToolRequest,
)


class _TaskState(BaseModel):
    title: str
    done: bool = False


def _input_text(inp: AgentInput) -> str:
    message = inp.message
    if message is None:
        return ''
    return ''.join(part.text for part in (message.content or []) if part.text)


def _ok_message(parts: list[Part] | None = None) -> Message:
    return Message(role='model', content=parts or [Part.from_text('ok')])


def _define_turns(
    registry: Registry,
    name: str,
    handle: Callable[[SessionRunner, AgentInput], Awaitable[TurnResult | None]],
    *,
    store: InMemorySessionStore | None,
    state_schema: type[BaseModel] | None = None,
) -> Agent:
    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def turn(inp: AgentInput, __: Any) -> TurnResult | None:  # noqa: ANN401
            return await handle(session_runner, inp)

        await session_runner.run(turn)
        return await session_runner.result()

    return define_custom_agent(registry, name, fn, store=store, state_schema=state_schema)


def _flaky(
    registry: Registry,
    name: str,
    *,
    store: InMemorySessionStore | None,
    reply: Message | None = None,
    fail: Exception | None = None,
) -> Agent:
    """'fail' raises, 'block' is blocked, 'pause' interrupts, anything else replies."""
    boom = fail or GenkitError(status='INTERNAL', message='boom')
    model_reply = reply or _ok_message()

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        text = _input_text(inp).lower()
        if inp.resume is not None or 'fail' in text:
            raise boom
        if 'block' in text:
            return TurnResult(finish_reason=AgentFinishReason.BLOCKED)
        if 'pause' in text:
            await session_runner.add_messages([
                Message(
                    role='model',
                    content=[
                        Part(
                            tool_request=ToolRequest(name='approve', ref='r1', input={'ok': True}),
                            metadata={'interrupt': True},
                        )
                    ],
                )
            ])
            return TurnResult(finish_reason=AgentFinishReason.INTERRUPTED)
        await session_runner.add_messages([model_reply])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    return _define_turns(registry, name, handle, store=store)


def _assert_failed(res: Any, *, message: str) -> GenkitRuntimeError:  # noqa: ANN401
    assert res.finish_reason == AgentFinishReason.FAILED
    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.message == message
    assert res.message is None
    assert res.text == ''
    return res.error


class _DroppingTransport:
    """Dies before a turn result exists."""

    state_management = 'client'

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[Any], Awaitable[AgentOutput]]:
        del agent_input, init
        raise ConnectionError('socket closed')

    async def get_snapshot(self, *, snapshot_id: str | None = None, session_id: str | None = None) -> None:
        del snapshot_id, session_id
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        del snapshot_id
        return None


@pytest.mark.asyncio
async def test_send_on_failed_turn_returns_response_with_failed_finish_reason() -> None:
    """A turn whose handler raises returns finish_reason == 'failed' instead of raising."""
    agent = _flaky(Registry(), 'failFinish', store=InMemorySessionStore())
    res = await agent.chat().send('please fail now')
    assert res.finish_reason == AgentFinishReason.FAILED
    assert res.error is not None


@pytest.mark.asyncio
async def test_send_on_failed_turn_error_is_genkit_runtime_error_with_status_and_message() -> None:
    """res.error is a GenkitRuntimeError with status INTERNAL and message boom."""
    agent = _flaky(Registry(), 'failType', store=InMemorySessionStore())
    res = await agent.chat().send('please fail now')
    err = _assert_failed(res, message='boom')
    assert err.status == 'INTERNAL'


@pytest.mark.asyncio
async def test_send_on_failed_turn_with_reason_reads_reason() -> None:
    """A failed turn whose error carries a reason exposes it on res.error.reason."""
    agent = _flaky(
        Registry(),
        'failReason',
        store=InMemorySessionStore(),
        fail=GenkitError(status='INTERNAL', message='tool died', reason=RuntimeErrorReason.TOOL_FAILED),
    )
    res = await agent.chat().send('please fail now')
    err = _assert_failed(res, message='tool died')
    assert err.reason is RuntimeErrorReason.TOOL_FAILED


@pytest.mark.asyncio
async def test_send_on_failed_turn_keeps_last_good_snapshot_id() -> None:
    """res.snapshot_id and chat.snapshot_id both stay on the last completed snapshot."""
    agent = _flaky(Registry(), 'failSnap', store=InMemorySessionStore())
    chat = agent.chat()
    ok = await chat.send('hello')
    last_good = ok.snapshot_id
    assert last_good is not None

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert res.snapshot_id == last_good
    assert chat.snapshot_id == last_good


@pytest.mark.asyncio
async def test_send_on_failed_turn_drops_unanswered_prompt() -> None:
    """chat.messages and res.messages end at the last completed turn."""
    agent = _flaky(Registry(), 'failDrop', store=InMemorySessionStore())
    chat = agent.chat()
    await chat.send('hello')
    before = list(chat.messages)

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert chat.messages == before
    assert res.messages == before


@pytest.mark.asyncio
async def test_send_on_failed_client_managed_turn_returns_last_good_state() -> None:
    """Without a store, res.state is the state before the failed turn."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        if 'fail' in _input_text(inp).lower():
            await session_runner.update_custom(lambda _prev: {'n': 999})
            raise GenkitError(status='INTERNAL', message='boom')
        prev = await session_runner.get_custom()
        n = prev.get('n', 0) if isinstance(prev, dict) else 0
        await session_runner.update_custom(lambda _prev: {'n': n + 1})
        await session_runner.add_messages([_ok_message()])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Registry(), 'failState', handle, store=None)
    chat = agent.chat(state={'n': 0})
    ok = await chat.send('hello')
    assert ok.state == {'n': 1}

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert res.state == {'n': 1}
    assert chat.state == {'n': 1}


@pytest.mark.asyncio
async def test_send_after_failed_turn_continues_from_last_good_snapshot() -> None:
    """The next send succeeds and builds on the last good turn."""
    agent = _flaky(Registry(), 'failContinue', store=InMemorySessionStore())
    chat = agent.chat()
    first = await chat.send('hello')
    failed = await chat.send('please fail now')
    _assert_failed(failed, message='boom')

    again = await chat.send('hello again')
    assert again.finish_reason == AgentFinishReason.STOP
    assert again.error is None
    assert again.text == 'ok'
    assert again.snapshot_id not in (None, first.snapshot_id)
    texts = [part.text for message in chat.messages for part in message.content if part.text]
    assert 'hello again' in texts
    assert 'please fail now' not in texts


@pytest.mark.asyncio
async def test_send_stream_on_failed_turn_ends_stream_and_response_has_error() -> None:
    """The chunk stream ends normally and await turn.response returns with .error set."""
    agent = _flaky(Registry(), 'failStream', store=InMemorySessionStore())
    chat = agent.chat()
    await chat.send('hello')
    turn = chat.send_stream('please fail now')
    async for _chunk in turn.stream:
        pass
    res = await turn.response
    _assert_failed(res, message='boom')


@pytest.mark.asyncio
async def test_resume_on_failed_turn_returns_response_with_error() -> None:
    """chat.resume(...) that fails returns a response with .error."""
    agent = _flaky(Registry(), 'failResume', store=InMemorySessionStore())
    chat = agent.chat()
    paused = await chat.send('please pause')
    assert paused.finish_reason == AgentFinishReason.INTERRUPTED
    assert paused.interrupts

    res = await chat.resume(respond=[paused.interrupts[0].respond({'approved': True})])
    _assert_failed(res, message='boom')


@pytest.mark.asyncio
async def test_send_on_successful_turn_has_no_error() -> None:
    """A normal turn returns res.error is None."""
    agent = _flaky(Registry(), 'okTurn', store=InMemorySessionStore())
    res = await agent.chat().send('hello')
    assert res.finish_reason == AgentFinishReason.STOP
    assert res.error is None
    assert res.text == 'ok'
    assert res.message is not None


@pytest.mark.asyncio
async def test_send_on_blocked_turn_returns_blocked_without_raising() -> None:
    """A blocked turn still returns with finish_reason == 'blocked'."""
    agent = _flaky(Registry(), 'blockedTurn', store=InMemorySessionStore())
    res = await agent.chat().send('please block this')
    assert res.finish_reason == AgentFinishReason.BLOCKED
    assert res.error is None


@pytest.mark.asyncio
async def test_send_on_failed_turn_keeps_data_as_first_data_part() -> None:
    """res.data reads the first data part as before."""
    agent = _flaky(
        Registry(),
        'failData',
        store=InMemorySessionStore(),
        reply=_ok_message([Part.from_text('ok'), Part.from_data({'n': 1})]),
    )
    chat = agent.chat()
    ok = await chat.send('hello')
    assert ok.data == {'n': 1}

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert res.data == {'n': 1}


@pytest.mark.asyncio
async def test_send_with_snapshot_id_on_agent_without_store_raises_agent_error() -> None:
    """An init the server rejects as misuse still raises AgentError with FAILED_PRECONDITION."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Registry(), 'noStore', handle, store=None)
    with pytest.raises(AgentError) as exc:
        await agent.chat(snapshot_id='snap-1').send('hi')
    assert exc.value.status == 'FAILED_PRECONDITION'
    assert exc.value.reason is RuntimeErrorReason.SESSION_STORE_NOT_CONFIGURED


@pytest.mark.asyncio
async def test_send_over_dropped_connection_raises_agent_error() -> None:
    """A transport that fails before any turn result raises AgentError with the last good state."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del inp
        await session_runner.update_custom(lambda _prev: {'n': 1})
        await session_runner.add_messages([_ok_message()])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Registry(), 'dropConn', handle, store=None)
    chat = agent.chat(state={'n': 0})
    ok = await chat.send('hello')
    assert ok.state == {'n': 1}

    chat._transport = _DroppingTransport()  # noqa: SLF001
    with pytest.raises(AgentError) as exc:
        await chat.send('next')
    assert exc.value.state == {'n': 1}


@pytest.mark.asyncio
async def test_send_to_missing_snapshot_returns_response_with_snapshot_not_found() -> None:
    """agent.chat(snapshot_id='gone').send('hi') returns res.error.reason == SNAPSHOT_NOT_FOUND."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Registry(), 'goneSnap', handle, store=InMemorySessionStore())
    res = await agent.chat(snapshot_id='gone').send('hi')
    err = _assert_failed(res, message="Snapshot 'gone' not found")
    assert err.status == 'NOT_FOUND'
    assert err.reason is RuntimeErrorReason.SNAPSHOT_NOT_FOUND


@pytest.mark.asyncio
async def test_send_on_failed_turn_message_is_none_and_text_empty() -> None:
    """A failed turn's res.message is None and res.text == ''."""
    agent = _flaky(Registry(), 'failBlank', store=InMemorySessionStore())
    chat = agent.chat()
    ok = await chat.send('hello')
    assert ok.text == 'ok'

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert any(part.text == 'ok' for message in chat.messages for part in message.content)


@pytest.mark.asyncio
async def test_send_with_invalid_custom_state_returns_response_with_error() -> None:
    """A custom state that fails the schema comes back on res.error, not as a raise."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Registry(), 'badState', handle, store=None, state_schema=_TaskState)
    res = await agent.chat(state={'done': 'nope'}).send('hi')  # type: ignore[arg-type]
    assert res.finish_reason == AgentFinishReason.FAILED
    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert res.error.status == 'INVALID_ARGUMENT'
    assert res.state is None


class _ScriptedTransport:
    """Hands back canned turn outputs, like a remote server would."""

    def __init__(self, outputs: list[AgentOutput], *, state_management: str) -> None:
        self.outputs = outputs
        self.state_management = state_management

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[Any], Awaitable[AgentOutput]]:
        del agent_input, init

        async def no_chunks() -> AsyncIterator[Any]:
            for chunk in ():
                yield chunk

        async def output() -> AgentOutput:
            return self.outputs.pop(0)

        return no_chunks(), output()

    async def get_snapshot(self, *, snapshot_id: str | None = None, session_id: str | None = None) -> None:
        del snapshot_id, session_id
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        del snapshot_id
        return None


@pytest.mark.asyncio
async def test_send_on_aborted_turn_hides_previous_reply_and_keeps_last_snapshot() -> None:
    """An aborted turn shows no message, no text, and the last completed snapshot id."""
    # 1. A remote server answers once, then reports the next turn aborted.
    #    Its output still carries the previous reply and no snapshot id.
    previous = _ok_message([Part.from_text('Smoked Salmon Tartine')])
    transport = _ScriptedTransport(
        [
            AgentOutput(finish_reason=AgentFinishReason.STOP, snapshot_id='s1', message=previous),
            AgentOutput(finish_reason=AgentFinishReason.ABORTED, message=previous),
        ],
        state_management='server',
    )
    chat = AgentClient(transport).chat()  # type: ignore[arg-type]
    await chat.send('suggest a dish')
    history = list(chat.messages)

    # 2. The aborted turn does not read as an answer.
    res = await chat.send('something without nuts')
    assert res.finish_reason == AgentFinishReason.ABORTED
    assert res.message is None
    assert res.text == ''

    # 3. The caller still resumes from s1, and the unanswered prompt is gone.
    assert res.snapshot_id == 's1'
    assert chat.snapshot_id == 's1'
    assert chat.messages == history


@pytest.mark.asyncio
async def test_send_on_aborted_turn_with_invalid_state_returns_none_state() -> None:
    """An aborted turn whose custom state fails the schema returns state None instead of raising."""
    transport = _ScriptedTransport(
        [
            AgentOutput(
                finish_reason=AgentFinishReason.ABORTED,
                state=SessionState(custom={'done': 'nope'}),
            ),
        ],
        state_management='client',
    )
    chat = AgentClient(transport, state_schema=_TaskState).chat()  # type: ignore[arg-type]

    res = await chat.send('mark the order done')

    assert res.finish_reason == AgentFinishReason.ABORTED
    assert res.state is None
