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

"""Agent turn spans: session id on the root span, session state on runTurn."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

import pytest

from genkit import Part
from genkit._ai._agents._base import define_custom_agent
from genkit._ai._agents._runtime import SessionRunner
from genkit._ai._agents._session import Session
from genkit._ai._agents._types import TurnContext, TurnResult
from genkit._core._action import ActionRunContext
from genkit._core._model import AgentInput, AgentResult, Message, SessionState
from genkit._core._registry import Registry
from genkit._core._telemetry._attrs import Attr, metadata_key
from genkit._core._telemetry.http import ActiveSpan
from genkit.exp.agent import AgentFinishReason, InMemorySessionStore

UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
SESSION_ID_ATTR = metadata_key('agent:sessionId')
SNAPSHOT_ID_ATTR = metadata_key('agent:snapshotId')


def _by_name(spans: Sequence[ActiveSpan], name: str) -> ActiveSpan:
    matches = [s for s in spans if s.name == name]
    assert matches, f'no span named {name!r} in {[s.name for s in spans]}'
    return matches[-1]


def _counter_agent(
    *,
    registry: Registry,
    name: str,
    store: InMemorySessionStore | None,
):
    async def fn(session_runner: SessionRunner, _: ActionRunContext) -> AgentResult:
        async def handle_turn(_: AgentInput, __: TurnContext) -> TurnResult | None:
            def bump(custom: dict | None) -> dict:
                return {'count': (custom or {}).get('count', 0) + 1}

            await session_runner.update_custom(bump)
            await session_runner.add_messages([Message(role='model', content=[Part.from_text('done')])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await session_runner.run(handle_turn)
        return await session_runner.result()

    return define_custom_agent(registry, name, fn, store=store)


def test_session_mints_session_id_when_missing() -> None:
    """Session() without a session_id assigns one."""
    session = Session()
    assert session.session_state.session_id
    assert UUID_RE.match(session.session_state.session_id)


def test_session_preserves_existing_session_id() -> None:
    """Session(state) keeps the session_id they already set."""
    session = Session(SessionState(session_id='keep-me', custom={'x': 1}))
    assert session.session_state.session_id == 'keep-me'
    assert session.session_state.custom == {'x': 1}


def test_session_does_not_mutate_caller_state() -> None:
    """Session(state) does not write a session_id back onto the object they passed in."""
    seed = SessionState(custom={'n': 1})
    session = Session(seed)
    assert session.session_state.session_id
    assert seed.session_id is None


@pytest.mark.asyncio
async def test_run_turn_span_output_is_session_state_with_store(
    exporter,
) -> None:
    """Session store: root span has session id; runTurn output is the stored state."""
    registry = Registry()
    store = InMemorySessionStore()
    agent = _counter_agent(registry=registry, name='turnSpanStore', store=store)

    out = await agent.chat().send('hi')
    assert out.snapshot_id
    assert out.session_id
    assert UUID_RE.match(out.session_id)

    spans = exporter.get_finished_spans()
    root = _by_name(spans, 'turnSpanStore')
    assert root.attributes is not None
    assert root.attributes[SESSION_ID_ATTR] == out.session_id

    turn_span = _by_name(spans, 'runTurn-1')
    assert turn_span.attributes is not None
    assert turn_span.attributes[SNAPSHOT_ID_ATTR] == out.snapshot_id
    assert SESSION_ID_ATTR not in turn_span.attributes

    payload = json.loads(turn_span.attributes[Attr.OUTPUT])
    assert payload['state']['custom'] == {'count': 1}
    assert payload['state']['sessionId'] == out.session_id
    assert 'messages' in payload['state']
    assert 'finishReason' not in payload


@pytest.mark.asyncio
async def test_run_turn_span_output_is_session_state_client_managed(
    exporter,
) -> None:
    """No store: root span still has session id; runTurn output is the in-memory state."""
    registry = Registry()
    agent = _counter_agent(registry=registry, name='turnSpanClient', store=None)

    out = await agent.chat().send('hi')
    assert out.raw.state is not None
    assert out.raw.state.session_id
    assert UUID_RE.match(out.raw.state.session_id)
    assert out.session_id == out.raw.state.session_id

    spans = exporter.get_finished_spans()
    root = _by_name(spans, 'turnSpanClient')
    assert root.attributes is not None
    assert root.attributes[SESSION_ID_ATTR] == out.session_id

    turn_span = _by_name(spans, 'runTurn-1')
    assert turn_span.attributes is not None
    assert SNAPSHOT_ID_ATTR not in turn_span.attributes
    assert SESSION_ID_ATTR not in turn_span.attributes

    payload = json.loads(turn_span.attributes[Attr.OUTPUT])
    assert payload['state']['custom'] == {'count': 1}
    assert payload['state']['sessionId'] == out.session_id
    assert 'finishReason' not in payload


@pytest.mark.asyncio
async def test_client_managed_preserves_session_id_across_turns() -> None:
    """Two send() calls on the same chat keep the same session_id."""
    registry = Registry()
    agent = _counter_agent(registry=registry, name='preserveClientSid', store=None)

    chat = agent.chat()
    out1 = await chat.send('one')
    assert out1.raw.state is not None
    sid = out1.raw.state.session_id
    assert sid

    out2 = await chat.send('two')
    assert out2.raw.state is not None
    assert out2.raw.state.session_id == sid
