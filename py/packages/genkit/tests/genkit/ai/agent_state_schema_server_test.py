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

"""Server-side validation of custom state against an agent's state_schema."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from genkit._ai._agents._runtime import load_session, validate_custom_state
from genkit._core._error import GenkitError
from genkit._core._typing import (
    AgentInit,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)
from genkit.agent import InMemorySessionStore

INVALID_ARGUMENT = 'INVALID_ARGUMENT'


class TaskState(BaseModel):
    title: str
    done: bool = False


def test_validate_custom_state_noops_without_schema() -> None:
    validate_custom_state(custom={'anything': 1}, state_schema=None, agent_name='agent')


def test_validate_custom_state_skips_unset_state() -> None:
    # A required-field schema must not trip a session that never wrote state.
    validate_custom_state(custom=None, state_schema=TaskState, agent_name='agent')


def test_validate_custom_state_accepts_valid() -> None:
    validate_custom_state(custom={'title': 'ship it', 'done': True}, state_schema=TaskState, agent_name='agent')


def test_validate_custom_state_rejects_invalid() -> None:
    with pytest.raises(GenkitError) as exc:
        validate_custom_state(custom={'done': 'not-a-bool'}, state_schema=TaskState, agent_name='taskAgent')
    assert exc.value.status == INVALID_ARGUMENT
    assert 'taskAgent' in str(exc.value)


def test_validate_custom_state_error_carries_field_details() -> None:
    with pytest.raises(GenkitError) as exc:
        validate_custom_state(custom={'done': 'not-a-bool'}, state_schema=TaskState, agent_name='taskAgent')
    details = exc.value.details
    # The expected shape plus each per-field failure, so callers can show why.
    assert 'schema' in details
    failed_fields = {tuple(err['loc']) for err in details['errors']}
    assert ('title',) in failed_fields  # required, missing
    assert ('done',) in failed_fields  # wrong type


@pytest.mark.asyncio
async def test_load_session_client_managed_validates_state() -> None:
    valid = AgentInit(state=SessionState(custom={'title': 'x'}))
    session, snap = await load_session(init=valid, store=None, agent_name='a', state_schema=TaskState)
    assert snap is None
    assert (await session.get_custom()) == {'title': 'x'}

    bad = AgentInit(state=SessionState(custom={'done': True}))  # missing required title
    with pytest.raises(GenkitError) as exc:
        await load_session(init=bad, store=None, agent_name='a', state_schema=TaskState)
    assert exc.value.status == INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_load_session_validates_snapshot_custom() -> None:
    store = InMemorySessionStore()
    session_id = 'sess-bad'
    snap = SessionSnapshot(
        snapshot_id='snap-1',
        parent_id=None,
        created_at='2026-06-18T12:00:00Z',
        status=SnapshotStatus.COMPLETED,
        state=SessionState(session_id=session_id, messages=[], artifacts=[], custom={'done': True}),
    )
    await store.save_snapshot(None, lambda _existing: snap)

    with pytest.raises(GenkitError) as exc:
        await load_session(init=AgentInit(session_id=session_id), store=store, agent_name='a', state_schema=TaskState)
    assert exc.value.status == INVALID_ARGUMENT
