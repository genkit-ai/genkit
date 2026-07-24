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

"""Resume guards in load_session: only completed snapshots resume, leaf walk-back."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from genkit._ai._agents._runtime import load_session
from genkit._ai._agents._session import SessionStore
from genkit._core._error import GenkitError
from genkit._core._typing import (
    AgentInit,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
)

INVALID_ARGUMENT = 'INVALID_ARGUMENT'
FAILED_PRECONDITION = 'FAILED_PRECONDITION'

SESSION_ID = 's1'


def _snap(
    snapshot_id: str,
    status: SnapshotStatus,
    parent_id: str | None = None,
) -> SessionSnapshot:
    return SessionSnapshot(
        snapshot_id=snapshot_id,
        session_id=SESSION_ID,
        parent_id=parent_id,
        created_at='2026-06-18T12:00:00Z',
        status=status,
        state=SessionState(session_id=SESSION_ID, messages=[], artifacts=[]),
    )


class _ScriptedStore(SessionStore[Any]):
    """Returns snapshots by id and a designated leaf by session, to drive load_session."""

    def __init__(self, by_id: dict[str, SessionSnapshot], leaf: SessionSnapshot | None) -> None:
        self._by_id = by_id
        self._leaf = leaf

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        if snapshot_id is not None:
            return self._by_id.get(snapshot_id)
        if session_id is not None:
            return self._leaf
        return None

    async def save_snapshot(
        self,
        snapshot_id: str | None,
        fn: Callable[[SessionSnapshot | None], SessionSnapshot | None],
    ) -> SessionSnapshot | None:
        return None


@pytest.mark.asyncio
async def test_resume_by_snapshot_id_rejects_non_completed() -> None:
    failed = _snap('snap-f', SnapshotStatus.FAILED)
    store = _ScriptedStore({'snap-f': failed}, leaf=None)

    with pytest.raises(GenkitError) as exc:
        await load_session(init=AgentInit(snapshot_id='snap-f'), store=store, agent_name='a')
    assert exc.value.status == INVALID_ARGUMENT
    assert 'not resumable' in str(exc.value)


@pytest.mark.asyncio
async def test_resume_by_session_id_walks_back_to_last_completed() -> None:
    completed = _snap('snap-c', SnapshotStatus.COMPLETED)
    failed = _snap('snap-f', SnapshotStatus.FAILED, parent_id='snap-c')
    store = _ScriptedStore({'snap-c': completed, 'snap-f': failed}, leaf=failed)

    _session, snap = await load_session(init=AgentInit(session_id=SESSION_ID), store=store, agent_name='a')
    assert snap is not None
    assert snap.snapshot_id == 'snap-c'


@pytest.mark.asyncio
async def test_resume_by_session_id_cyclic_chain_raises() -> None:
    a = _snap('a', SnapshotStatus.FAILED, parent_id='b')
    b = _snap('b', SnapshotStatus.FAILED, parent_id='a')
    store = _ScriptedStore({'a': a, 'b': b}, leaf=a)

    with pytest.raises(GenkitError) as exc:
        await load_session(init=AgentInit(session_id=SESSION_ID), store=store, agent_name='a')
    assert exc.value.status == FAILED_PRECONDITION
    assert 'cyclic' in str(exc.value)


@pytest.mark.asyncio
async def test_resume_by_session_id_no_completed_seeds_fresh() -> None:
    failed = _snap('snap-f', SnapshotStatus.FAILED)  # no parent, not resumable
    store = _ScriptedStore({'snap-f': failed}, leaf=failed)

    session, snap = await load_session(init=AgentInit(session_id=SESSION_ID), store=store, agent_name='a')
    assert snap is None
    state = await session.state()
    assert state.session_id == SESSION_ID
