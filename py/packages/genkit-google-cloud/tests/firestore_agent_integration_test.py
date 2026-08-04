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

"""End-to-End Genkit Agent Integration Tests with FirestoreSessionStore.

Tests full agent execution paths (agent.chat(), send(), tool loops, branching,
and detach/background turns) backed by FirestoreSessionStore.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from genkit_google_cloud import FirestoreSessionStore

from genkit._ai._aio import Genkit
from genkit._ai._testing import define_programmable_model
from genkit._core._model import FinishReason, Message, ModelResponse
from genkit._core._typing import (
    AgentInput,
    MessageData,
    Part,
    Role,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
)


def _mock_txn_client() -> tuple[MagicMock, MagicMock]:
    """Return (client, transaction) with async transactional plumbing mocked."""
    mock_client = MagicMock()
    mock_transaction = MagicMock()
    mock_transaction._max_attempts = 1
    mock_transaction._read_only = False
    mock_transaction._begin = AsyncMock()
    mock_transaction._commit = AsyncMock()
    mock_transaction._rollback = AsyncMock()
    mock_client.transaction.return_value = mock_transaction
    return mock_client, mock_transaction


def _doc(
    *,
    path: str,
    exists: bool = True,
    data: dict[str, Any] | None = None,
    doc_id: str | None = None,
) -> MagicMock:
    snap = MagicMock()
    snap.exists = exists
    snap.id = doc_id or path.rsplit('/', 1)[-1]
    snap.reference.path = path
    snap.to_dict.return_value = data
    return snap


class FakeStoreHarness:
    """In-memory Firestore stand-in wired for AsyncClient-style access."""

    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.client, self.transaction = _mock_txn_client()
        self.deleted: list[str] = []

        def collection(name: str) -> MagicMock:
            col = MagicMock()
            col_name = name

            def document(doc_id: str) -> MagicMock:
                prefix_ref = MagicMock()

                def subcollection(sub: str) -> MagicMock:
                    sub_col = MagicMock()

                    def item_document(item_id: str) -> MagicMock:
                        path = f'{col_name}/{doc_id}/{sub}/{item_id}'
                        ref = MagicMock(spec=['get', 'path', 'id', 'collection'])
                        ref.path = path
                        ref.id = item_id

                        async def get(*, transaction: Any = None) -> MagicMock:
                            if path in self.docs:
                                return _doc(path=path, exists=True, data=self.docs[path], doc_id=item_id)
                            return _doc(path=path, exists=False, data=None, doc_id=item_id)

                        ref.get = get
                        return ref

                    sub_col.document.side_effect = item_document
                    return sub_col

                prefix_ref.collection.side_effect = subcollection
                return prefix_ref

            col.document.side_effect = document
            return col

        self.client.collection.side_effect = collection

        def set_doc(ref: MagicMock, data: dict[str, Any]) -> None:
            self.docs[ref.path] = data

        def update_doc(ref: MagicMock, data: dict[str, Any]) -> None:
            existing = self.docs.get(ref.path, {})
            existing.update(data)
            self.docs[ref.path] = existing

        def delete_doc(ref: MagicMock) -> None:
            self.deleted.append(ref.path)
            self.docs.pop(ref.path, None)

        def get_all(refs: list[MagicMock]) -> list[MagicMock]:
            out = []
            for r in refs:
                if r.path in self.docs:
                    out.append(_doc(path=r.path, exists=True, data=self.docs[r.path], doc_id=r.id))
                else:
                    out.append(_doc(path=r.path, exists=False, data=None, doc_id=r.id))
            return out

        self.transaction.set.side_effect = set_doc
        self.transaction.update.side_effect = update_doc
        self.transaction.delete.side_effect = delete_doc
        self.transaction.get_all.side_effect = get_all

        async def txn_get(ref: MagicMock) -> MagicMock:
            if ref.path in self.docs:
                return _doc(path=ref.path, exists=True, data=self.docs[ref.path], doc_id=ref.id)
            return _doc(path=ref.path, exists=False, data=None, doc_id=ref.id)

        self.transaction.get.side_effect = txn_get

    def store(self, **kwargs: Any) -> FirestoreSessionStore[Any]:
        st = FirestoreSessionStore[Any](client=self.client, **kwargs)
        st._ensure_sync_client = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
        return st


def fake_transactional(fn: Any) -> Any:
    async def wrapper(transaction: Any, *args: Any, **kwargs: Any) -> Any:
        return await fn(transaction, *args, **kwargs)

    return wrapper


@pytest.fixture(autouse=True)
def patch_firestore_transactional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('google.cloud.firestore.async_transactional', fake_transactional)


@pytest.mark.asyncio
async def test_agent_multi_turn_chat_session_persistence() -> None:
    """Verify multi-turn agent chat persists history in Firestore and resumes correctly by session_id."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = FakeStoreHarness()
    store = h.store()

    # Define agent backed by FirestoreSessionStore
    agent = ai.define_agent(
        name='multiTurnFirestoreAgent',
        model='programmableModel',
        system='You are a helpful assistant.',
        store=store,
    )

    # Queue Model Responses for Turn 1 and Turn 2
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='4'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='9'))]),
            finish_reason=FinishReason.STOP,
        )
    )

    # Turn 1
    chat = agent.chat()
    res1 = await chat.send('What is 2+2?')
    assert res1.text == '4'
    session_id = chat.session_id
    assert session_id is not None

    # Turn 2
    res2 = await chat.send('Add 5 to that')
    assert res2.text == '9'

    # Resume the session via session_id on a new chat instance
    resumed_chat = agent.chat(session_id=session_id)
    # Force state load from Firestore store
    snapshot = await store.get_snapshot(session_id=session_id)
    assert snapshot is not None
    assert snapshot.state is not None
    assert snapshot.state.messages is not None
    assert len(snapshot.state.messages) == 4

    # Queue Model Response for Turn 3 on resumed chat
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='27'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    res3 = await resumed_chat.send('Multiply by 3')
    assert res3.text == '27'

    # Final Firestore verification
    final_snap = await store.get_snapshot(session_id=session_id)
    assert final_snap is not None
    assert final_snap.state is not None
    assert final_snap.state.messages is not None
    assert len(final_snap.state.messages) == 6
    assert final_snap.state.messages[0].content[0].root.text == 'What is 2+2?'
    assert final_snap.state.messages[1].content[0].root.text == '4'
    assert final_snap.state.messages[2].content[0].root.text == 'Add 5 to that'
    assert final_snap.state.messages[3].content[0].root.text == '9'
    assert final_snap.state.messages[4].content[0].root.text == 'Multiply by 3'
    assert final_snap.state.messages[5].content[0].root.text == '27'


@pytest.mark.asyncio
async def test_agent_tool_loop_firestore_persistence() -> None:
    """Verify agent execution over a tool loop persists full tool request/response messages in Firestore."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = FakeStoreHarness()
    store = h.store()

    @ai.tool()
    async def get_weather(city: str) -> str:
        return '22C Sunny'

    ai.define_prompt(
        name='toolAgentPrompt',
        model='programmableModel',
        system='Check weather using get_weather.',
        tools=[get_weather],
    )
    agent = ai.define_prompt_agent(name='toolAgentPrompt', store=store)

    # Queue model responses: 1) Tool Call Request, 2) Final Answer using Tool Response
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='get_weather', ref='req1', input='Paris')))
                ],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='The weather in Paris is 22C Sunny.'))]),
        )
    )

    chat = agent.chat()
    res = await chat.send('Weather in Paris?')
    assert res.text == 'The weather in Paris is 22C Sunny.'
    session_id = chat.session_id
    assert session_id is not None

    # Load from Firestore and verify full tool message sequence
    snapshot = await store.get_snapshot(session_id=session_id)
    assert snapshot is not None
    assert snapshot.state is not None
    assert snapshot.state.messages is not None
    [m.role for m in snapshot.state.messages]
    tr = snapshot.state.messages[1].content[0].root.tool_request
    assert tr is not None
    assert tr.name == 'get_weather'
    assert snapshot.state.messages[3].content[0].root.text == 'The weather in Paris is 22C Sunny.'


@pytest.mark.asyncio
async def test_agent_branching_turns_firestore_pointers() -> None:
    """Verify branching agent conversations maintain multiple leaves in Firestore and update specific branch tips."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = FakeStoreHarness()
    store = h.store()

    agent = ai.define_agent(
        name='branchingAgent',
        model='programmableModel',
        store=store,
    )

    # Root Turn
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Root Turn Response'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    chat_root = agent.chat()
    await chat_root.send('Root Turn Question')
    session_id = chat_root.session_id
    root_snap_id = chat_root.snapshot_id
    assert session_id is not None
    assert root_snap_id is not None

    # Branch 1 from root_snap_id
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Branch 1 Response'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    chat_b1 = agent.chat(snapshot_id=root_snap_id)
    await chat_b1.send('Branch 1 Question')

    # Branch 2 from root_snap_id
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Branch 2 Response'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    chat_b2 = agent.chat(snapshot_id=root_snap_id)
    await chat_b2.send('Branch 2 Question')

    # Check Firestore pointer tracks both branches and marks isAmbiguous=True
    pointer_path = f'genkit-sessions-pointers/global/pointers/{session_id}'
    pointer_doc = h.docs[pointer_path]
    assert pointer_doc['isAmbiguous'] is True
    assert len(pointer_doc['leaves']) == 2
    assert chat_b1.snapshot_id in pointer_doc['leaves']
    assert chat_b2.snapshot_id in pointer_doc['leaves']

    # Follow-up turn on Branch 1
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Branch 1 Followup Response'))]),
            finish_reason=FinishReason.STOP,
        )
    )
    await chat_b1.send('Branch 1 Followup')

    # Verify Firestore pointer updates Branch 1 tip while leaving Branch 2 tip intact
    pointer_doc_updated = h.docs[pointer_path]
    assert pointer_doc_updated['isAmbiguous'] is True
    assert len(pointer_doc_updated['leaves']) == 2
    assert chat_b1.snapshot_id in pointer_doc_updated['leaves']
    assert chat_b2.snapshot_id in pointer_doc_updated['leaves']


@pytest.mark.asyncio
async def test_agent_detach_background_turn_firestore_persistence() -> None:
    """Verify running a detached agent turn persists snapshot status and state in Firestore."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    h = FakeStoreHarness()
    store = h.store()

    agent = ai.define_agent(
        name='detachAgent',
        model='programmableModel',
        store=store,
    )

    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Async Detached Reply'))]),
            finish_reason=FinishReason.STOP,
        )
    )

    # Issue detached turn using AgentInput(detach=True)
    input_payload = AgentInput(
        message=MessageData(role=Role.USER, content=[Part(root=TextPart(text='Run in background'))]),
        detach=True,
    )
    result = await agent.run(input_payload)

    # Detached run returns ActionResponse carrying response=AgentOutput
    assert result.response is not None
    assert result.response.snapshot_id is not None
    snap_id = result.response.snapshot_id

    # Wait briefly for background execution to write final snapshot to Firestore
    snapshot = None
    for _ in range(50):
        snapshot = await store.get_snapshot(snapshot_id=snap_id)
        if snapshot is not None and snapshot.status == SnapshotStatus.COMPLETED:
            break
        await asyncio.sleep(0.05)

    assert snapshot is not None
    assert snapshot.status == SnapshotStatus.COMPLETED
    assert snapshot.state is not None
    assert snapshot.state.messages is not None
    assert snapshot.state.messages[-1].content[0].root.text == 'Async Detached Reply'
