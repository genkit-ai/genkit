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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import Any

import pytest
from pydantic import BaseModel

from genkit._ai._agents._client import (
    AgentChat,
    AgentError,
    AgentInterrupt,
    AgentTransport,
    DetachedTask,
    TurnDriver,
)
from genkit._ai._agents._runtime import AgentInitError
from genkit._ai._agents._types import StateManagement
from genkit._ai._aio import Genkit
from genkit._ai._json_patch import apply_json_patch
from genkit._ai._testing import define_programmable_model
from genkit._core._channel import CloseableQueue
from genkit._core._model import Message, ModelResponse, ModelResponseChunk as ModelResponseChunkModel
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    FinishReason,
    JsonPatch,
    JsonPatchOp,
    JsonPatchOperation,
    MessageData,
    ModelResponseChunk,
    Part,
    Role,
    SessionSnapshot,
    SessionState,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
    TurnEnd,
)
from genkit.agent import InMemorySessionStore

# ---------------------------------------------------------------------------
# Unit tests for JSON patch application
# ---------------------------------------------------------------------------


def test_apply_json_patch_root_replace() -> None:
    patch = [JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'status': 'idle', 'score': 10})]
    res = apply_json_patch(doc=None, patch=patch)
    assert res == {'status': 'idle', 'score': 10}


def test_apply_json_patch_nested_replace() -> None:
    doc = {'status': 'idle', 'nested': {'value': 1}}
    patch = [JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/nested/value', value=2)]
    res = apply_json_patch(doc=doc, patch=patch)
    assert res == {'status': 'idle', 'nested': {'value': 2}}


def test_apply_json_patch_array_add() -> None:
    doc = {'items': [1, 2]}
    patch = [JsonPatchOperation(op=JsonPatchOp.ADD, path='/items/-', value=3)]
    res = apply_json_patch(doc=doc, patch=patch)
    assert res == {'items': [1, 2, 3]}


def test_apply_json_patch_array_remove() -> None:
    doc = {'items': [1, 2, 3]}
    patch = [JsonPatchOperation(op=JsonPatchOp.REMOVE, path='/items/1')]
    res = apply_json_patch(doc=doc, patch=patch)
    assert res == {'items': [1, 3]}


# ---------------------------------------------------------------------------
# Mock Transport for Testing Stateful Connections
# ---------------------------------------------------------------------------


class MockAgentTransport(AgentTransport[Any]):
    def __init__(self, *, state_management: StateManagement = 'server') -> None:
        self.connect_init: AgentInit | None = None
        self.send_payloads: list[AgentInput] = []
        self.final_output: AgentOutput | None = None
        self.abort_snapshot_id: str | None = None
        self.state_management: StateManagement = state_management
        self._receive_queue: asyncio.Queue[AgentStreamChunk | None] = asyncio.Queue()

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        self.connect_init = init
        self.send_payloads.append(agent_input)

        async def _generator() -> AsyncIterator[AgentStreamChunk]:
            while True:
                chunk = await self._receive_queue.get()
                if chunk is None:
                    break
                yield chunk

        async def _output_waiter() -> AgentOutput:
            assert self.final_output is not None
            return self.final_output

        return _generator(), _output_waiter()

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        self.abort_snapshot_id = snapshot_id
        return SnapshotStatus.PENDING

    def push_chunk(self, chunk: AgentStreamChunk | None) -> None:
        self._receive_queue.put_nowait(chunk)


# ---------------------------------------------------------------------------
# AgentInterrupt builders
# ---------------------------------------------------------------------------


def test_restart_applies_replace_input() -> None:
    intr = AgentInterrupt('transfer', 'ref-1', {'amount': 100})
    part = intr.restart(replace_input={'amount': 50, 'approved': True})

    assert part.tool_request.input == {'amount': 50, 'approved': True}
    assert part.metadata is not None
    assert part.metadata.get('replacedInput') == {'amount': 100}
    assert part.metadata.get('resumed') is True


# ---------------------------------------------------------------------------
# AgentInit validation
# ---------------------------------------------------------------------------


def test_connect_init_rejects_multiple_resume_fields() -> None:
    with pytest.raises(AgentInitError, match="Cannot send 'state' together with"):
        AgentChat(
            MockAgentTransport(),
            AgentInit(state=SessionState(), snapshot_id='snap-1'),
        )


@pytest.mark.asyncio
async def test_detached_task_abort_returns_previous_status() -> None:
    transport = MockAgentTransport()
    task = DetachedTask(snapshot_id='snap-1', transport=transport)
    assert await task.abort() == SnapshotStatus.PENDING
    assert transport.abort_snapshot_id == 'snap-1'


def test_connect_init_allows_snapshot_id_and_session_id() -> None:
    # snapshot_id picks the row; session_id is an ownership check.
    chat = AgentChat(
        MockAgentTransport(state_management='server'),
        AgentInit(snapshot_id='snap-1', session_id='sess-1'),
    )
    assert chat.snapshot_id == 'snap-1'
    assert chat.session_id == 'sess-1'


def test_connect_init_applies_state_only() -> None:
    state = SessionState(session_id='sess-1', custom={'x': 1})
    chat = AgentChat(MockAgentTransport(state_management='client'), AgentInit(state=state))

    assert chat.session_id == 'sess-1'
    assert chat.state == {'x': 1}
    assert chat.snapshot_id is None


def test_connect_init_rejects_state_on_server_managed_chat() -> None:
    with pytest.raises(AgentInitError, match="Cannot send 'state'"):
        AgentChat(MockAgentTransport(state_management='server'), AgentInit(state=SessionState(custom={'x': 1})))


def test_connect_init_applies_snapshot_id_only() -> None:
    chat = AgentChat(MockAgentTransport(), AgentInit(snapshot_id='snap-1'))

    assert chat.snapshot_id == 'snap-1'
    assert chat.session_id is None


def test_connect_init_applies_session_id_only() -> None:
    chat = AgentChat(MockAgentTransport(), AgentInit(session_id='sess-1'))

    assert chat.session_id == 'sess-1'
    assert chat.snapshot_id is None


@pytest.mark.asyncio
async def test_wire_init_derives_from_live_session_state() -> None:
    """The chat rebuilds the resume payload from live state each turn, not a stored init."""
    transport = MockAgentTransport()
    chat = AgentChat(transport, AgentInit(session_id='sess-bootstrap'))

    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Hi'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    turn = chat.send_stream('Hello')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn.response

    # First turn (no snapshot yet) resumes by the bootstrap session id.
    assert transport.connect_init == AgentInit(session_id='sess-bootstrap')
    # Output advanced the live snapshot id; later turns resume that row only.
    assert chat.snapshot_id == 'snap-1'
    assert chat._wire_init() == AgentInit(snapshot_id='snap-1')


# ---------------------------------------------------------------------------
# Turn and Session Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_sends_input_and_aggregates_state() -> None:
    transport = MockAgentTransport()

    # Every turn ships the whole session back; the client copies it verbatim.
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
        state=SessionState(
            messages=[
                MessageData(role='user', content=[Part(root=TextPart(text='Weather in Tokyo?'))]),
                MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
            ],
            custom={'unit': 'celsius'},
        ),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    # Queue up chunks to simulate streaming
    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Weather is '))]))
    )
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Sunny.'))])))
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(
                root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'unit': 'celsius'})]
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    # Consume stream chunks
    chunks = []
    async for chunk in turn.stream:
        chunks.append(chunk)

    assert len(chunks) == 4
    assert chunks[0].text == 'Weather is '
    assert chunks[1].text == 'Sunny.'
    assert chunks[2].text is None

    # Verify custom state patch applied
    assert chat.state == {'unit': 'celsius'}

    # Await output to verify final response resolved correctly
    output = await turn.response
    assert output.finish_reason == AgentFinishReason.STOP
    assert output.message is not None
    assert output.message.content is not None
    assert output.message.content[0].root.text == 'Final output!'

    # Verify chat fields are updated after turn completion
    assert chat.snapshot_id == 'snapshot_1'
    assert len(chat.messages) == 2  # Turn 1 User input + model final output
    assert chat.messages[0].content[0].root.text == 'Weather in Tokyo?'
    assert chat.messages[1].content[0].root.text == 'Final output!'


class _Progress(BaseModel):
    turns: int = 0


@pytest.mark.asyncio
async def test_state_schema_coerces_custom_into_model() -> None:
    """With a state_schema the live state, streamed patch, and response materialize the model."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='ok'))]),
        state=SessionState(custom={'turns': 1}),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport, state_schema=_Progress)
    turn = chat.send_stream('go')
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'turns': 1})])
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))

    streamed = [chunk.custom async for chunk in turn.stream if chunk.custom is not None]
    res = await turn.response

    assert isinstance(chat.state, _Progress) and chat.state.turns == 1
    assert isinstance(res.state, _Progress) and res.state.turns == 1
    assert streamed and all(isinstance(c, _Progress) for c in streamed)


@pytest.mark.asyncio
async def test_no_state_schema_leaves_custom_as_dict() -> None:
    """Without a schema, custom stays the raw wire mapping (backward compatible)."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='ok'))]),
        state=SessionState(custom={'turns': 1}),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('go')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    res = await turn.response

    assert chat.state == {'turns': 1}
    assert res.state == {'turns': 1}


@pytest.mark.asyncio
async def test_server_managed_appends_messages_incrementally() -> None:
    """Server-managed turns ship only snapshot_id + final reply; the client keeps
    a running view by appending the user input and the turn's final message."""
    transport = MockAgentTransport(state_management='server')
    chat = AgentChat(transport)

    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='A1'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('U1')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn.response

    assert chat.snapshot_id == 'snap-1'
    assert [m.content[0].root.text for m in chat.messages] == ['U1', 'A1']

    transport.final_output = AgentOutput(
        snapshot_id='snap-2',
        message=MessageData(role='model', content=[Part(root=TextPart(text='A2'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn2 = chat.send_stream('U2')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-2', finish_reason=AgentFinishReason.STOP)))
    await turn2.response

    assert chat.snapshot_id == 'snap-2'
    assert [m.content[0].root.text for m in chat.messages] == ['U1', 'A1', 'U2', 'A2']


@pytest.mark.asyncio
async def test_server_managed_reconstructs_intermediate_tool_messages() -> None:
    """A server-managed turn's tool steps ride home on the chunk stream, not the
    output, so the running view must stitch them back from the chunks: text
    deltas merge into the model message, and the tool reply lands in between."""
    transport = MockAgentTransport(state_management='server')
    chat = AgentChat(transport)

    # The wire returns only the snapshot id + the final reply.
    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('Weather in Tokyo?')

    # Model message that calls a tool, streamed as text deltas + a tool request.
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[Part(root=TextPart(text='Let me '))],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[
                    Part(root=TextPart(text='check.')),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='weather', ref='c1', input={'city': 'Tokyo'})
                        )
                    ),
                ],
            )
        )
    )
    # Tool reply, streamed whole.
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.TOOL,
                index=1,
                content=[
                    Part(root=ToolResponsePart(tool_response=ToolResponse(name='weather', ref='c1', output='12C')))
                ],
            )
        )
    )
    # Final model message, streamed as text deltas (superseded by raw.message).
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL, index=2, content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]
            )
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn.response

    roles = [m.role for m in chat.messages]
    assert roles == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]

    # User input, the tool-calling model message (deltas merged + tool request),
    # the tool reply, then the authoritative final reply.
    user_msg, tool_call_msg, tool_reply_msg, final_msg = chat.messages
    assert user_msg.content[0].root.text == 'Weather in Tokyo?'
    assert tool_call_msg.content[0].root.text == 'Let me check.'
    tool_req = tool_call_msg.content[1].root
    assert isinstance(tool_req, ToolRequestPart)
    assert tool_req.tool_request.name == 'weather'
    tool_resp = tool_reply_msg.content[0].root
    assert isinstance(tool_resp, ToolResponsePart)
    assert tool_resp.tool_response.output == '12C'
    assert final_msg.content[0].root.text == 'It is 12C in Tokyo.'


@pytest.mark.asyncio
async def test_client_managed_stitches_tool_messages_from_chunks_not_output_state() -> None:
    """Client-managed turns build the running view the same way server-managed ones
    do — from the chunk stream — even though the output round-trips the whole blob.
    The output state is authoritative only for the non-message bits (custom); the
    intermediate tool steps come from the chunks, and the full stitched view is what
    ships back for the next turn's resume."""
    transport = MockAgentTransport(state_management='client')
    chat = AgentChat(transport)

    # The output round-trips state, but its messages deliberately omit the tool
    # steps so the test proves the view is stitched from chunks, not raw.state.
    # session_id inside the round-tripped state is adopted so the next turn's
    # state blob stays self-describing.
    transport.final_output = AgentOutput(
        message=MessageData(role='model', content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]),
        state=SessionState(session_id='sess-client-1', custom={'unit': 'celsius'}),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('Weather in Tokyo?')

    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[
                    Part(root=TextPart(text='Let me check.')),
                    Part(root=ToolRequestPart(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo'))),
                ],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.TOOL,
                index=1,
                content=[
                    Part(root=ToolResponsePart(tool_response=ToolResponse(name='weather', ref='c1', output='12C')))
                ],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL, index=2, content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]
            )
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(finish_reason=AgentFinishReason.STOP)))
    await turn.response

    # Same stitched shape as the server-managed tool loop: the tool steps are
    # present even though raw.state never carried them.
    assert [m.role for m in chat.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    tool_req = chat.messages[1].content[1].root
    assert isinstance(tool_req, ToolRequestPart)
    assert tool_req.tool_request.name == 'weather'
    assert chat.messages[-1].content[0].root.text == 'It is 12C in Tokyo.'
    # Custom is adopted from the round-tripped output.
    assert chat.state == {'unit': 'celsius'}
    assert chat.session_id == 'sess-client-1'
    # The running view is what ships back as state for a client-managed resume.
    init_state = chat._wire_init().state
    assert init_state is not None
    assert init_state.messages == chat.messages
    assert init_state.session_id == 'sess-client-1'


@pytest.mark.asyncio
async def test_server_managed_running_view_matches_snapshot_over_real_tool_loop() -> None:
    """Against the real in-process runtime, a server-managed turn's running view
    rebuilt from chunks must line up with the authoritative store snapshot."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    store = InMemorySessionStore()

    @ai.tool()
    async def weather(city: str) -> str:
        return '12C'

    ai.define_prompt(name='weatherAgent', model='programmableModel', system='Use the weather tool.', tools=[weather])
    agent = ai.define_prompt_agent(name='weatherAgent', store=store)

    # Turn 1: model calls the tool; turn 2: model answers with the tool result.
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo')))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunkModel(
                role=Role.MODEL,
                content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo')))],
            )
        ],
        [ModelResponseChunkModel(role=Role.MODEL, content=[Part(root=TextPart(text='It is 12C in Tokyo.'))])],
    ]

    chat = agent.chat()
    await chat.send('Weather in Tokyo?')

    # The running view carries the whole turn, not just user + final reply.
    assert [m.role for m in chat.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    call_req = chat.messages[1].content[0].root
    assert isinstance(call_req, ToolRequestPart)
    assert call_req.tool_request.name == 'weather'
    reply_resp = chat.messages[2].content[0].root
    assert isinstance(reply_resp, ToolResponsePart)
    assert reply_resp.tool_response.output == '12C'
    assert chat.messages[3].content[0].root.text == 'It is 12C in Tokyo.'

    # And it matches the durable store snapshot the server actually persisted.
    snapshot = await chat.get_snapshot()
    assert snapshot is not None
    assert snapshot.state is not None
    assert [m.role for m in (snapshot.state.messages or [])] == [m.role for m in chat.messages]


@pytest.mark.asyncio
async def test_server_managed_failed_turn_rolls_back_optimistic_user_message() -> None:
    """A failed server-managed turn returns no reply; the optimistically appended
    user message is rolled back so it isn't left stranded in the local view."""
    transport = MockAgentTransport(state_management='server')
    chat = AgentChat(transport)

    transport.final_output = AgentOutput(
        snapshot_id='snap-good',
        finish_reason=AgentFinishReason.FAILED,
    )
    turn = chat.send_stream('U1')
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-good', finish_reason=AgentFinishReason.FAILED))
    )
    with pytest.raises(AgentError) as exc_info:
        await turn.response

    assert exc_info.value.snapshot_id == 'snap-good'
    assert chat.messages == []
    assert chat.snapshot_id == 'snap-good'


@pytest.mark.asyncio
async def test_no_store_inprocess_transport_assembles_output_message() -> None:
    """InProcessTransport must return a complete AgentOutput even without a session store."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    pm.chunks = [
        [
            ModelResponseChunkModel(role=Role.MODEL, content=[Part(root=TextPart(text='Hi '))]),
            ModelResponseChunkModel(role=Role.MODEL, content=[Part(root=TextPart(text='there!'))]),
        ]
    ]
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Hi there!'))]),
            finish_reason=FinishReason.STOP,
        )
    )

    agent = ai.define_agent(name='noStoreAgent', model='programmableModel', system='Reply briefly.')
    chat = agent.chat()
    out = await chat.send('Hello')

    assert out.text == 'Hi there!'
    assert len(chat.messages) == 2
    assert chat.messages[1].content[0].root.text == 'Hi there!'
    assert chat.session_id is not None
    assert chat.snapshot_id is None

    # You assemble the resume blob yourself from the chat's tracked fields.
    saved = SessionState(
        session_id=chat.session_id,
        messages=chat.messages,
        custom=chat.state,
        artifacts=chat.artifacts,
    )
    assert saved.messages == chat.messages
    assert saved.custom == chat.state
    assert saved.session_id == chat.session_id


class _ServerEmulatingClientManagedTransport(AgentTransport[Any]):
    """Stateless client-managed transport that mimics the real server round-trip.

    On each turn it loads history from ``init.state``, appends the turn's input
    message and a model reply, then echoes the full state back — the same path
    that would duplicate a message if the client also bundled it into ``init``.
    """

    def __init__(self) -> None:
        self.state_management: StateManagement = 'client'
        self.init_histories: list[list[str]] = []
        self._model_turn = 0

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        loaded = list(init.state.messages or []) if init.state else []
        self.init_histories.append([
            root.text
            for m in loaded
            for part in (m.content or [])
            if isinstance((root := getattr(part, 'root', part)), TextPart) and root.text
        ])

        if agent_input.message:
            loaded.append(agent_input.message)
        self._model_turn += 1
        model_msg = MessageData(role='model', content=[Part(root=TextPart(text=f'reply-{self._model_turn}'))])
        loaded.append(model_msg)
        server_state = SessionState(messages=loaded)

        async def _gen() -> AsyncIterator[AgentStreamChunk]:
            yield AgentStreamChunk(turn_end=TurnEnd(finish_reason=AgentFinishReason.STOP))

        async def _out() -> AgentOutput:
            return AgentOutput(finish_reason=AgentFinishReason.STOP, message=model_msg, state=server_state)

        return _gen(), _out()

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        return None

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_client_managed_does_not_double_append_messages() -> None:
    """Client-managed init carries prior history only; the server appends the new message."""
    transport = _ServerEmulatingClientManagedTransport()
    chat = AgentChat(transport, AgentInit())

    await chat.send('hello')
    # The new message must NOT ride along in init — the server records it from input.
    assert transport.init_histories[0] == []
    assert [m.content[0].root.text for m in chat.messages] == ['hello', 'reply-1']

    await chat.send('again')
    # Turn 2's init replays the prior two messages, never the message in flight.
    assert transport.init_histories[1] == ['hello', 'reply-1']
    assert [m.content[0].root.text for m in chat.messages] == ['hello', 'reply-1', 'again', 'reply-2']


@pytest.mark.asyncio
async def test_session_id_populated_from_output_state() -> None:
    """The server assigns the session id on the first turn; the client must adopt it.

    A server-managed turn carries the id on the output itself, never inside a
    round-tripped state blob (the store owns the state)."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        session_id='session_abc',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Done.'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    # Fresh session with no init, so it starts without a session id.
    chat = AgentChat(transport)
    assert chat.session_id is None

    turn = chat.send_stream('Hello')
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )
    await turn.response

    assert chat.session_id == 'session_abc'


@pytest.mark.asyncio
async def test_session_handling_tool_interrupt() -> None:
    transport = MockAgentTransport()

    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        finish_reason=AgentFinishReason.INTERRUPTED,
        message=MessageData(
            role='model',
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(name='userApproval', ref='call_1', input={'amount': 500}),
                        metadata={'interrupt': True},
                    )
                )
            ],
        ),
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Approve $500 transfer')

    # Queue up a tool request chunk representing an interrupt
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='userApproval', ref='call_1', input={'amount': 500})
                        )
                    )
                ]
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.INTERRUPTED))
    )

    out = await turn.response

    assert len(out.interrupts) == 1
    assert out.interrupts[0].name == 'userApproval'
    assert out.interrupts[0].ref == 'call_1'
    assert out.interrupts[0].input == {'amount': 500}

    # Acknowledge the interrupt and trigger response turn
    # This mock resume expects sending tool response to transport
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_2',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Transfer done.'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    resume_turn = chat.resume_stream(respond=[out.interrupts[0].respond({'approved': True})])

    # Queue up turn_end for the resume turn
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_2', finish_reason=AgentFinishReason.STOP))
    )

    # Consume resume turn stream to trigger execution
    async for _chunk in resume_turn.stream:
        pass

    # Verify transport received the ToolResponse payload
    assert len(transport.send_payloads) == 2
    sent_resume = transport.send_payloads[1].resume
    assert sent_resume is not None
    assert sent_resume.respond is not None
    assert sent_resume.respond[0].tool_response.name == 'userApproval'
    assert sent_resume.respond[0].tool_response.output == {'approved': True}


@pytest.mark.asyncio
async def test_session_handling_multiple_tool_interrupts() -> None:
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        finish_reason=AgentFinishReason.INTERRUPTED,
        message=MessageData(
            role='model',
            content=[
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(name='transferA', ref='ra', input={'amount': 100}),
                        metadata={'interrupt': True},
                    )
                ),
                Part(
                    root=ToolRequestPart(
                        tool_request=ToolRequest(name='transferB', ref='rb', input={'amount': 200}),
                        metadata={'interrupt': True},
                    )
                ),
            ],
        ),
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Transfer to two accounts')

    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                content=[
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='transferA', ref='ra', input={'amount': 100})
                        )
                    ),
                    Part(
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='transferB', ref='rb', input={'amount': 200})
                        )
                    ),
                ]
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.INTERRUPTED))
    )

    out = await turn.response

    assert len(out.interrupts) == 2
    assert {i.name for i in out.interrupts} == {'transferA', 'transferB'}

    transport.final_output = AgentOutput(
        snapshot_id='snapshot_2',
        finish_reason=AgentFinishReason.STOP,
    )
    restart_parts = [intr.restart(resumed_metadata={'tool_approved': True}) for intr in out.interrupts]
    resume_turn = chat.resume_stream(restart=restart_parts)
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_2', finish_reason=AgentFinishReason.STOP))
    )
    await resume_turn.response

    sent_resume = transport.send_payloads[1].resume
    assert sent_resume is not None
    assert sent_resume.restart is not None
    assert len(sent_resume.restart) == 2
    assert {p.tool_request.name for p in sent_resume.restart} == {'transferA', 'transferB'}


@pytest.mark.asyncio
async def test_in_process_persistent_connection() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    store = InMemorySessionStore()

    ai.define_prompt(name='testEchoAgent', model='programmableModel', system='You echo things.')
    agent = ai.define_prompt_agent(name='testEchoAgent', store=store)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='Echo 1'))]),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='Echo 2'))]),
        )
    )

    chat = agent.chat()
    # Turn 1
    turn1 = chat.send_stream('Hello')
    chunks1 = []
    async for chunk in turn1.stream:
        chunks1.append(chunk)
    res1 = await turn1.response
    assert res1.message is not None
    assert res1.message.content is not None
    assert res1.message.content[0].root.text == 'Echo 1'

    # Turn 2
    turn2 = chat.send_stream('World')
    chunks2 = []
    async for chunk in turn2.stream:
        chunks2.append(chunk)
    res2 = await turn2.response
    assert res2.message is not None
    assert res2.message.content is not None
    assert res2.message.content[0].root.text == 'Echo 2'


@pytest.mark.asyncio
async def test_attached_turn_abort() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    store = InMemorySessionStore()

    # Define a simple agent
    ai.define_prompt(name='abortAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='abortAgent', store=store)

    # We make the mock model sleep to simulate a slow response
    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Slow response finished'))]),
        )

    pm.response_cb = slow_response

    chat = agent.chat()
    turn = chat.send_stream('Hello')

    # Let it run a bit
    await asyncio.sleep(0.1)

    # Abort the turn client-side (stops reading the stream)
    await turn.abort()

    # Verify awaiting the turn raises CancelledError
    with pytest.raises(asyncio.CancelledError):
        await turn.response

    # Abort is a client-side detach only: the prompt was still asked, so the
    # optimistic user message stays in history (just without a reply).
    texts_after_abort = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
    assert texts_after_abort == ['Hello']

    # Restore normal fast response for the second turn
    pm.response_cb = None
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Second turn echo'))]),
        )
    )

    # We can keep going; the next turn appends onto the kept history.
    turn2 = chat.send_stream('Continue conversation')
    res2 = await turn2.response

    # The detached turn's 'Hello' is still there, followed by the new exchange.
    texts = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
    assert texts == ['Hello', 'Continue conversation', 'Second turn echo']
    assert res2.message is not None
    assert res2.message.content is not None
    assert res2.message.content[0].root.text == 'Second turn echo'


@pytest.mark.asyncio
async def test_await_turn_under_timeout_detaches() -> None:
    """A deadline around `await turn.response` detaches like turn.abort(): the deadline
    surfaces as TimeoutError, the prompt stays in history, and the next turn works."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(name='timeoutAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='timeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='too late'))]),
        )

    pm.response_cb = slow_response

    chat = agent.chat()
    turn = chat.send_stream('Hello')

    # The deadline fires before the slow model responds → surfaces as TimeoutError.
    async def _await_turn() -> None:
        await turn.response

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_await_turn(), 0.2)

    # Detach kept the optimistic prompt; the session reads as a turn with no reply.
    texts_after = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
    assert texts_after == ['Hello']

    # And we can continue cleanly.
    pm.response_cb = None
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Second turn echo'))]),
        )
    )
    res2 = await chat.send('Continue conversation')
    assert res2.message is not None
    assert res2.message.content[0].root.text == 'Second turn echo'


@pytest.mark.asyncio
async def test_stream_turn_under_timeout_detaches() -> None:
    """A deadline around `async for chunk in turn` detaches the same way."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(name='streamTimeoutAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='streamTimeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='too late'))]),
        )

    pm.response_cb = slow_response

    chat = agent.chat()
    turn = chat.send_stream('Hello')

    async def _drain() -> None:
        async for _chunk in turn.stream:
            pass

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_drain(), 0.2)

    texts_after = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
    assert texts_after == ['Hello']


@pytest.mark.asyncio
async def test_session_abort() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    store = InMemorySessionStore()

    tool_executed = False
    tool_cancelled = False

    @ai.tool()
    async def slow_tool(arg: str) -> str:
        nonlocal tool_executed, tool_cancelled
        tool_executed = True
        try:
            await asyncio.sleep(10)
            return 'Slow tool complete'
        except asyncio.CancelledError:
            tool_cancelled = True
            raise

    # Define a simple agent that uses this tool
    ai.define_prompt(
        name='sessionAbortAgent', model='programmableModel', system='Use the slow tool.', tools=[slow_tool]
    )
    agent = ai.define_prompt_agent(name='sessionAbortAgent', store=store)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(
                        root=ToolRequestPart(tool_request=ToolRequest(name='slow_tool', ref='call_1', input='blocking'))
                    )
                ],
            ),
        )
    )

    chat = agent.chat()
    # Start a detached turn to get a snapshot ID on the server
    task = await chat.detach('Trigger slow action')
    assert task.snapshot_id is not None

    # Give it a tiny moment to start execution
    await asyncio.sleep(0.2)

    # Abort the running snapshot on the server (requires a store)
    status = await chat.abort()
    # abort() returns the snapshot's *previous* status — pending, since the
    # detached turn was still running (spec: tests/specs/agent.yaml).
    assert status == SnapshotStatus.PENDING

    # Give the background task a moment to process cancellation
    await asyncio.sleep(0.5)

    # Verify the tool was started and successfully cancelled by the server abort!
    assert tool_executed
    assert tool_cancelled


@pytest.mark.asyncio
async def test_session_abort_without_snapshot_raises() -> None:
    ai = Genkit()
    define_programmable_model(ai)

    # No store → client-managed → there's never a server snapshot to abort.
    ai.define_prompt(name='noStoreAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='noStoreAgent')

    chat = agent.chat()
    with pytest.raises(ValueError, match='No active snapshot to abort'):
        await chat.abort()


@pytest.mark.asyncio
async def test_agent_turn_direct_async_iteration() -> None:
    """Tests that AgentTurn itself can be directly iterated over to consume stream chunks (DX feature)."""
    transport = MockAgentTransport()

    # Configure final output
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    # Queue up chunks
    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Weather is '))]))
    )
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Sunny.'))])))
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    # Consume chunks by iterating directly over the turn!
    chunks = []
    async for chunk in turn.stream:
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].text == 'Weather is '
    assert chunks[1].text == 'Sunny.'
    assert chunks[2].text is None

    # Verify we can still await the turn after streaming
    output = await turn.response
    assert output.message is not None
    assert output.message.content is not None
    assert output.message.content[0].root.text == 'Final output!'


@pytest.mark.asyncio
async def test_agent_turn_direct_await() -> None:
    """Awaiting the turn itself runs it to completion and returns the final response."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='ignored chunk'))]))
    )
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    # Awaiting the turn alone drives it to completion — no need to iterate first.
    output = await turn.response

    assert output.message is not None
    assert output.message.content is not None
    assert output.message.content[0].root.text == 'Final output!'


@pytest.mark.asyncio
async def test_agent_turn_stream_and_response_accessors() -> None:
    """`turn.stream` yields the chunks and `turn.response` resolves the result.

    Genkit's other streaming handles expose these, so a turn offers the same
    surface. Both route through the detach-on-cancel wrappers."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Weather is '))]))
    )
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Sunny.'))])))
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    chunks = [chunk async for chunk in turn.stream]
    assert [c.text for c in chunks] == ['Weather is ', 'Sunny.', None]

    output = await turn.response
    assert output.message is not None
    assert output.message.content[0].root.text == 'Final output!'


# ---------------------------------------------------------------------------
# TurnDriver background error surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_background_resolves_output_when_on_turn_error_raises() -> None:
    """If ``on_turn_error`` itself raises, ``await turn.response`` must not hang."""

    async def boom_run_turn(
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        raise RuntimeError('transport failed')

    def broken_on_turn_error(e: Exception) -> Exception:
        raise RuntimeError('on_turn_error failed') from e

    driver = TurnDriver(
        inp=AgentInput(),
        init=AgentInit(),
        run_turn=boom_run_turn,
        commit_output=lambda _raw: (_ for _ in ()).throw(AssertionError('commit_output should not run')),
        commit_custom_patch=lambda _patch: None,
        on_turn_error=broken_on_turn_error,
        chunks=CloseableQueue(),
    )
    turn = driver.start()

    with pytest.raises(RuntimeError, match='on_turn_error failed'):
        await asyncio.wait_for(turn.response, timeout=1.0)


# ---------------------------------------------------------------------------
# Undrained stream / cross-turn isolation
#
# Each send_stream/resume_stream owns its own caller-facing chunk queue. The
# transport is always pumped (patches + stitching), so awaiting .response
# without reading chunks is fine — unread chunks stay on that turn handle and
# must not show up on the next turn's stream.
# ---------------------------------------------------------------------------


async def text_chunks(turn: Any) -> list[str]:
    return [c.text async for c in turn if c.text]


def text_chunk(text: str) -> AgentStreamChunk:
    return AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text=text))]))


def stop_end(snapshot_id: str) -> AgentStreamChunk:
    return AgentStreamChunk(turn_end=TurnEnd(snapshot_id=snapshot_id, finish_reason=AgentFinishReason.STOP))


def interrupt_tool_part() -> Part:
    return Part(
        root=ToolRequestPart(
            tool_request=ToolRequest(name='userApproval', ref='c1', input={'amount': 1}),
            metadata={'interrupt': True},
        )
    )


class PerTurnMockTransport(AgentTransport[Any]):
    """Fixture transport: each ``run_turn`` pops its own preloaded chunk queue."""

    def __init__(self) -> None:
        self.state_management: StateManagement = 'server'
        self.queues: list[asyncio.Queue[AgentStreamChunk | None]] = []
        self.finals: list[AgentOutput] = []

    def enqueue(
        self,
        *,
        chunks: list[AgentStreamChunk],
        output: AgentOutput,
    ) -> None:
        q: asyncio.Queue[AgentStreamChunk | None] = asyncio.Queue()
        for chunk in chunks:
            q.put_nowait(chunk)
        q.put_nowait(None)
        self.queues.append(q)
        self.finals.append(output)

    def enqueue_text_turn(self, *, texts: list[str], snapshot_id: str, final_text: str) -> None:
        self.enqueue(
            chunks=[*(text_chunk(t) for t in texts), stop_end(snapshot_id)],
            output=AgentOutput(
                snapshot_id=snapshot_id,
                message=MessageData(role='model', content=[Part(root=TextPart(text=final_text))]),
                finish_reason=AgentFinishReason.STOP,
            ),
        )

    def enqueue_interrupt_turn(self, *, snapshot_id: str) -> None:
        part = interrupt_tool_part()
        self.enqueue(
            chunks=[
                AgentStreamChunk(model_chunk=ModelResponseChunk(content=[part])),
                AgentStreamChunk(
                    turn_end=TurnEnd(snapshot_id=snapshot_id, finish_reason=AgentFinishReason.INTERRUPTED)
                ),
            ],
            output=AgentOutput(
                snapshot_id=snapshot_id,
                message=MessageData(role='model', content=[part]),
                finish_reason=AgentFinishReason.INTERRUPTED,
            ),
        )

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        q = self.queues.pop(0)
        final = self.finals.pop(0)

        async def generator() -> AsyncIterator[AgentStreamChunk]:
            while True:
                chunk = await q.get()
                if chunk is None:
                    break
                yield chunk

        async def output_waiter() -> AgentOutput:
            return final

        return generator(), output_waiter()

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        return SnapshotStatus.ABORTED


@pytest.mark.asyncio
async def test_await_response_without_reading_chunks() -> None:
    """``await turn.response`` works with zero chunk reads; late drain still sees them."""
    transport = PerTurnMockTransport()
    transport.enqueue_text_turn(texts=['Alpha-', 'one'], snapshot_id='s1', final_text='Alpha-one')
    chat = AgentChat(transport)

    turn = chat.send_stream('u1')
    assert (await turn.response).text == 'Alpha-one'
    assert await text_chunks(turn) == ['Alpha-', 'one']


@pytest.mark.asyncio
async def test_unread_chunks_do_not_appear_on_next_turn() -> None:
    """Unread turn-1 chunks must not show up when iterating turn 2."""
    transport = PerTurnMockTransport()
    transport.enqueue_text_turn(texts=['TURN1-A', 'TURN1-B'], snapshot_id='s1', final_text='t1')
    transport.enqueue_text_turn(texts=['TURN2-A', 'TURN2-B'], snapshot_id='s2', final_text='t2')
    chat = AgentChat(transport)

    turn1 = chat.send_stream('first')
    await turn1.response  # leave turn1's chunk queue unread

    turn2 = chat.send_stream('second')
    assert await text_chunks(turn2) == ['TURN2-A', 'TURN2-B']
    assert (await turn2.response).text == 't2'

    # Late drain of turn1 is still only turn1.
    assert await text_chunks(turn1) == ['TURN1-A', 'TURN1-B']


@pytest.mark.asyncio
async def test_partial_stream_read_then_await_keeps_remaining_on_same_turn() -> None:
    """Reading one chunk, then awaiting response, leaves the rest on that turn."""
    transport = PerTurnMockTransport()
    transport.enqueue_text_turn(texts=['p1', 'p2', 'p3'], snapshot_id='s1', final_text='all')
    chat = AgentChat(transport)

    turn = chat.send_stream('go')
    stream = turn.stream.__aiter__()
    assert (await stream.__anext__()).text == 'p1'

    assert (await turn.response).text == 'all'
    assert [c.text async for c in stream if c.text] == ['p2', 'p3']


@pytest.mark.asyncio
async def test_send_applies_custom_patches_without_caller_stream() -> None:
    """Internal pump applies patches even when the caller never reads chunks."""
    transport = MockAgentTransport()
    chat = AgentChat(transport)

    transport.final_output = AgentOutput(
        snapshot_id='snap-a',
        message=MessageData(role='model', content=[Part(root=TextPart(text='hi'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('hello')
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(
                root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'mark': 'from-stream'})]
            )
        )
    )
    transport.push_chunk(text_chunk('hi'))
    transport.push_chunk(stop_end('snap-a'))
    await turn.response
    assert chat.state == {'mark': 'from-stream'}

    transport.final_output = AgentOutput(
        snapshot_id='snap-b',
        message=MessageData(role='model', content=[Part(root=TextPart(text='again'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    send_task = asyncio.create_task(chat.send('next'))
    await asyncio.sleep(0)  # let send() block on the mock receive queue
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/mark', value='from-send')])
        )
    )
    transport.push_chunk(stop_end('snap-b'))
    assert (await send_task).text == 'again'
    assert chat.state == {'mark': 'from-send'}


@pytest.mark.asyncio
async def test_undrained_resume_stream_does_not_leak_into_next_turn() -> None:
    transport = PerTurnMockTransport()
    transport.enqueue_interrupt_turn(snapshot_id='s1')
    transport.enqueue_text_turn(texts=['RESUME-ONLY'], snapshot_id='s2', final_text='resumed')
    transport.enqueue_text_turn(texts=['AFTER'], snapshot_id='s3', final_text='after')
    chat = AgentChat(transport)

    interrupted = await chat.send_stream('transfer').response
    assert interrupted.interrupts

    resume_turn = chat.resume_stream(respond=[interrupted.interrupts[0].respond({'approved': True})])
    assert (await resume_turn.response).text == 'resumed'  # unread resume chunks

    assert await text_chunks(chat.send_stream('follow-up')) == ['AFTER']
    assert await text_chunks(resume_turn) == ['RESUME-ONLY']


@pytest.mark.asyncio
async def test_inprocess_undrained_streams_stay_isolated() -> None:
    """Same isolation with a real in-process agent (programmable model)."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)
    ai.define_prompt(name='isoAgent', model='programmableModel', system='echo')
    agent = ai.define_prompt_agent(name='isoAgent', store=InMemorySessionStore())

    pm.chunks = [
        [
            ModelResponseChunkModel(content=[Part(TextPart(text='ONE-A'))]),
            ModelResponseChunkModel(content=[Part(TextPart(text='ONE-B'))]),
        ],
        [
            ModelResponseChunkModel(content=[Part(TextPart(text='TWO-A'))]),
            ModelResponseChunkModel(content=[Part(TextPart(text='TWO-B'))]),
        ],
    ]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='ONE-FINAL'))]),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='TWO-FINAL'))]),
        ),
    ]

    chat = agent.chat()
    turn1 = chat.send_stream('first')
    assert (await turn1.response).text == 'ONE-FINAL'

    turn2 = chat.send_stream('second')
    assert await text_chunks(turn2) == ['TWO-A', 'TWO-B']
    assert (await turn2.response).text == 'TWO-FINAL'
    assert await text_chunks(turn1) == ['ONE-A', 'ONE-B']
