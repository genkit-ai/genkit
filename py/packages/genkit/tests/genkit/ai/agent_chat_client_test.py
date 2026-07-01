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
    AgentInterrupt,
    AgentTransport,
)
from genkit._ai._agents._types import StateManagement
from genkit._ai._aio import Genkit
from genkit._ai._json_patch import apply_json_patch
from genkit._ai._testing import define_programmable_model
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

# ---------------------------------------------------------------------------
# Unit tests for JSON patch application
# ---------------------------------------------------------------------------


def test_apply_json_patch_root_replace() -> None:
    patch = [JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'status': 'idle', 'score': 10})]
    res = apply_json_patch(None, patch)
    assert res == {'status': 'idle', 'score': 10}


def test_apply_json_patch_nested_replace() -> None:
    doc = {'status': 'idle', 'nested': {'value': 1}}
    patch = [JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/nested/value', value=2)]
    res = apply_json_patch(doc, patch)
    assert res == {'status': 'idle', 'nested': {'value': 2}}


def test_apply_json_patch_array_add() -> None:
    doc = {'items': [1, 2]}
    patch = [JsonPatchOperation(op=JsonPatchOp.ADD, path='/items/-', value=3)]
    res = apply_json_patch(doc, patch)
    assert res == {'items': [1, 2, 3]}


def test_apply_json_patch_array_remove() -> None:
    doc = {'items': [1, 2, 3]}
    patch = [JsonPatchOperation(op=JsonPatchOp.REMOVE, path='/items/1')]
    res = apply_json_patch(doc, patch)
    assert res == {'items': [1, 3]}


# ---------------------------------------------------------------------------
# Mock Transport for Testing Stateful Connections
# ---------------------------------------------------------------------------


class MockAgentTransport(AgentTransport[dict[str, Any]]):
    def __init__(self, *, state_management: StateManagement = 'server') -> None:
        self.connect_init: AgentInit | None = None
        self.send_payloads: list[AgentInput] = []
        self.final_output: AgentOutput | None = None
        self.close_called: bool = False
        self.abort_snapshot_id: str | None = None
        self.state_management: StateManagement = state_management
        self._receive_queue: asyncio.Queue[AgentStreamChunk | None] = asyncio.Queue()

    async def run_turn(
        self,
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
        return SnapshotStatus.ABORTED

    async def close(self) -> None:
        self.close_called = True

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
    with pytest.raises(ValueError, match='at most one'):
        AgentChat(
            MockAgentTransport(),
            AgentInit(state=SessionState(), snapshot_id='snap-1'),
        )


def test_connect_init_applies_state_only() -> None:
    state = SessionState(session_id='sess-1', custom={'x': 1})
    chat = AgentChat(MockAgentTransport(), AgentInit(state=state))

    assert chat.session_id == 'sess-1'
    assert chat.state == {'x': 1}
    assert chat.snapshot_id is None


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

    turn = chat.send('Hello')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn

    # First turn (no snapshot yet) resumes by the bootstrap session id.
    assert transport.connect_init == AgentInit(session_id='sess-bootstrap')
    # Output advanced the live snapshot id, so the next turn would resume by snapshot.
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
    turn = chat.send('Weather in Tokyo?')

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
    async for chunk in turn:
        chunks.append(chunk)

    assert len(chunks) == 4
    assert chunks[0].text == 'Weather is '
    assert chunks[1].text == 'Sunny.'
    assert chunks[2].text is None

    # Verify custom state patch applied
    assert chat.state == {'unit': 'celsius'}

    # Await output to verify final response resolved correctly
    output = await turn
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
    turn = chat.send('go')
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='', value={'turns': 1})])
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))

    streamed = [chunk.custom async for chunk in turn if chunk.custom is not None]
    res = await turn

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
    turn = chat.send('go')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    res = await turn

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
    turn = chat.send('U1')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn

    assert chat.snapshot_id == 'snap-1'
    assert [m.content[0].root.text for m in chat.messages] == ['U1', 'A1']

    transport.final_output = AgentOutput(
        snapshot_id='snap-2',
        message=MessageData(role='model', content=[Part(root=TextPart(text='A2'))]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn2 = chat.send('U2')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-2', finish_reason=AgentFinishReason.STOP)))
    await turn2

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
    turn = chat.send('Weather in Tokyo?')

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
    await turn

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
    # A stray session_id here must be ignored: client-managed has no server store
    # to key a session on, so the id stays None.
    transport.final_output = AgentOutput(
        message=MessageData(role='model', content=[Part(root=TextPart(text='It is 12C in Tokyo.'))]),
        state=SessionState(session_id='ignored', custom={'unit': 'celsius'}),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send('Weather in Tokyo?')

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
    await turn

    # Same stitched shape as the server-managed tool loop: the tool steps are
    # present even though raw.state never carried them.
    assert [m.role for m in chat.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    tool_req = chat.messages[1].content[1].root
    assert isinstance(tool_req, ToolRequestPart)
    assert tool_req.tool_request.name == 'weather'
    assert chat.messages[-1].content[0].root.text == 'It is 12C in Tokyo.'
    # Custom is adopted from the round-tripped output.
    assert chat.state == {'unit': 'celsius'}
    # session_id stays None for client-managed, even when the output carries one.
    assert chat.session_id is None
    # The running view is what ships back as state for a client-managed resume.
    init_state = chat._wire_init().state
    assert init_state is not None
    assert init_state.messages == chat.messages


@pytest.mark.asyncio
async def test_server_managed_running_view_matches_snapshot_over_real_tool_loop() -> None:
    """Against the real in-process runtime, a server-managed turn's running view
    rebuilt from chunks must line up with the authoritative store snapshot."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

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

    async with agent.chat() as chat:
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
    turn = chat.send('U1')
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-good', finish_reason=AgentFinishReason.FAILED))
    )
    await turn

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
    assert chat.session_id is None
    assert chat.snapshot_id is None

    # You assemble the resume blob yourself from the chat's tracked fields.
    saved = SessionState(messages=chat.messages, custom=chat.state, artifacts=chat.artifacts)
    assert saved.messages == chat.messages
    assert saved.custom == chat.state


class _ServerEmulatingClientManagedTransport(AgentTransport[dict[str, Any]]):
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

    turn = chat.send('Hello')
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )
    await turn

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
    turn = chat.send('Approve $500 transfer')

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

    out = await turn

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

    resume_turn = chat.resume(respond=[out.interrupts[0].respond({'approved': True})])

    # Queue up turn_end for the resume turn
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_2', finish_reason=AgentFinishReason.STOP))
    )

    # Consume resume turn stream to trigger execution
    async for _chunk in resume_turn:
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
    turn = chat.send('Transfer to two accounts')

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

    out = await turn

    assert len(out.interrupts) == 2
    assert {i.name for i in out.interrupts} == {'transferA', 'transferB'}

    transport.final_output = AgentOutput(
        snapshot_id='snapshot_2',
        finish_reason=AgentFinishReason.STOP,
    )
    restart_parts = [intr.restart(resumed_metadata={'tool_approved': True}) for intr in out.interrupts]
    resume_turn = chat.resume(restart=restart_parts)
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_2', finish_reason=AgentFinishReason.STOP))
    )
    await resume_turn

    sent_resume = transport.send_payloads[1].resume
    assert sent_resume is not None
    assert sent_resume.restart is not None
    assert len(sent_resume.restart) == 2
    assert {p.tool_request.name for p in sent_resume.restart} == {'transferA', 'transferB'}


@pytest.mark.asyncio
async def test_session_context_manager_autocloses() -> None:
    transport = MockAgentTransport()

    async with AgentChat(transport):
        assert not transport.close_called

    assert transport.close_called


@pytest.mark.asyncio
async def test_in_process_persistent_connection() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

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

    async with agent.chat() as chat:
        # Turn 1
        turn1 = chat.send('Hello')
        chunks1 = []
        async for chunk in turn1:
            chunks1.append(chunk)
        res1 = await turn1
        assert res1.message is not None
        assert res1.message.content is not None
        assert res1.message.content[0].root.text == 'Echo 1'

        # Turn 2
        turn2 = chat.send('World')
        chunks2 = []
        async for chunk in turn2:
            chunks2.append(chunk)
        res2 = await turn2
        assert res2.message is not None
        assert res2.message.content is not None
        assert res2.message.content[0].root.text == 'Echo 2'


@pytest.mark.asyncio
async def test_attached_turn_abort() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

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

    async with agent.chat() as chat:
        turn = chat.send('Hello')

        # Let it run a bit
        await asyncio.sleep(0.1)

        # Abort the turn client-side (stops reading the stream)
        await turn.abort()

        # Verify awaiting the turn raises CancelledError
        with pytest.raises(asyncio.CancelledError):
            await turn

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
        turn2 = chat.send('Continue conversation')
        res2 = await turn2

        # The detached turn's 'Hello' is still there, followed by the new exchange.
        texts = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
        assert texts == ['Hello', 'Continue conversation', 'Second turn echo']
        assert res2.message is not None
        assert res2.message.content is not None
        assert res2.message.content[0].root.text == 'Second turn echo'


@pytest.mark.asyncio
async def test_await_turn_under_timeout_detaches() -> None:
    """A deadline around `await turn` detaches like turn.abort(): the deadline
    surfaces as TimeoutError, the prompt stays in history, and the next turn works."""
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

    ai.define_prompt(name='timeoutAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='timeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='too late'))]),
        )

    pm.response_cb = slow_response

    async with agent.chat() as chat:
        turn = chat.send('Hello')

        # The deadline fires before the slow model responds → surfaces as TimeoutError.
        async def _await_turn() -> None:
            await turn

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

    from genkit.agent import InMemorySessionStore

    ai.define_prompt(name='streamTimeoutAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='streamTimeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='too late'))]),
        )

    pm.response_cb = slow_response

    async with agent.chat() as chat:
        turn = chat.send('Hello')

        async def _drain() -> None:
            async for _chunk in turn:
                pass

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_drain(), 0.2)

        texts_after = [p.root.text for m in chat.messages for p in (m.content or []) if hasattr(p.root, 'text')]
        assert texts_after == ['Hello']


@pytest.mark.asyncio
async def test_session_abort() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

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

    async with agent.chat() as chat:
        # Start a detached turn to get a snapshot ID on the server
        task = await chat.detach('Trigger slow action')
        assert task.snapshot_id is not None

        # Give it a tiny moment to start execution
        await asyncio.sleep(0.2)

        # Abort the running snapshot on the server (requires a store)
        status = await chat.abort()
        assert status == SnapshotStatus.ABORTED

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
    turn = chat.send('Weather in Tokyo?')

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
    async for chunk in turn:
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].text == 'Weather is '
    assert chunks[1].text == 'Sunny.'
    assert chunks[2].text is None

    # Verify we can still await the turn after streaming
    output = await turn
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
    turn = chat.send('Weather in Tokyo?')

    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='ignored chunk'))]))
    )
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    # Awaiting the turn alone drives it to completion — no need to iterate first.
    output = await turn

    assert output.message is not None
    assert output.message.content is not None
    assert output.message.content[0].root.text == 'Final output!'
