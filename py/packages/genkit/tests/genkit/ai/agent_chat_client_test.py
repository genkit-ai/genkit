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
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import Part
from genkit._ai._agents._base import Agent, define_custom_agent
from genkit._ai._agents._client import (
    AgentChat,
    AgentClient,
    AgentInterrupt,
    AgentTransport,
    DetachedTask,
    TurnDriver,
)
from genkit._ai._agents._runtime import AgentInitError, SessionRunner
from genkit._ai._agents._types import StateManagement, TurnResult
from genkit._ai._json_patch import apply_json_patch
from genkit._core._channel import CloseableQueue
from genkit._core._error import GenkitError, GenkitRuntimeError, RuntimeErrorReason
from genkit._core._model import (
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    Message,
    ModelResponse,
    ModelResponseChunk,
    ModelResponseChunk as ModelResponseChunkModel,
    SessionSnapshot,
    SessionState,
)
from genkit._core._typing import (
    AgentFinishReason,
    FinishReason,
    JsonPatch,
    JsonPatchOp,
    JsonPatchOperation,
    Role,
    SnapshotStatus,
    ToolRequest,
    ToolResponse,
    TurnEnd,
)
from genkit.exp import Genkit
from genkit.exp.agent import AgentContext, InMemorySessionStore
from genkit.testing import define_scripted_model

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


def test_agent_interrupt_restart_has_no_replace_input() -> None:
    """intr.restart(replace_input=...) is a TypeError; the helper does not take a new input."""
    intr = AgentInterrupt('transfer', 'ref-1', {'amount': 100})
    with pytest.raises(TypeError, match='replace_input'):
        intr.restart(replace_input={'amount': 50})  # type: ignore[call-arg]


def test_restart_with_no_args_marks_resumed() -> None:
    """intr.restart() marks the tool request resumed."""
    intr = AgentInterrupt('transfer', 'ref-1', {'amount': 100})
    part = intr.restart()
    assert part.tool_request is not None
    assert part.tool_request.input == {'amount': 100}
    assert part.metadata is not None
    assert part.metadata.get('resumed') is True


def test_respond_carries_output_only() -> None:
    """intr.respond(output) carries the output and no interruptResponse metadata."""
    intr = AgentInterrupt('transfer', 'ref-1', {'amount': 100})
    part = intr.respond({'approved': True})
    assert part.tool_response is not None
    assert part.tool_response.output == {'approved': True}
    assert part.metadata is None


def test_chat_resume_rejects_paused_part() -> None:
    """chat.resume(restart=[paused]) raises before the turn is sent."""
    chat = AgentChat(MockAgentTransport())
    paused = Part.from_tool_request(name='pay', ref='r1', input={}, metadata={'interrupt': True})
    with pytest.raises(GenkitError, match='still an interrupt'):
        chat.resume_stream(restart=[paused])


def test_chat_resume_respond_with_paused_part_raises_before_send() -> None:
    """chat.resume_stream(respond=[paused]) is INVALID_ARGUMENT before anything is sent."""
    transport = MockAgentTransport()
    chat = AgentChat(transport)
    paused = Part.from_tool_request(name='pay', ref='r1', input={}, metadata={'interrupt': True})
    with pytest.raises(GenkitError, match='Part.respond') as exc:
        chat.resume_stream(respond=[paused])
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert transport.send_payloads == []


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
        message=Message(role='model', content=[Part.from_text('Hi')]),
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
        message=Message(role='model', content=[Part.from_text('Final output!')]),
        state=SessionState(
            messages=[
                Message(role='user', content=[Part.from_text('Weather in Tokyo?')]),
                Message(role='model', content=[Part.from_text('Final output!')]),
            ],
            custom={'unit': 'celsius'},
        ),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    # Queue up chunks to simulate streaming
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Weather is ')])))
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Sunny.')])))
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
    assert output.message.content[0].text == 'Final output!'

    # Verify chat fields are updated after turn completion
    assert chat.snapshot_id == 'snapshot_1'
    assert len(chat.messages) == 2  # Turn 1 User input + model final output
    assert chat.messages[0].content[0].text == 'Weather in Tokyo?'
    assert chat.messages[1].content[0].text == 'Final output!'


class _Progress(BaseModel):
    turns: int = 0


@pytest.mark.asyncio
async def test_state_schema_coerces_custom_into_model() -> None:
    """With a state_schema the live state, streamed patch, and response materialize the model."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snap-1',
        message=Message(role='model', content=[Part.from_text('ok')]),
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
        message=Message(role='model', content=[Part.from_text('ok')]),
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
        message=Message(role='model', content=[Part.from_text('A1')]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('U1')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn.response

    assert chat.snapshot_id == 'snap-1'
    assert [m.content[0].text for m in chat.messages] == ['U1', 'A1']

    transport.final_output = AgentOutput(
        snapshot_id='snap-2',
        message=Message(role='model', content=[Part.from_text('A2')]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn2 = chat.send_stream('U2')
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-2', finish_reason=AgentFinishReason.STOP)))
    await turn2.response

    assert chat.snapshot_id == 'snap-2'
    assert [m.content[0].text for m in chat.messages] == ['U1', 'A1', 'U2', 'A2']


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
        message=Message(role='model', content=[Part.from_text('It is 12C in Tokyo.')]),
        finish_reason=AgentFinishReason.STOP,
    )
    turn = chat.send_stream('Weather in Tokyo?')

    # Model message that calls a tool, streamed as text deltas + a tool request.
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[Part.from_text('Let me ')],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.MODEL,
                index=0,
                content=[
                    Part.from_text('check.'),
                    Part(tool_request=ToolRequest(name='weather', ref='c1', input={'city': 'Tokyo'})),
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
                content=[Part(tool_response=ToolResponse(name='weather', ref='c1', output='12C'))],
            )
        )
    )
    # Final model message, streamed as text deltas (superseded by raw.message).
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(role=Role.MODEL, index=2, content=[Part.from_text('It is 12C in Tokyo.')])
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snap-1', finish_reason=AgentFinishReason.STOP)))
    await turn.response

    roles = [m.role for m in chat.messages]
    assert roles == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]

    # User input, the tool-calling model message (deltas merged + tool request),
    # the tool reply, then the authoritative final reply.
    user_msg, tool_call_msg, tool_reply_msg, final_msg = chat.messages
    assert user_msg.content[0].text == 'Weather in Tokyo?'
    assert tool_call_msg.content[0].text == 'Let me check.'
    tool_req = tool_call_msg.content[1]
    assert tool_req.tool_request is not None
    assert tool_req.tool_request.name == 'weather'
    tool_resp = tool_reply_msg.content[0]
    assert tool_resp.tool_response is not None
    assert tool_resp.tool_response.output == '12C'
    assert final_msg.content[0].text == 'It is 12C in Tokyo.'


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
        message=Message(role='model', content=[Part.from_text('It is 12C in Tokyo.')]),
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
                    Part.from_text('Let me check.'),
                    Part(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo')),
                ],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(
                role=Role.TOOL,
                index=1,
                content=[Part(tool_response=ToolResponse(name='weather', ref='c1', output='12C'))],
            )
        )
    )
    transport.push_chunk(
        AgentStreamChunk(
            model_chunk=ModelResponseChunk(role=Role.MODEL, index=2, content=[Part.from_text('It is 12C in Tokyo.')])
        )
    )
    transport.push_chunk(AgentStreamChunk(turn_end=TurnEnd(finish_reason=AgentFinishReason.STOP)))
    await turn.response

    # Same stitched shape as the server-managed tool loop: the tool steps are
    # present even though raw.state never carried them.
    assert [m.role for m in chat.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    tool_req = chat.messages[1].content[1]
    assert tool_req.tool_request is not None
    assert tool_req.tool_request.name == 'weather'
    assert chat.messages[-1].content[0].text == 'It is 12C in Tokyo.'
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
    pm, _ = define_scripted_model(ai)

    store = InMemorySessionStore()

    @ai.tool()
    async def weather(city: str) -> str:
        return '12C'

    ai.define_prompt(name='weatherAgent', model='scriptedModel', system='Use the weather tool.', tools=[weather])
    agent = ai.define_prompt_agent(name='weatherAgent', store=store)

    # Turn 1: model calls the tool; turn 2: model answers with the tool result.
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo'))],
            ),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('It is 12C in Tokyo.')]),
        )
    )
    pm.chunks = [
        [
            ModelResponseChunkModel(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='weather', ref='c1', input='Tokyo'))],
            )
        ],
        [ModelResponseChunkModel(role=Role.MODEL, content=[Part.from_text('It is 12C in Tokyo.')])],
    ]

    chat = agent.chat()
    await chat.send('Weather in Tokyo?')

    # The running view carries the whole turn, not just user + final reply.
    assert [m.role for m in chat.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    call_req = chat.messages[1].content[0]
    assert call_req.tool_request is not None
    assert call_req.tool_request.name == 'weather'
    reply_resp = chat.messages[2].content[0]
    assert reply_resp.tool_response is not None
    assert reply_resp.tool_response.output == '12C'
    assert chat.messages[3].content[0].text == 'It is 12C in Tokyo.'

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
    res = await turn.response

    assert res.finish_reason == AgentFinishReason.FAILED
    assert res.snapshot_id == 'snap-good'
    assert chat.messages == []
    assert chat.snapshot_id == 'snap-good'


@pytest.mark.asyncio
async def test_no_store_inprocess_transport_assembles_output_message() -> None:
    """InProcessTransport must return a complete AgentOutput even without a session store."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    pm.chunks = [
        [
            ModelResponseChunkModel(role=Role.MODEL, content=[Part.from_text('Hi ')]),
            ModelResponseChunkModel(role=Role.MODEL, content=[Part.from_text('there!')]),
        ]
    ]
    pm.responses.append(
        ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text('Hi there!')]),
            finish_reason=FinishReason.STOP,
        )
    )

    agent = ai.define_agent(name='noStoreAgent', model='scriptedModel', system='Reply briefly.')
    chat = agent.chat()
    out = await chat.send('Hello')

    assert out.text == 'Hi there!'
    assert len(chat.messages) == 2
    assert chat.messages[1].content[0].text == 'Hi there!'
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
        self.init_histories.append([part.text for m in loaded for part in (m.content or []) if part.text])

        if agent_input.message:
            loaded.append(agent_input.message)
        self._model_turn += 1
        model_msg = Message(role='model', content=[Part.from_text(f'reply-{self._model_turn}')])
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
    assert [m.content[0].text for m in chat.messages] == ['hello', 'reply-1']

    await chat.send('again')
    # Turn 2's init replays the prior two messages, never the message in flight.
    assert transport.init_histories[1] == ['hello', 'reply-1']
    assert [m.content[0].text for m in chat.messages] == ['hello', 'reply-1', 'again', 'reply-2']


@pytest.mark.asyncio
async def test_session_id_populated_from_output_state() -> None:
    """The server assigns the session id on the first turn; the client must adopt it.

    A server-managed turn carries the id on the output itself, never inside a
    round-tripped state blob (the store owns the state)."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        session_id='session_abc',
        message=Message(role='model', content=[Part.from_text('Done.')]),
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
        message=Message(
            role='model',
            content=[
                Part(
                    tool_request=ToolRequest(name='userApproval', ref='call_1', input={'amount': 500}),
                    metadata={'interrupt': True},
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
                content=[Part(tool_request=ToolRequest(name='userApproval', ref='call_1', input={'amount': 500}))]
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
        message=Message(role='model', content=[Part.from_text('Transfer done.')]),
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
    assert sent_resume.respond[0].tool_response is not None
    assert sent_resume.respond[0].tool_response.name == 'userApproval'
    assert sent_resume.respond[0].tool_response.output == {'approved': True}


@pytest.mark.asyncio
async def test_session_handling_multiple_tool_interrupts() -> None:
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        finish_reason=AgentFinishReason.INTERRUPTED,
        message=Message(
            role='model',
            content=[
                Part(
                    tool_request=ToolRequest(name='transferA', ref='ra', input={'amount': 100}),
                    metadata={'interrupt': True},
                ),
                Part(
                    tool_request=ToolRequest(name='transferB', ref='rb', input={'amount': 200}),
                    metadata={'interrupt': True},
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
                    Part(tool_request=ToolRequest(name='transferA', ref='ra', input={'amount': 100})),
                    Part(tool_request=ToolRequest(name='transferB', ref='rb', input={'amount': 200})),
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
    assert {p.tool_request.name for p in sent_resume.restart if p.tool_request is not None} == {
        'transferA',
        'transferB',
    }


@pytest.mark.asyncio
async def test_in_process_persistent_connection() -> None:
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    store = InMemorySessionStore()

    ai.define_prompt(name='testEchoAgent', model='scriptedModel', system='You echo things.')
    agent = ai.define_prompt_agent(name='testEchoAgent', store=store)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Echo 1')]),
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Echo 2')]),
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
    assert res1.message.content[0].text == 'Echo 1'

    # Turn 2
    turn2 = chat.send_stream('World')
    chunks2 = []
    async for chunk in turn2.stream:
        chunks2.append(chunk)
    res2 = await turn2.response
    assert res2.message is not None
    assert res2.message.content is not None
    assert res2.message.content[0].text == 'Echo 2'


@pytest.mark.asyncio
async def test_attached_turn_abort() -> None:
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    store = InMemorySessionStore()

    # Define a simple agent
    ai.define_prompt(name='abortAgent', model='scriptedModel', system='Hello')
    agent = ai.define_prompt_agent(name='abortAgent', store=store)

    # We make the mock model sleep to simulate a slow response
    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Slow response finished')]),
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
    texts_after_abort = [p.text for m in chat.messages for p in (m.content or []) if p.text is not None]
    assert texts_after_abort == ['Hello']

    # Restore normal fast response for the second turn
    pm.response_cb = None
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Second turn echo')]),
        )
    )

    # We can keep going; the next turn appends onto the kept history.
    turn2 = chat.send_stream('Continue conversation')
    res2 = await turn2.response

    # The detached turn's 'Hello' is still there, followed by the new exchange.
    texts = [p.text for m in chat.messages for p in (m.content or []) if p.text is not None]
    assert texts == ['Hello', 'Continue conversation', 'Second turn echo']
    assert res2.message is not None
    assert res2.message.content is not None
    assert res2.message.content[0].text == 'Second turn echo'


@pytest.mark.asyncio
async def test_await_turn_under_timeout_detaches() -> None:
    """A deadline around `await turn.response` detaches like turn.abort(): the deadline
    surfaces as TimeoutError, the prompt stays in history, and the next turn works."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    ai.define_prompt(name='timeoutAgent', model='scriptedModel', system='Hello')
    agent = ai.define_prompt_agent(name='timeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('too late')]),
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
    texts_after = [p.text for m in chat.messages for p in (m.content or []) if p.text is not None]
    assert texts_after == ['Hello']

    # And we can continue cleanly.
    pm.response_cb = None
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('Second turn echo')]),
        )
    )
    res2 = await chat.send('Continue conversation')
    assert res2.message is not None
    assert res2.message.content[0].text == 'Second turn echo'


@pytest.mark.asyncio
async def test_stream_turn_under_timeout_detaches() -> None:
    """A deadline around `async for chunk in turn` detaches the same way."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    ai.define_prompt(name='streamTimeoutAgent', model='scriptedModel', system='Hello')
    agent = ai.define_prompt_agent(name='streamTimeoutAgent', store=InMemorySessionStore())

    async def slow_response(*args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('too late')]),
        )

    pm.response_cb = slow_response

    chat = agent.chat()
    turn = chat.send_stream('Hello')

    async def _drain() -> None:
        async for _chunk in turn.stream:
            pass

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_drain(), 0.2)

    texts_after = [p.text for m in chat.messages for p in (m.content or []) if p.text is not None]
    assert texts_after == ['Hello']


@pytest.mark.asyncio
async def test_session_abort() -> None:
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

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
    ai.define_prompt(name='sessionAbortAgent', model='scriptedModel', system='Use the slow tool.', tools=[slow_tool])
    agent = ai.define_prompt_agent(name='sessionAbortAgent', store=store)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='slow_tool', ref='call_1', input='blocking'))],
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
    define_scripted_model(ai)

    # No store → client-managed → there's never a server snapshot to abort.
    ai.define_prompt(name='noStoreAgent', model='scriptedModel', system='Hello')
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
        message=Message(role='model', content=[Part.from_text('Final output!')]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    # Queue up chunks
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Weather is ')])))
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Sunny.')])))
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
    assert output.message.content[0].text == 'Final output!'


@pytest.mark.asyncio
async def test_agent_turn_direct_await() -> None:
    """Awaiting the turn itself runs it to completion and returns the final response."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=Message(role='model', content=[Part.from_text('Final output!')]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('ignored chunk')])))
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    # Awaiting the turn alone drives it to completion — no need to iterate first.
    output = await turn.response

    assert output.message is not None
    assert output.message.content is not None
    assert output.message.content[0].text == 'Final output!'


@pytest.mark.asyncio
async def test_agent_turn_stream_and_response_accessors() -> None:
    """`turn.stream` yields the chunks and `turn.response` resolves the result.

    Genkit's other streaming handles expose these, so a turn offers the same
    surface. Both route through the detach-on-cancel wrappers."""
    transport = MockAgentTransport()
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=Message(role='model', content=[Part.from_text('Final output!')]),
        finish_reason=AgentFinishReason.STOP,
    )

    chat = AgentChat(transport)
    turn = chat.send_stream('Weather in Tokyo?')

    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Weather is ')])))
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Sunny.')])))
    transport.push_chunk(
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    chunks = [chunk async for chunk in turn.stream]
    assert [c.text for c in chunks] == ['Weather is ', 'Sunny.', None]

    output = await turn.response
    assert output.message is not None
    assert output.message.content[0].text == 'Final output!'


# ---------------------------------------------------------------------------
# TurnDriver background error surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_background_resolves_output_when_rollback_raises() -> None:
    """If ``rollback`` itself raises, ``await turn.response`` must not hang."""

    async def boom_run_turn(
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        raise RuntimeError('transport failed')

    def broken_rollback() -> None:
        raise RuntimeError('rollback failed')

    driver = TurnDriver(
        inp=AgentInput(),
        init=AgentInit(),
        run_turn=boom_run_turn,
        commit_output=lambda _raw: (_ for _ in ()).throw(AssertionError('commit_output should not run')),
        commit_custom_patch=lambda _patch: None,
        rollback=broken_rollback,
        chunks=CloseableQueue(),
    )
    turn = driver.start()

    with pytest.raises(RuntimeError, match='rollback failed'):
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
    return AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text(text)]))


def stop_end(snapshot_id: str) -> AgentStreamChunk:
    return AgentStreamChunk(turn_end=TurnEnd(snapshot_id=snapshot_id, finish_reason=AgentFinishReason.STOP))


def interrupt_tool_part() -> Part:
    return Part(
        tool_request=ToolRequest(name='userApproval', ref='c1', input={'amount': 1}), metadata={'interrupt': True}
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
                message=Message(role='model', content=[Part.from_text(final_text)]),
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
                message=Message(role='model', content=[part]),
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
        message=Message(role='model', content=[Part.from_text('hi')]),
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
        message=Message(role='model', content=[Part.from_text('again')]),
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
    """Same isolation with a real in-process agent (scripted model)."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)
    ai.define_prompt(name='isoAgent', model='scriptedModel', system='echo')
    agent = ai.define_prompt_agent(name='isoAgent', store=InMemorySessionStore())

    pm.chunks = [
        [
            ModelResponseChunkModel(content=[Part.from_text('ONE-A')]),
            ModelResponseChunkModel(content=[Part.from_text('ONE-B')]),
        ],
        [
            ModelResponseChunkModel(content=[Part.from_text('TWO-A')]),
            ModelResponseChunkModel(content=[Part.from_text('TWO-B')]),
        ],
    ]
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('ONE-FINAL')]),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('TWO-FINAL')]),
        ),
    ]

    chat = agent.chat()
    turn1 = chat.send_stream('first')
    assert (await turn1.response).text == 'ONE-FINAL'

    turn2 = chat.send_stream('second')
    assert await text_chunks(turn2) == ['TWO-A', 'TWO-B']
    assert (await turn2.response).text == 'TWO-FINAL'
    assert await text_chunks(turn1) == ['ONE-A', 'ONE-B']


# ---------------------------------------------------------------------------
# A failed turn returns AgentResponse.error instead of raising
# ---------------------------------------------------------------------------


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
    ai: Genkit,
    name: str,
    handle: Callable[[SessionRunner, AgentInput], Awaitable[TurnResult | None]],
    *,
    store: InMemorySessionStore | None,
    state_schema: type[BaseModel] | None = None,
) -> Agent:
    async def fn(session_runner: SessionRunner, _: AgentContext) -> AgentResult:
        async def turn(inp: AgentInput, __: Any) -> TurnResult | None:  # noqa: ANN401
            return await handle(session_runner, inp)

        await session_runner.run(turn)
        return await session_runner.result()

    return define_custom_agent(ai, name, fn, store=store, state_schema=state_schema)


def _flaky(
    ai: Genkit,
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

    return _define_turns(ai, name, handle, store=store)


def _assert_failed(res: Any, *, message: str) -> GenkitRuntimeError:  # noqa: ANN401
    assert res.finish_reason == AgentFinishReason.FAILED
    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.message == message
    assert res.message is None
    assert res.text == ''
    assert res.data is None
    assert res.media is None
    assert res.reasoning == ''
    assert res.tool_requests == []
    assert res.interrupts == []
    return res.error


class _DroppingTransport(AgentTransport[Any]):
    """Dies before a turn result exists."""

    state_management: StateManagement = 'client'

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


class _MidStreamDropTransport(_DroppingTransport):
    """Streams a reply chunk and a custom patch, then the connection dies."""

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[Any], Awaitable[AgentOutput]]:
        del agent_input, init

        async def chunks() -> AsyncIterator[AgentStreamChunk]:
            yield AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part.from_text('Booking for ')]))
            yield AgentStreamChunk(
                custom_patch=JsonPatch(root=[JsonPatchOperation(op=JsonPatchOp.REPLACE, path='/n', value=99)])
            )
            raise ConnectionError('socket closed mid-turn')

        # Never resolves: the stream dies before a turn result exists.
        output: asyncio.Future[AgentOutput] = asyncio.get_running_loop().create_future()
        return chunks(), output


@pytest.mark.asyncio
async def test_send_on_failed_turn_with_reason_reads_reason() -> None:
    """A failed turn whose error carries a reason exposes it on res.error.reason."""
    agent = _flaky(
        Genkit(),
        'failReason',
        store=InMemorySessionStore(),
        fail=GenkitError(status='INTERNAL', message='tool died', reason=RuntimeErrorReason.TOOL_FAILED),
    )
    res = await agent.chat().send('please fail now')
    err = _assert_failed(res, message='tool died')
    assert err.status == 'INTERNAL'
    assert err.reason is RuntimeErrorReason.TOOL_FAILED


@pytest.mark.asyncio
async def test_send_on_failed_turn_keeps_last_good_snapshot_id() -> None:
    """res.snapshot_id and chat.snapshot_id both stay on the last completed snapshot."""
    agent = _flaky(Genkit(), 'failSnap', store=InMemorySessionStore())
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
    agent = _flaky(Genkit(), 'failDrop', store=InMemorySessionStore())
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

    agent = _define_turns(Genkit(), 'failState', handle, store=None)
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
    agent = _flaky(Genkit(), 'failContinue', store=InMemorySessionStore())
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
    agent = _flaky(Genkit(), 'failStream', store=InMemorySessionStore())
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
    agent = _flaky(Genkit(), 'failResume', store=InMemorySessionStore())
    chat = agent.chat()
    paused = await chat.send('please pause')
    assert paused.finish_reason == AgentFinishReason.INTERRUPTED
    assert paused.interrupts

    res = await chat.resume(respond=[paused.interrupts[0].respond({'approved': True})])
    _assert_failed(res, message='boom')
    # An approve loop keyed on res.interrupts would otherwise retry forever.
    assert res.interrupts == []


@pytest.mark.asyncio
async def test_send_on_successful_turn_has_no_error() -> None:
    """A normal turn returns res.error is None."""
    agent = _flaky(Genkit(), 'okTurn', store=InMemorySessionStore())
    res = await agent.chat().send('hello')
    assert res.finish_reason == AgentFinishReason.STOP
    assert res.error is None
    assert res.text == 'ok'
    assert res.message is not None


@pytest.mark.asyncio
async def test_send_on_blocked_turn_returns_blocked_without_raising() -> None:
    """A blocked turn still returns with finish_reason == 'blocked'."""
    agent = _flaky(Genkit(), 'blockedTurn', store=InMemorySessionStore())
    res = await agent.chat().send('please block this')
    assert res.finish_reason == AgentFinishReason.BLOCKED
    assert res.error is None


@pytest.mark.asyncio
async def test_send_with_snapshot_id_on_agent_without_store_raises_genkit_error() -> None:
    """An init the server rejects as misuse raises GenkitError with FAILED_PRECONDITION."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Genkit(), 'noStore', handle, store=None)
    chat = agent.chat(snapshot_id='snap-1')
    with pytest.raises(GenkitError) as exc:
        await chat.send('hi')
    assert exc.value.status == 'FAILED_PRECONDITION'
    assert exc.value.reason is RuntimeErrorReason.SESSION_STORE_NOT_CONFIGURED
    # Same type the client raises for client-side misuse; no wrapper in between.
    assert isinstance(exc.value, AgentInitError)
    assert chat.messages == []


async def _chat_after_one_good_turn(name: str) -> AgentChat[Any]:
    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del inp
        await session_runner.update_custom(lambda _prev: {'n': 1})
        await session_runner.add_messages([_ok_message()])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Genkit(), name, handle, store=None)
    chat = agent.chat(state={'n': 0})
    ok = await chat.send('hello')
    assert ok.state == {'n': 1}
    return chat


@pytest.mark.asyncio
async def test_send_over_dropped_connection_raises_unavailable_and_keeps_last_good_state() -> None:
    """No turn result: GenkitError(UNAVAILABLE), and the chat stays at the last good turn."""
    chat = await _chat_after_one_good_turn('dropConn')
    messages_before = list(chat.messages)

    chat._transport = _DroppingTransport()  # noqa: SLF001
    with pytest.raises(GenkitError) as exc:
        await chat.send('next')
    assert exc.value.status == 'UNAVAILABLE'
    assert isinstance(exc.value.cause, ConnectionError)
    assert chat.state == {'n': 1}
    assert chat.messages == messages_before


@pytest.mark.asyncio
async def test_send_mid_stream_drop_rolls_back_prompt_and_streamed_patch() -> None:
    """A drop after a streamed patch undoes the patch and the unanswered prompt."""
    chat = await _chat_after_one_good_turn('midDrop')
    messages_before = list(chat.messages)

    chat._transport = _MidStreamDropTransport()  # noqa: SLF001
    with pytest.raises(GenkitError) as exc:
        await chat.send('book a table for 4')
    assert exc.value.status == 'UNAVAILABLE'
    assert chat.state == {'n': 1}
    assert chat.messages == messages_before


@pytest.mark.asyncio
async def test_send_stream_mid_stream_drop_raises_genkit_error_on_stream_and_response() -> None:
    """Streaming surfaces the same GenkitError on .stream and .response, with the same rollback."""
    chat = await _chat_after_one_good_turn('midDropStream')
    messages_before = list(chat.messages)

    chat._transport = _MidStreamDropTransport()  # noqa: SLF001
    turn = chat.send_stream('book a table for 4')
    seen: list[str] = []
    with pytest.raises(GenkitError) as stream_exc:
        async for chunk in turn.stream:
            seen.append(chunk.text or '')
    with pytest.raises(GenkitError) as response_exc:
        await turn.response
    assert seen[0] == 'Booking for '
    assert stream_exc.value is response_exc.value
    assert response_exc.value.status == 'UNAVAILABLE'
    assert chat.state == {'n': 1}
    assert chat.messages == messages_before


class _BadFinalStateTransport(_DroppingTransport):
    """Completes the turn, but the final custom state fails the chat's schema."""

    async def run_turn(
        self,
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[Any], Awaitable[AgentOutput]]:
        del agent_input, init

        async def no_chunks() -> AsyncIterator[AgentStreamChunk]:
            empty: list[AgentStreamChunk] = []
            for chunk in empty:
                yield chunk

        output: asyncio.Future[AgentOutput] = asyncio.get_running_loop().create_future()
        output.set_result(
            AgentOutput(
                finish_reason=AgentFinishReason.STOP,
                snapshot_id='snap-2',
                message=_ok_message(),
                state=SessionState(
                    session_id='sess-2',
                    custom={'title': 'Dinner for 4', 'done': 'not-a-bool'},
                    artifacts=[Artifact(name='receipt', parts=[Part.from_text('total: $80')])],
                ),
            )
        )
        return no_chunks(), output


@pytest.mark.asyncio
async def test_send_when_final_state_fails_schema_restores_every_field() -> None:
    """commit_output raises after folding in the output; the chat still goes fully back.

    Restoring only messages and custom would leave snapshot_id, session_id,
    and artifacts on the new turn, so the next send would resume from a
    snapshot the local history doesn't have.
    """
    chat = AgentChat(
        _BadFinalStateTransport(),
        AgentInit(state=SessionState(session_id='sess-1', custom={'title': 'Dinner for 2'})),
        state_schema=_TaskState,
    )
    chat._snapshot_id = 'snap-1'  # noqa: SLF001
    chat._resume_snapshot_id = 'snap-1'  # noqa: SLF001
    messages = chat.messages

    with pytest.raises(GenkitError):
        await chat.send('make it 4')

    assert chat.messages == []
    assert chat.messages is messages
    assert chat.state == _TaskState(title='Dinner for 2')
    assert chat.session_id == 'sess-1'
    assert chat.snapshot_id == 'snap-1'
    assert chat._resume_snapshot_id == 'snap-1'  # noqa: SLF001
    assert chat.artifacts == []


@pytest.mark.asyncio
async def test_send_with_invalid_custom_state_returns_response_with_error() -> None:
    """A custom state that fails the schema comes back on res.error, not as a raise."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    agent = _define_turns(Genkit(), 'badState', handle, store=None, state_schema=_TaskState)
    res = await agent.chat(state={'done': 'nope'}).send('hi')  # type: ignore[arg-type]
    assert res.finish_reason == AgentFinishReason.FAILED
    assert isinstance(res.error, GenkitRuntimeError)
    assert res.error.reason is RuntimeErrorReason.INVALID_INPUT
    assert res.error.status == 'INVALID_ARGUMENT'
    assert res.state is None


@pytest.mark.asyncio
async def test_send_on_failed_turn_after_data_reply_has_no_data() -> None:
    """res.data on a failed turn is None, not the previous turn's data."""
    agent = _flaky(
        Genkit(),
        'failData',
        store=InMemorySessionStore(),
        reply=_ok_message([Part.from_text('ok'), Part.from_data({'n': 1})]),
    )
    chat = agent.chat()
    ok = await chat.send('hello')
    assert ok.data == {'n': 1}

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')


@pytest.mark.asyncio
async def test_send_on_failed_turn_without_error_still_has_error() -> None:
    """A custom agent that returns FAILED without raising still reads as failed on res.error."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        del session_runner, inp
        return TurnResult(finish_reason=AgentFinishReason.FAILED)

    agent = _define_turns(Genkit(), 'silentFail', handle, store=InMemorySessionStore())
    res = await agent.chat().send('hi')

    err = _assert_failed(res, message='Agent turn failed.')
    assert err.status == 'UNKNOWN'


@pytest.mark.asyncio
async def test_send_on_failed_server_managed_turn_rolls_back_custom_patches() -> None:
    """Custom patches streamed during a failed turn are undone, matching the resume snapshot."""

    async def handle(session_runner: SessionRunner, inp: AgentInput) -> TurnResult | None:
        if 'fail' in _input_text(inp).lower():
            await session_runner.update_custom(lambda _prev: {'n': 999})
            raise GenkitError(status='INTERNAL', message='boom')
        await session_runner.update_custom(lambda _prev: {'n': 1})
        await session_runner.add_messages([_ok_message()])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    store = InMemorySessionStore()
    agent = _define_turns(Genkit(), 'failPatch', handle, store=store)
    chat = agent.chat()
    ok = await chat.send('hello')
    assert ok.state == {'n': 1}

    res = await chat.send('please fail now')
    _assert_failed(res, message='boom')
    assert res.state == {'n': 1}
    assert chat.state == {'n': 1}
    snapshot = await store.get_snapshot(snapshot_id=res.snapshot_id)
    assert snapshot is not None
    assert snapshot.state is not None
    assert snapshot.state.custom == {'n': 1}


class _ScriptedTransport(AgentTransport[Any]):
    """Hands back canned turn outputs, like a remote server would."""

    def __init__(self, outputs: list[AgentOutput], *, state_management: StateManagement) -> None:
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
    chat = AgentClient(transport).chat()
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
    chat = AgentClient(transport, state_schema=_TaskState).chat()

    res = await chat.send('mark the order done')

    assert res.finish_reason == AgentFinishReason.ABORTED
    assert res.state is None


@pytest.mark.asyncio
async def test_send_on_failed_turn_without_snapshot_falls_back_to_last_completed() -> None:
    """A failed turn with no id reports the last completed snapshot, not an earlier failed row."""
    # 1. s1 completes, then a failed turn writes its own row f1.
    # 2. The next failed turn carries no id at all.
    transport = _ScriptedTransport(
        [
            AgentOutput(finish_reason=AgentFinishReason.STOP, snapshot_id='s1', message=_ok_message()),
            AgentOutput(finish_reason=AgentFinishReason.FAILED, snapshot_id='f1'),
            AgentOutput(finish_reason=AgentFinishReason.FAILED),
        ],
        state_management='server',
    )
    chat = AgentClient(transport).chat()
    await chat.send('suggest a dish')
    await chat.send('something without nuts')

    res = await chat.send('something without dairy')

    # 3. f1 can't be resumed, so the caller is pointed at s1.
    assert res.finish_reason == AgentFinishReason.FAILED
    assert res.snapshot_id == 's1'
