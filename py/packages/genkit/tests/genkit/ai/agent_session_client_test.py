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
from collections.abc import AsyncIterable, Awaitable
from typing import Any

import pytest

from genkit._ai._agents._client import (
    AgentSession,
    AgentTransport,
)
from genkit._ai._aio import Genkit
from genkit._ai._json_patch import apply_json_patch
from genkit._ai._testing import define_programmable_model
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    FinishReason,
    JsonPatch,
    JsonPatchOperation,
    MessageData,
    ModelResponseChunk,
    Part,
    Role,
    SessionSnapshot,
    SnapshotStatus,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    TurnEnd,
)

# ---------------------------------------------------------------------------
# Unit tests for JSON patch application
# ---------------------------------------------------------------------------


def test_apply_json_patch_root_replace() -> None:
    patch = [JsonPatchOperation(op='replace', path='', value={'status': 'idle', 'score': 10})]
    res = apply_json_patch(None, patch)
    assert res == {'status': 'idle', 'score': 10}


def test_apply_json_patch_nested_replace() -> None:
    doc = {'status': 'idle', 'nested': {'value': 1}}
    patch = [JsonPatchOperation(op='replace', path='/nested/value', value=2)]
    res = apply_json_patch(doc, patch)
    assert res == {'status': 'idle', 'nested': {'value': 2}}


def test_apply_json_patch_array_add() -> None:
    doc = {'items': [1, 2]}
    patch = [JsonPatchOperation(op='add', path='/items/-', value=3)]
    res = apply_json_patch(doc, patch)
    assert res == {'items': [1, 2, 3]}


def test_apply_json_patch_array_remove() -> None:
    doc = {'items': [1, 2, 3]}
    patch = [JsonPatchOperation(op='remove', path='/items/1')]
    res = apply_json_patch(doc, patch)
    assert res == {'items': [1, 3]}


# ---------------------------------------------------------------------------
# Mock Transport for Testing Stateful Connections
# ---------------------------------------------------------------------------


class MockAgentTransport(AgentTransport[dict[str, Any], str]):
    def __init__(self) -> None:
        self.connect_init = None
        self.send_payloads = []
        self.final_output = None
        self.close_called = False
        self.abort_snapshot_id = None
        self._receive_queue = asyncio.Queue()

    async def run_turn(
        self,
        input: AgentInput,
        init: AgentInit[dict[str, Any]],
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput[dict[str, Any]]]]:
        self.connect_init = init
        self.send_payloads.append(input)

        async def _generator():
            while True:
                chunk = await self._receive_queue.get()
                if chunk is None:
                    break
                yield chunk

        async def _output_waiter():
            return self.final_output

        return _generator(), _output_waiter()

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot[dict[str, Any]] | None:
        return None

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        self.abort_snapshot_id = snapshot_id
        return SnapshotStatus.ABORTED

    async def close(self) -> None:
        self.close_called = True

    def push_chunk(self, chunk: AgentStreamChunk | None) -> None:
        self._receive_queue.put_nowait(chunk)


# ---------------------------------------------------------------------------
# Turn and Session Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_sends_input_and_aggregates_state() -> None:
    transport = MockAgentTransport()

    # Configure final output the transport will resolve with
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_1',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Final output!'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    session = AgentSession(transport)
    turn = session.send('Weather in Tokyo?')

    # Queue up chunks to simulate streaming
    transport.push_chunk(
        AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Weather is '))]))
    )
    transport.push_chunk(AgentStreamChunk(model_chunk=ModelResponseChunk(content=[Part(root=TextPart(text='Sunny.'))])))
    transport.push_chunk(
        AgentStreamChunk(
            custom_patch=JsonPatch(root=[JsonPatchOperation(op='replace', path='', value={'unit': 'celsius'})])
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
    assert session.state == {'unit': 'celsius'}

    # Await output to verify final response resolved correctly
    output = await turn.output
    assert output.finish_reason == AgentFinishReason.STOP
    assert output.message.content[0].root.text == 'Final output!'

    # Verify session fields are updated after turn completion
    assert session.snapshot_id == 'snapshot_1'
    assert len(session.messages) == 2  # Turn 1 User input + model final output
    assert session.messages[0].content[0].root.text == 'Weather in Tokyo?'
    assert session.messages[1].content[0].root.text == 'Final output!'


@pytest.mark.asyncio
async def test_session_handling_tool_interrupt() -> None:
    transport = MockAgentTransport()

    transport.final_output = AgentOutput(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP)

    session = AgentSession(transport)
    turn = session.send('Approve $500 transfer')

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
        AgentStreamChunk(turn_end=TurnEnd(snapshot_id='snapshot_1', finish_reason=AgentFinishReason.STOP))
    )

    async for _chunk in turn.stream:
        pass

    # Verify interrupt is caught on the turn
    assert turn.interrupt is not None
    assert turn.interrupt.name == 'userApproval'
    assert turn.interrupt.ref == 'call_1'
    assert turn.interrupt.input == {'amount': 500}

    # Acknowledge the interrupt and trigger response turn
    # This mock resume expects sending tool response to transport
    transport.final_output = AgentOutput(
        snapshot_id='snapshot_2',
        message=MessageData(role='model', content=[Part(root=TextPart(text='Transfer done.'))]),
        finish_reason=AgentFinishReason.STOP,
    )

    resume_turn = turn.interrupt.respond({'approved': True})

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
    assert sent_resume.respond[0].tool_response.name == 'userApproval'
    assert sent_resume.respond[0].tool_response.output == {'approved': True}


@pytest.mark.asyncio
async def test_session_context_manager_autocloses() -> None:
    transport = MockAgentTransport()

    async with AgentSession(transport):
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

    async with agent.connect() as session:
        # Turn 1
        turn1 = session.send('Hello')
        chunks1 = []
        async for chunk in turn1.stream:
            chunks1.append(chunk)
        res1 = await turn1.output
        assert res1.message.content[0].root.text == 'Echo 1'

        # Turn 2
        turn2 = session.send('World')
        chunks2 = []
        async for chunk in turn2.stream:
            chunks2.append(chunk)
        res2 = await turn2.output
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
    async def slow_response(*args, **kwargs):
        await asyncio.sleep(5)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Slow response finished'))]),
        )

    pm.response_cb = slow_response

    async with agent.connect() as session:
        turn = session.send('Hello')

        # Let it run a bit
        await asyncio.sleep(0.1)

        # Abort the turn client-side (stops reading the stream)
        await turn.abort()

        # Verify turn.output raises CancelledError
        with pytest.raises(asyncio.CancelledError):
            await turn.output

        # Restore normal fast response for the second turn
        pm.response_cb = None
        pm.responses.append(
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Second turn echo'))]),
            )
        )

        # We should be able to cleanly send a new turn and continue the conversation!
        turn2 = session.send('Continue conversation')
        res2 = await turn2.output
        assert res2.message.content[0].root.text == 'Second turn echo'


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
                        root=ToolRequestPart(
                            tool_request=ToolRequest(name='slow_tool', ref='call_1', input='blocking')
                        )
                    )
                ],
            ),
        )
    )

    async with agent.connect() as session:
        # Start a detached turn to get a snapshot ID on the server
        task = await session.run_detached('Trigger slow action')
        assert task.snapshot_id is not None

        # Give it a tiny moment to start execution
        await asyncio.sleep(0.2)

        # Abort the running snapshot on the server (requires a store)
        status = await session.abort()
        assert status == SnapshotStatus.ABORTED

        # Give the background task a moment to process cancellation
        await asyncio.sleep(0.5)

        # Verify the tool was started and successfully cancelled by the server abort!
        assert tool_executed
        assert tool_cancelled


@pytest.mark.asyncio
async def test_external_abort_signal() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    from genkit.agent import InMemorySessionStore

    store = InMemorySessionStore()

    ai.define_prompt(name='signalAgent', model='programmableModel', system='Hello')
    agent = ai.define_prompt_agent(name='signalAgent', store=store)

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='Response'))]),
        )
    )

    async with agent.connect() as session:
        abort_signal = asyncio.Event()

        # Send a turn passing the external abort_signal in opts
        turn = session.send('Hello', opts={'abort_signal': abort_signal})

        # Trigger the external signal immediately
        abort_signal.set()

        # Verify the turn is aborted and raises CancelledError
        with pytest.raises(asyncio.CancelledError):
            await turn.output
