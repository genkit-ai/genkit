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

from genkit._ai._agents._base import (
    HISTORY_TAG,
    PREAMBLE_KEY,
    apply_preamble_tags,
    tag_history_for_render,
)
from genkit._ai._aio import Genkit
from genkit._ai._testing import define_programmable_model
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import (
    AgentInit,
    FinishReason,
    MessageData,
    Part,
    Role,
    SessionState,
    TextPart,
    ToolRequest,
    ToolRequestPart,
    ToolResponse,
    ToolResponsePart,
)


def test_tag_history_for_render_copies_messages() -> None:
    original = Message(role=Role.USER, content=[Part(TextPart(text='hi'))], metadata={'keep': True})
    tagged = tag_history_for_render([original])[0]

    assert tagged.metadata is not None
    assert tagged.metadata[HISTORY_TAG] is True
    assert tagged.metadata['keep'] is True
    assert original.metadata == {'keep': True}


def test_apply_preamble_tags_tags_template_messages_and_strips_history_marker() -> None:
    history = Message(role=Role.USER, content=[Part(TextPart(text='turn 1'))], metadata={HISTORY_TAG: True})
    system = Message(role=Role.SYSTEM, content=[Part(TextPart(text='be helpful'))])

    tagged = apply_preamble_tags([system, history])

    assert tagged[0].metadata == {PREAMBLE_KEY: True}
    assert tagged[1].metadata is None


def test_apply_preamble_tags_does_not_mutate_shared_prompt_messages() -> None:
    shared = Message(role=Role.SYSTEM, content=[Part(TextPart(text='static system'))])
    tagged = apply_preamble_tags([shared])[0]

    assert tagged.metadata == {PREAMBLE_KEY: True}
    assert shared.metadata is None


@pytest.mark.asyncio
async def test_prompt_agent_does_not_persist_system_preamble() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(name='preambleAgent', model='programmableModel', system='You are terse.')
    agent = ai.define_prompt_agent(name='preambleAgent')

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='ok'))]),
        )
    )

    async with agent.chat() as session:
        turn = session.send('hello')
        async for _chunk in turn.stream:
            pass
        await turn.output

    roles = [m.role for m in session.messages]
    assert Role.SYSTEM not in roles
    assert roles == [Role.USER, Role.MODEL]


@pytest.mark.asyncio
async def test_prompt_agent_multi_turn_session_has_no_accumulated_preamble() -> None:
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(name='preambleAgent', model='programmableModel', system='You are terse.')
    agent = ai.define_prompt_agent(name='preambleAgent')

    pm.responses.extend([
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='first'))]),
        ),
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(TextPart(text='second'))]),
        ),
    ])

    async with agent.chat() as session:
        turn1 = session.send('hello')
        async for _chunk in turn1.stream:
            pass
        await turn1.output

        turn2 = session.send('again')
        async for _chunk in turn2.stream:
            pass
        await turn2.output

    roles = [m.role for m in session.messages]
    assert Role.SYSTEM not in roles
    assert roles == [Role.USER, Role.MODEL, Role.USER, Role.MODEL]

    assert pm.request_count == 2
    assert pm.last_request is not None
    # Each generate call still gets a fresh system preamble for the model.
    turn_two_roles = [m.role for m in pm.last_request.messages]
    assert Role.SYSTEM in turn_two_roles


@pytest.mark.asyncio
async def test_prompt_agent_explicit_history_tag_preamble() -> None:
    """Verifies that explicit {{history}} tags work correctly with preamble marking.

    When the prompt template explicitly references history, any instructions compiled
    before (e.g. prefix system prompts) and after (e.g. suffix user queries) the history
    should be marked as preamble and excluded from persistence. Only the runtime history
    and model responses are persisted.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(
        name='explicitHistory',
        model='programmableModel',
        messages="""
        {{role "system"}}
        Prefix system instruction.
        {{history}}
        {{role "user"}}
        Suffix user instruction.
        """,
    )
    agent = ai.define_prompt_agent(name='explicitHistory')

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='response'))]),
        )
    )

    async with agent.chat() as session:
        turn = session.send('turn 1')
        async for _chunk in turn.stream:
            pass
        await turn.output

    # Verify that the LLM received the prefix instructions, history, and suffix instructions
    assert pm.request_count == 1
    assert pm.last_request is not None
    req_msgs = pm.last_request.messages
    assert len(req_msgs) == 3
    assert req_msgs[0].role == Role.SYSTEM
    t0 = req_msgs[0].content[0].root.text
    assert t0 is not None
    assert 'Prefix' in t0
    assert req_msgs[1].role == Role.USER
    assert req_msgs[1].content[0].root.text == 'turn 1'
    assert req_msgs[2].role == Role.USER
    t2 = req_msgs[2].content[0].root.text
    assert t2 is not None
    assert 'Suffix' in t2

    # Verify that ONLY history and model response are stored (Prefix & Suffix are filtered out)
    roles = [m.role for m in session.messages]
    assert Role.SYSTEM not in roles
    assert roles == [Role.USER, Role.MODEL]
    assert session.messages[0].content[0].root.text == 'turn 1'
    assert session.messages[1].content[0].root.text == 'response'


@pytest.mark.asyncio
async def test_prompt_agent_few_shot_preamble() -> None:
    """Verifies that static few-shot messages in the template are treated as preamble.

    Static examples in the template do not belong to the runtime conversation history.
    They must be sent to the LLM model to provide context, but must be discarded
    from the session store at the end of the turn.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(
        name='fewShotAgent',
        model='programmableModel',
        messages="""
        {{role "system"}}
        System help.
        {{role "user"}}
        Q: 1+1
        {{role "model"}}
        A: 2
        {{history}}
        """,
    )
    agent = ai.define_prompt_agent(name='fewShotAgent')

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='response'))]),
        )
    )

    async with agent.chat() as session:
        turn = session.send('turn 1')
        async for _chunk in turn.stream:
            pass
        await turn.output

    # Verify the LLM received the system prompt, few-shots, and user query
    assert pm.request_count == 1
    assert pm.last_request is not None
    req_msgs = pm.last_request.messages
    assert len(req_msgs) == 4
    t1 = req_msgs[1].content[0].root.text
    assert t1 is not None
    assert 'Q: 1+1' in t1
    t2 = req_msgs[2].content[0].root.text
    assert t2 is not None
    assert 'A: 2' in t2

    # Verify few-shots are stripped, and only runtime history & response are saved
    assert len(session.messages) == 2
    assert [m.role for m in session.messages] == [Role.USER, Role.MODEL]
    assert session.messages[0].content[0].root.text == 'turn 1'
    assert session.messages[1].content[0].root.text == 'response'


@pytest.mark.asyncio
async def test_prompt_agent_tool_messages_preserved_verbatim() -> None:
    """Verifies that tool execution messages in the history are preserved verbatim.

    Tool call and response messages represent part of the conversation history.
    These must maintain their history tags through compilation and not be flagged
    as preambles, ensuring tool traces are successfully saved to the database.
    """
    ai = Genkit()
    pm, _ = define_programmable_model(ai)

    ai.define_prompt(name='toolHistoryAgent', model='programmableModel', system='You are helpful.')
    agent = ai.define_prompt_agent(name='toolHistoryAgent')

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='done'))]),
        )
    )

    # Pre-seed tool call and tool response history
    tool_request_msg = Message(
        role=Role.MODEL,
        content=[Part(root=ToolRequestPart(tool_request=ToolRequest(name='myTool', ref='r1', input={'x': 1})))],
    )
    tool_response_msg = Message(
        role=Role.TOOL,
        content=[
            Part(root=ToolResponsePart(tool_response=ToolResponse(name='myTool', ref='r1', output={'result': 'ok'})))
        ],
    )

    history = [
        Message(role=Role.USER, content=[Part(root=TextPart(text='run tool'))]),
        tool_request_msg,
        tool_response_msg,
    ]

    init = AgentInit(state=SessionState(messages=[MessageData.model_validate(m.model_dump()) for m in history]))
    async with agent.chat(init) as session:
        turn = session.send('continue')
        async for _chunk in turn.stream:
            pass
        await turn.output

    # Verify that LLM receives all history messages, including tool components
    assert pm.request_count == 1
    assert pm.last_request is not None
    req_msgs = pm.last_request.messages
    assert len(req_msgs) == 5
    assert req_msgs[1].role == Role.USER
    assert req_msgs[2].role == Role.MODEL
    assert req_msgs[3].role == Role.TOOL
    assert req_msgs[4].role == Role.USER

    # Verify that tool request/responses are present in the session messages
    assert len(session.messages) == 5
    assert session.messages[1].role == Role.MODEL
    assert isinstance(session.messages[1].content[0].root, ToolRequestPart)
    assert session.messages[2].role == Role.TOOL
    assert isinstance(session.messages[2].content[0].root, ToolResponsePart)
