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

from genkit._ai._agent import (
    _HISTORY_TAG,
    _PREAMBLE_KEY,
    apply_preamble_tags,
    tag_history_for_render,
)
from genkit._ai._aio import Genkit
from genkit._ai._testing import define_programmable_model
from genkit._core._model import Message, ModelResponse
from genkit._core._typing import FinishReason, Part, Role, TextPart


def test_tag_history_for_render_copies_messages() -> None:
    original = Message(role=Role.USER, content=[Part(TextPart(text='hi'))], metadata={'keep': True})
    tagged = tag_history_for_render([original])[0]

    assert tagged.metadata is not None
    assert tagged.metadata[_HISTORY_TAG] is True
    assert tagged.metadata['keep'] is True
    assert original.metadata == {'keep': True}


def test_apply_preamble_tags_tags_template_messages_and_strips_history_marker() -> None:
    history = Message(role=Role.USER, content=[Part(TextPart(text='turn 1'))], metadata={_HISTORY_TAG: True})
    system = Message(role=Role.SYSTEM, content=[Part(TextPart(text='be helpful'))])

    tagged = apply_preamble_tags([system, history])

    assert tagged[0].metadata == {_PREAMBLE_KEY: True}
    assert tagged[1].metadata is None


def test_apply_preamble_tags_does_not_mutate_shared_prompt_messages() -> None:
    shared = Message(role=Role.SYSTEM, content=[Part(TextPart(text='static system'))])
    tagged = apply_preamble_tags([shared])[0]

    assert tagged.metadata == {_PREAMBLE_KEY: True}
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

    conn = await agent.stream_bidi()
    await conn.send_text('hello')
    async for chunk in conn.receive():
        if chunk.turn_end is not None:
            break
    await conn.close()
    out = await conn.output()

    assert out.state is not None
    assert out.state.messages is not None
    roles = [m.role for m in out.state.messages]
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

    conn = await agent.stream_bidi()
    await conn.send_text('hello')
    async for chunk in conn.receive():
        if chunk.turn_end is not None:
            break
    await conn.send_text('again')
    async for chunk in conn.receive():
        if chunk.turn_end is not None:
            break
    await conn.close()
    out = await conn.output()

    assert out.state is not None
    assert out.state.messages is not None
    roles = [m.role for m in out.state.messages]
    assert Role.SYSTEM not in roles
    assert roles == [Role.USER, Role.MODEL, Role.USER, Role.MODEL]

    assert pm.request_count == 2
    assert pm.last_request is not None
    # Each generate call still gets a fresh system preamble for the model.
    turn_two_roles = [m.role for m in pm.last_request.messages]
    assert Role.SYSTEM in turn_two_roles
