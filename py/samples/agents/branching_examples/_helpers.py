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

"""Echo agent and turn helpers for the branching samples using AgentAPI."""

from __future__ import annotations

from genkit import Genkit, Message, Part, TextPart
from genkit._ai._agents._base import SessionRunner, TurnResult
from genkit._core._action import ActionRunContext
from genkit.agent import (
    Agent,
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentResult,
    InMemoryBranchingSessionStore,
)


def text_from_parts(parts) -> str:
    out = ''
    for part in parts or []:
        inner = part.root if isinstance(part, Part) else part
        if isinstance(inner, TextPart):
            out += inner.text
    return out


def model_text(out: AgentOutput) -> str:
    return text_from_parts(out.message.content if out.message else None)


async def run_turn(agent: Agent, init: AgentInit, text: str) -> AgentOutput:
    session_id = None
    if init.state and init.state.session_id:
        session_id = init.state.session_id
    elif init.session_id:
        session_id = init.session_id

    async with agent.chat(AgentInit(session_id=session_id, snapshot_id=init.snapshot_id)) as session:
        turn = session.send(text)
        async for _chunk in turn.stream:
            pass
        return await turn.output


def define_echo_agent(
    ai: Genkit,
    store: InMemoryBranchingSessionStore,
    *,
    name: str = 'branchEcho',
) -> Agent:
    async def echo_fn(sess: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            user_text = text_from_parts(inp.messages[-1].content if inp.messages else None)
            await sess.add_messages(Message(role='model', content=[Part(root=TextPart(text=f'Echo: {user_text}'))]))
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    return ai.define_custom_agent(name=name, fn=echo_fn, store=store)
