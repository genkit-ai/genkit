#!/usr/bin/env python3
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

"""Backend: customPatch + artifact stream chunks using AgentAPI."""

from __future__ import annotations

from uuid import uuid4

from genkit import ActionRunContext, FinishReason, Genkit, Message, Part, TextPart
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    Artifact,
    InMemorySessionStore,
    SessionRunner,
    TurnResult,
    AgentInit,
)
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


async def stateful_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        await sess.update_custom(lambda c: {'turns': (c or {}).get('turns', 0) + 1})
        await sess.add_artifacts(Artifact(name='status', parts=[Part(TextPart(text=f'turn {sess.turn_index + 1}'))]))
        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None

        stream_resp = ai.generate_stream(
            model='googleai/gemini-flash-latest',
            system='Acknowledge progress in one sentence.',
            messages=messages,
        )
        async for chunk in stream_resp.stream:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk.model_dump()))

        res = await stream_resp.response
        if res.message:
            await sess.add_messages(res.message)

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='statefulAgent', fn=stateful_fn, store=store)


async def main() -> None:
    session = agent.connect(AgentInit(session_id=str(uuid4())))
    print("--- SENDING TURN ---")
    turn = session.send('Go')
    async for chunk in turn.stream:
        print('chunk:', chunk)

    out = await turn.output
    print('output:', out)
    print('accumulated artifacts:', session.artifacts)
    print('accumulated custom state:', session.state)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
