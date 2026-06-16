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

"""Backend: define_custom_agent (hand-written sess.run + generate loop)."""

from __future__ import annotations

from uuid import uuid4

from genkit import ActionRunContext, FinishReason, Genkit, Message
from genkit.agent import (
    AgentFinishReason,
    AgentInit,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    InMemorySessionStore,
    SessionRunner,
    TurnResult,
)
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


async def custom_coder_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None

        stream_resp = ai.generate_stream(
            model='googleai/gemini-flash-latest',
            system='Concise coding assistant.',
            messages=messages,
        )
        async for chunk in stream_resp.stream:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

        res = await stream_resp.response
        if res.message:
            await sess.add_messages(res.message)

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='customCoder', fn=custom_coder_fn, store=store)


async def main() -> None:
    conn = await agent.stream_bidi(AgentInit(session_id=str(uuid4())))
    await conn.send_text('What is a Python list comprehension?')
    await conn.close()

    async for chunk in conn.receive():
        print('chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out = await conn.output()
    print('output:', out.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    ai.run_main(main())
