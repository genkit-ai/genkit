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

"""Backend: graceful turn failure — finish_reason failed + error on AgentOutput."""

from __future__ import annotations

import asyncio
from uuid import uuid4

from genkit import Genkit, GenkitError
from genkit._ai._agent import SessionRunner, TurnResult
from genkit._core._action import ActionRunContext
from genkit._core._typing import AgentFinishReason, AgentInput, AgentResult, MessageData, Part, TextPart
from genkit.agent import AgentInit, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI


async def main() -> None:
    ai = Genkit(plugins=[GoogleAI()])

    store = InMemorySessionStore()

    async def flaky_fn(sess: SessionRunner, _ctx: ActionRunContext) -> AgentResult:
        async def handle_turn(inp: AgentInput) -> TurnResult | None:
            text = ''
            if inp.messages:
                for part in inp.messages[-1].content or []:
                    root = getattr(part, 'root', part)
                    if isinstance(root, TextPart) and root.text:
                        text += root.text
            if 'fail' in text.lower():
                raise GenkitError(status='INTERNAL', message='Simulated turn failure')
            msgs = await sess.get_messages()
            await sess.set_messages(msgs + [MessageData(role='model', content=[Part(TextPart(text='OK'))])])
            return TurnResult(finish_reason=AgentFinishReason.STOP)

        await sess.run(handle_turn)
        return await sess.result()

    agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=store)

    session_id = str(uuid4())

    conn = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn.send_text('hello')
    await conn.close()
    out_ok = await conn.output()
    print('ok turn:', out_ok.model_dump(by_alias=True, exclude_none=True))

    conn2 = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn2.send_text('please fail now')
    await conn2.close()

    async for chunk in conn2.receive():
        print('fail chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out_fail = await conn2.output()
    print('fail turn:', out_fail.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    asyncio.run(main())
