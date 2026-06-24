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

"""Backend: graceful turn failure — using AgentAPI."""

from __future__ import annotations

from genkit import ActionRunContext, Genkit, GenkitError, Message, Part, TextPart
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    InMemoryLinearSessionStore,
    SessionRunner,
    TurnResult,
)
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLinearSessionStore()


async def flaky_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        text = ''
        if inp.message:
            for part in inp.message.content or []:
                root = getattr(part, 'root', part)
                if isinstance(root, TextPart) and root.text:
                    text += root.text
        if 'fail' in text.lower():
            raise GenkitError(status='INTERNAL', message='Simulated turn failure')
        msgs = await sess.get_messages()
        await sess.set_messages(msgs + [Message(role='model', content=[Part(TextPart(text='OK'))])])
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='flakyAgent', fn=flaky_fn, store=store)


async def main() -> None:
    session = agent.chat()
    try:
        print('--- SENDING TURN 1 (OK) ---')
        turn1 = session.send('hello')
        async for chunk in turn1:
            print('ok chunk:', chunk)
        out_ok = await turn1.output
        print('ok turn output:', out_ok)

        print('\n--- SENDING TURN 2 (FAIL) ---')
        turn2 = session.send('please fail now')
        async for chunk in turn2:
            print('fail chunk:', chunk)
        out_fail = await turn2.output
        print('fail turn output:', out_fail)
    finally:
        await session.close()


if __name__ == '__main__':
    ai.run_main(main())
