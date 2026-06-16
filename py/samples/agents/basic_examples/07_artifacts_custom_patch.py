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

"""Backend: customPatch + artifact stream chunks (from update_custom / add_artifacts)."""

from __future__ import annotations

from uuid import uuid4

from genkit import Genkit
from genkit._ai._agent import SessionRunner, TurnResult, _to_agent_finish_reason
from genkit._ai._generate import generate_action
from genkit._ai._prompt import PromptConfig, to_generate_action_options
from genkit._core._action import ActionRunContext
from genkit._core._typing import AgentInput, AgentResult, AgentStreamChunk, Artifact, Part, TextPart
from genkit.agent import AgentInit, InMemorySessionStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


async def stateful_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        await sess.update_custom(lambda c: {'turns': (c or {}).get('turns', 0) + 1})
        await sess.add_artifacts(Artifact(name='status', parts=[Part(TextPart(text=f'turn {sess.turn_index + 1}'))]))
        history = await sess.get_messages()
        child = ai.registry.new_child()
        pc = PromptConfig(
            model='googleai/gemini-flash-latest',
            system='Acknowledge progress in one sentence.',
            messages=history or None,
        )
        opts = await to_generate_action_options(child, pc)

        def on_chunk(chunk) -> None:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

        res = await generate_action(child, opts, on_chunk=on_chunk, abort_signal=ctx.abort_signal)
        if res.message:
            await sess.add_messages(res.message)
        return TurnResult(finish_reason=_to_agent_finish_reason(res.finish_reason))

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='statefulAgent', fn=stateful_fn, store=store)


async def main() -> None:
    conn = await agent.stream_bidi(AgentInit(session_id=str(uuid4())))
    await conn.send_text('Go')
    await conn.close()

    async for chunk in conn.receive():
        dumped = chunk.model_dump(by_alias=True, exclude_none=True)
        print('chunk:', dumped)

    out = await conn.output()
    print('output:', out.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    ai.run_main(main())
