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

"""Stream live state patches as a typed model and accumulate artifacts.

A custom turn bumps a counter and writes an artifact before answering. Declaring a
``state_schema`` means the custom state comes back as that model — so ``chat.state``,
``response.state``, and each streamed ``chunk.custom`` are a ``Progress`` with typed
attribute access, not a bare dict. This is the state model behind a live-updating
agent UI. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

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
)

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


class Progress(BaseModel):
    turns: int = 0


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
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

        res = await stream_resp.response
        if res.message:
            await sess.add_messages(res.message)

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='statefulAgent', fn=stateful_fn, store=store, state_schema=Progress)


async def main() -> None:
    chat = agent.chat()  # AgentChat[Progress] — state is typed

    # Text and live state stream through the same chunk — the two halves a
    # live-updating component renders. accumulated_text is the reply so far;
    # chunk.custom is the Progress model after each patch, so reading
    # chunk.custom.turns is typed attribute access, never chunk.custom['turns'].
    turn = chat.send('Go')
    async for chunk in turn.stream:
        if chunk.custom is not None:
            print(f'\rturn {chunk.custom.turns} · {chunk.accumulated_text}', end='', flush=True)
    print()

    # state_schema materializes the wire blob into the model on the way out, so
    # the awaited response and the chat handle both expose .turns, not ['turns'].
    res = await turn.response
    if res.state is not None:
        print(f'{res.state.turns} turn(s), {len(chat.artifacts)} artifact(s)')


if __name__ == '__main__':
    ai.run_main(main())
