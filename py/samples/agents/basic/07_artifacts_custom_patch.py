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

A custom turn bumps a state counter and maintains a live `session_log.md` artifact
before answering. Declaring a ``state_schema`` means custom state comes back as a
typed model — so ``chat.state``, ``response.state``, and each streamed ``chunk.custom``
are a ``Progress`` model with typed attribute access. This demonstrates how live state
and session artifacts work together in a custom agent turn. Requires GEMINI_API_KEY.
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
    TurnContext,
    TurnResult,
)

ai = Genkit(plugins=[GoogleAI()])
store = InMemorySessionStore()


class Progress(BaseModel):
    turns: int = 0
    last_prompt: str | None = None


async def stateful_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
        # Extract user input text if present
        prompt_text = ''
        if inp.message and inp.message.content:
            for p in inp.message.content:
                root = p.root
                if isinstance(root, TextPart) and root.text:
                    prompt_text += root.text

        # 1. Update custom state (typed Progress model)
        await sess.update_custom(
            lambda c: {
                'turns': (c or {}).get('turns', 0) + 1,
                'last_prompt': prompt_text or (c or {}).get('last_prompt'),
            }
        )

        # 2. Append turn log entry to session_log.md artifact
        existing_artifacts = await sess.get_artifacts()
        log_content = ''
        for art in existing_artifacts:
            if art.name == 'session_log.md':
                log_parts: list[str] = []
                for p in art.parts:
                    root = p.root
                    if isinstance(root, TextPart) and root.text:
                        log_parts.append(root.text)
                log_content = ''.join(log_parts)
                break

        turn_num = sess.turn_index + 1
        entry_text = prompt_text or 'Turn request'
        log_content += f'### Turn {turn_num}\n- **Prompt**: {entry_text}\n\n'

        await sess.add_artifacts(
            Artifact(
                name='session_log.md',
                parts=[Part(TextPart(text=log_content))],
            )
        )

        # 3. Stream model response
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

    turn = chat.send('Go')
    async for chunk in turn.stream:
        if chunk.custom is not None:
            print(f'\rturn {chunk.custom.turns} · {chunk.accumulated_text}', end='', flush=True)
    print()

    res = await turn.response
    if res.state is not None:
        print(f'{res.state.turns} turn(s), {len(chat.artifacts)} artifact(s)')
        log_art = next((a for a in chat.artifacts if a.name == 'session_log.md'), None)
        if log_art:
            log_parts: list[str] = []
            for p in log_art.parts:
                root = p.root
                if isinstance(root, TextPart) and root.text:
                    log_parts.append(root.text)
            log_text = ''.join(log_parts)
            print(f"\nCreated artifact 'session_log.md':\n{log_text}")


if __name__ == '__main__':
    ai.run_main(main())
