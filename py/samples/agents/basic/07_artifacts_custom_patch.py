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

A Research Assistant custom agent streams live research state (topics explored, depth,
insights count) via a typed model while continuously building an executive briefing
artifact (`research_brief.md`).

Declaring a ``state_schema`` means custom state comes back as a typed model — so
``chat.state``, ``response.state``, and each streamed ``chunk.custom`` are a ``ResearchState``
with typed attribute access. This demonstrates how live state patches and session
artifacts work together in a real-world custom agent workflow. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

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


class ResearchState(BaseModel):
    topics_explored: list[str] = Field(default_factory=list)
    depth: str = 'Initial Overview'
    insights_count: int = 0


async def research_agent_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput, _: TurnContext) -> TurnResult | None:
        # Extract user input text if present
        prompt_text = ''
        if inp.message and inp.message.content:
            for p in inp.message.content:
                root = p.root
                if isinstance(root, TextPart) and root.text:
                    prompt_text += root.text
        topic = prompt_text.strip() or 'General Overview'

        # 1. Update custom state (typed ResearchState model)
        await sess.update_custom(
            lambda c: {
                'topics_explored': [*(c or {}).get('topics_explored', []), topic[:40]],
                'depth': 'Deep Dive' if (c or {}).get('insights_count', 0) > 0 else 'Initial Overview',
                'insights_count': (c or {}).get('insights_count', 0) + 1,
            }
        )

        # 2. Build or update executive briefing artifact (research_brief.md)
        existing_artifacts = await sess.get_artifacts()
        brief_content = ''
        for art in existing_artifacts:
            if art.name == 'research_brief.md':
                log_parts: list[str] = []
                for p in art.parts:
                    root = p.root
                    if isinstance(root, TextPart) and root.text:
                        log_parts.append(root.text)
                brief_content = ''.join(log_parts)
                break

        turn_num = sess.turn_index + 1
        if not brief_content:
            brief_content = '# Executive Research Briefing\n\n'

        brief_content += f'### Topic {turn_num}: {topic}\n'
        brief_content += f'- **Added in Turn**: {turn_num}\n'
        brief_content += f'- **Status**: Briefing compiled for *{topic}*\n\n---\n\n'

        await sess.add_artifacts([
            Artifact(
                name='research_brief.md',
                parts=[Part(TextPart(text=brief_content))],
            )
        ])

        # 3. Stream model response
        history = await sess.get_messages()
        messages = [Message(m) for m in history] if history else None

        stream_resp = ai.generate_stream(
            model=GoogleAI.gemini_model('gemini-flash-latest'),
            system=(
                'You are a Senior Research Analyst. Provide concise, clear, '
                'and structured research insights for the user prompt.'
            ),
            messages=messages,
        )
        async for chunk in stream_resp.stream:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))

        res = await stream_resp.response
        if res.message:
            await sess.add_messages([res.message])

        fr = AgentFinishReason.STOP if res.finish_reason == FinishReason.STOP else AgentFinishReason.UNKNOWN
        return TurnResult(finish_reason=fr)

    await sess.run(handle_turn)
    return await sess.result()


agent = ai.define_custom_agent(name='researchAgent', fn=research_agent_fn, store=store, state_schema=ResearchState)


async def main() -> None:
    chat = agent.chat()  # AgentChat[ResearchState] — state is typed

    turn = chat.send_stream('Analyze Python async performance best practices')
    async for chunk in turn.stream:
        if chunk.custom is not None:
            topics = ', '.join(chunk.custom.topics_explored)
            print(f'\r[State: {chunk.custom.depth} | Topics: {topics}] · {chunk.accumulated_text}', end='', flush=True)
    print()

    res = await turn.response
    if res.state is not None:
        print(f'\n{res.state.insights_count} insight(s) compiled across topics: {res.state.topics_explored}')
        brief_art = next((a for a in chat.artifacts if a.name == 'research_brief.md'), None)
        if brief_art:
            log_parts: list[str] = []
            for p in brief_art.parts:
                root = p.root
                if isinstance(root, TextPart) and root.text:
                    log_parts.append(root.text)
            brief_text = ''.join(log_parts)
            print(f"\nGenerated Artifact 'research_brief.md':\n{brief_text}")


if __name__ == '__main__':
    ai.run_main(main())
