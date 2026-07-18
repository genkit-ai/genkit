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

"""A multi-step researcher that streams its progress, not just its answer.

This is what ``define_custom_agent`` unlocks: the turn is a little pipeline —
decompose the question (cheap model), research each part, then synthesize a final
answer (streamed). Between steps it bumps a typed ``status`` on the session, and
each bump auto-emits a ``custom`` chunk, so the UI shows "Researching (2/3)…"
live while the model works.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import LITE_MODEL, ai
from pydantic import BaseModel

from genkit import ActionRunContext
from genkit.agent import (
    AgentFinishReason,
    AgentInput,
    AgentResult,
    AgentStreamChunk,
    SessionRunner,
    TurnResult,
)


class ResearchState(BaseModel):
    # A human-readable progress line. Mutating it mid-turn streams a `custom`
    # chunk, so the client's displayed status stays live while we work.
    status: str = ''
    sub_questions: list[str] = []


class SubQuestions(BaseModel):
    questions: list[str]


def _text(parts: list[Any] | None) -> str:
    return ''.join(getattr(p.root, 'text', '') or '' for p in (parts or []))


async def research_fn(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
    async def handle_turn(inp: AgentInput) -> TurnResult | None:
        question = _text(inp.message.content) if inp.message else ''

        # Step 1 — decompose (cheap model, structured output).
        await sess.update_custom(lambda s: {**(s or {}), 'status': 'Decomposing question into sub-topics…'})
        plan = await ai.generate(
            model=LITE_MODEL,
            prompt=(
                'Break this question into exactly 2-3 focused sub-questions that together '
                f'answer it comprehensively.\n\nQuestion: "{question}"'
            ),
            output_schema=SubQuestions,
        )
        sub_questions = plan.output.questions if plan.output else [question]
        await sess.update_custom(lambda s: {**(s or {}), 'sub_questions': sub_questions})

        # Step 2 — research each sub-question, narrating progress.
        findings: list[str] = []
        for i, q in enumerate(sub_questions):
            await sess.update_custom(
                lambda s, i=i, q=q: {**(s or {}), 'status': f'Researching ({i + 1}/{len(sub_questions)}): {q}'}
            )
            answer = await ai.generate(prompt=f'Answer concisely in 2-3 sentences, factual.\n\nQuestion: {q}')
            findings.append(f'### {q}\n{answer.text}')

        # Step 3 — synthesize the final answer, streamed to the client.
        await sess.update_custom(lambda s: {**(s or {}), 'status': 'Synthesizing final response…'})
        synthesis = ai.generate_stream(
            prompt=(
                'Synthesize these findings into one clear, cohesive answer in markdown. '
                f'Do not just list them.\n\nOriginal question: "{question}"\n\n' + '\n\n'.join(findings)
            ),
        )
        async for chunk in synthesis.stream:
            ctx.send_chunk(AgentStreamChunk(model_chunk=chunk))
        res = await synthesis.response
        if res.message:
            await sess.add_messages(res.message)

        await sess.update_custom(lambda s: {**(s or {}), 'status': 'Done'})
        return TurnResult(finish_reason=AgentFinishReason.STOP)

    await sess.run(handle_turn)
    return await sess.result()


research_agent = ai.define_custom_agent(name='researchAgent', fn=research_fn, state_schema=ResearchState)


@ai.flow()
async def test_research_agent(text: str, ctx: ActionRunContext) -> str:
    """Watch the status line advance (chunk.custom) while the answer streams in."""
    chat = research_agent.chat(state={'custom': {'status': '', 'sub_questions': []}, 'messages': [], 'artifacts': []})
    turn = chat.send(text or 'What are the environmental and economic impacts of electric vehicles?')
    async for chunk in turn:
        if chunk.custom is not None and chunk.custom.status:
            ctx.send_chunk(f'[status] {chunk.custom.status}')
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    return res.text


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
