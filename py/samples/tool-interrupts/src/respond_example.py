#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""Tool interrupts (Respond path) — pause generation for user input, then fulfill the tool directly without re-running.

Run:
    uv run src/respond_example.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from genkit import Genkit, Interrupt, respond_to_interrupt
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


class TriviaQuestions(BaseModel):
    question: str = Field(description='The trivia question')
    options: list[str] = Field(description='Multiple choice options')


# 1. Define a tool that pauses generation to present questions (`raise Interrupt`)
@ai.tool()
async def present_questions(questions: TriviaQuestions) -> None:
    """Presents questions to the user and pauses execution (`raise Interrupt`)."""
    # => Interrupt raised: generation pauses instantly, returning `response.interrupts`
    raise Interrupt(questions.model_dump(mode='json'))


async def main() -> None:
    """Drive tool interruption and fulfill directly using `respond_to_interrupt` (`resume_respond`)."""
    try:
        # 2. First `generate` call triggers the tool and pauses execution
        res1 = await ai.generate(
            prompt='Ask me a multiple choice trivia question about Python.',
            tools=[present_questions],
        )
        interrupt = res1.interrupts[0]
        payload = interrupt.tool_request.input or {}
        print(f'Question: {payload.get("question")}')
        # => Question: What keyword is used to define a function in Python?

        # 3. Fulfill directly using `respond_to_interrupt` (`resume_respond`) — does NOT re-run the tool
        user_answer = respond_to_interrupt('def', interrupt=interrupt, metadata={'source': 'user'})
        res2 = await ai.generate(
            messages=res1.messages,
            resume_respond=[user_answer],
            tools=[present_questions],
        )
        print(res2.text)
        # => "Correct! 'def' is indeed the keyword used to define a function."
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
