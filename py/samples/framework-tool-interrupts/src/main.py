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

"""Tool interrupts sample - Human-in-the-loop with tool interruptions.

This sample demonstrates how to use tool interruptions to pause AI execution
and wait for human input before continuing, enabling interactive experiences.

See README.md for testing instructions.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Tool Interrupt      │ AI pauses and asks for human input. Like a         │
    │                     │ waiter asking "How do you want that cooked?"       │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Human-in-the-Loop   │ A person reviews/approves AI actions before they   │
    │                     │ happen. Safety check for important decisions.      │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ctx.interrupt()     │ The function that pauses execution. Pass data      │
    │                     │ the human needs to make a decision.                │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ tool_response()     │ Resume execution with the human's answer.          │
    │                     │ "The user chose option B."                         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ response.interrupts │ Check if AI is waiting for input. Non-empty        │
    │                     │ means there's a question to answer.                │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow (Human-in-the-Loop)::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW TOOL INTERRUPTS PAUSE AND RESUME                   │
    │                                                                         │
    │    AI: "Here's a trivia question!"                                      │
    │         │                                                               │
    │         │  (1) AI calls tool with question                              │
    │         ▼                                                               │
    │    ┌─────────────────┐                                                  │
    │    │  present_       │   Tool calls ctx.interrupt({question: ...})      │
    │    │  questions()    │                                                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (2) Execution PAUSES, returns to your code               │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  Your Code      │   response.interrupts has the question           │
    │    │  (waiting)      │   Display to user, get their answer              │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (3) User answers: "Option B"                             │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  tool_response()│   Pass user's answer back to AI                  │
    │    └────────┬────────┘                                                  │
    │             │                                                           │
    │             │  (4) AI continues with the answer                         │
    │             ▼                                                           │
    │    ┌─────────────────┐                                                  │
    │    │  AI Continues   │   "That's correct!" or "Sorry, wrong."           │
    │    └─────────────────┘                                                  │
    └─────────────────────────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Tool Interruption                       | `ctx.interrupt(payload)`            |
| Handling Interrupts in Loop             | `response.interrupts` check         |
| Resuming with Tool Response             | `tool_response(request, input)`     |
| Interactive CLI Loop                    | `while True: ... input()`           |
"""

import os

from pydantic import BaseModel, Field

from genkit.ai import (
    Genkit,
    ToolRunContext,
    tool_response,
)
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.google_genai.models import gemini
from samples.shared.logging import setup_sample

setup_sample()

ai = Genkit(
    plugins=[GoogleAI()],
    model=f'googleai/{gemini.GoogleAIGeminiVersion.GEMINI_3_FLASH_PREVIEW}',
)


class TriviaQuestions(BaseModel):
    """Trivia questions."""

    question: str = Field(description='the main question')
    answers: list[str] = Field(description='list of multiple choice answers (typically 4), 1 correct 3 wrong')


@ai.tool()
def present_questions(questions: TriviaQuestions, ctx: ToolRunContext) -> None:
    """Presents questions to the user and responds with the selected answer."""
    ctx.interrupt(questions.model_dump())


@ai.flow()
async def play_trivia(theme: str = 'Science') -> str:
    """Plays a trivia game with the user."""
    response = await ai.generate(
        prompt='You are a trivia game host. Cheerfully greet the user when they '
        + f'first join. The user has selected the theme: "{theme}". '
        + 'Call `present_questions` tool with questions and the tools will present '
        + 'the questions in a nice UI. The user will pick an answer and then you '
        + 'tell them if they were right or wrong. Be dramatic (but terse)! It is a '
        + 'show!\n\n[user joined the game]',
        tools=['present_questions'],
    )

    # Check for interrupts and return the question to the user
    if len(response.interrupts):
        request = response.interrupts[0]
        question_data = request.tool_request.input
        if question_data:
            # For a full interactive flow, you would typically:
            # 1. Prompt the user for their answer here (e.g., using input()).
            # 2. Call tool_response(request, user_answer) to resume the AI conversation.
            # 3. Regenerate with the tool_response.

            # Prepend the greeting/text response if available
            text_response = (response.text + '\n\n') if response.text else ''
            question = question_data.get('question')
            answers = question_data.get('answers')
            return f'{text_response}INTERRUPTED: {question}\nAnswers: {answers}'
        return 'INTERRUPTED: (No input data)'

    return response.text


async def main() -> None:
    """Main function."""
    # In Dev mode (genkit start), we must skip the blocking input() loop
    # so the reflection server can handle requests (like play_trivia)
    if os.environ.get('GENKIT_ENV') == 'dev':
        return

    response = await ai.generate(
        prompt='You are a trivia game host. Cheerfully greet the user when they '
        + 'first join and ank them to for the theme of the trivia game, suggest '
        + "a few theme options, they don't have to use your suggestion, feel free "
        + 'to be silly. When they user us ready, call '
        + '`present_questions` tool with questions and the tools will present '
        + 'the questions in a nice UI. The user will pick an answer and then you '
        + 'tell them if they were right or wrong. Be dramatic (but terse)! It is a '
        + 'show!\n\n[user joined the game]',
    )
    messages = response.messages
    while True:
        response = await ai.generate(
            messages=messages,
            prompt=input('Say: '),
            tools=['present_questions'],
        )
        messages = response.messages
        if len(response.interrupts) > 0:
            request = response.interrupts[0]
            input_data = request.tool_request.input
            if input_data:
                i = 1
                for _question in input_data.get('answers', []):
                    i += 1

            tr = tool_response(request, input('Your answer (number): '))
            response = await ai.generate(
                messages=messages,
                tool_responses=[tr],
                tools=['present_questions'],
            )
            messages = response.messages


if __name__ == '__main__':
    ai.run_main(main())
