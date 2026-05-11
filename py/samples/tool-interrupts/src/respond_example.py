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

"""Tool interrupts — trivia via ``present_questions`` and ``respond_to_interrupt``.

``present_questions`` raises ``Interrupt`` with the question payload → the user
picks an answer → ``respond_to_interrupt(pick, interrupt=…, metadata=…)`` → second
``generate`` with ``resume_respond``.

For **bank-transfer-style** tool approval (restart path), run ``approval_example.py`` instead.

Run: ``uv run src/respond_example.py``. See README.md.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from genkit import (
    Genkit,
    Interrupt,
    respond_to_interrupt,
)
from genkit.model import ModelResponse
from genkit.plugins.google_genai import GoogleAI  # pyright: ignore[reportMissingImports]

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / 'prompts'

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-flash-preview',
    prompt_dir=_PROMPTS_DIR,
)


class TriviaQuestions(BaseModel):
    """Payload passed into ``present_questions`` when the model calls the tool."""

    question: str = Field(description='the main question')
    answers: list[str] = Field(
        description='list of multiple choice answers (typically 4), 1 correct 3 wrong',
    )


@ai.tool()
async def present_questions(questions: TriviaQuestions) -> None:
    """Presents questions to the user and responds with the selected answer."""
    raise Interrupt(questions.model_dump(mode='json'))


DEMO_TOOLS = [present_questions]


async def interactive_trivia_cli() -> None:
    """Run the CLI: opening turn, then chat with trivia interrupts (respond path)."""

    def show(label: str, r: ModelResponse) -> None:
        """Print one model turn (what the host said)."""
        print(f'\n[{label}]')
        if r.text:
            print(r.text)

    quit_words = frozenset({'q', 'quit', 'exit', 'bye'})
    bar = '=' * 52

    print(f'\n{bar}')
    print('  Tool interrupt demo — trivia (respond path)')
    print(bar)
    print('  1) Host speaks first (you do not type yet).')
    print('  2) Then you chat one line at a time.')
    print('  3) When you see numbered answers, reply with a number.')
    print('  Say quit / exit / q / bye anytime to stop.')
    print(f'{bar}\n')

    print('Starting: host opening (please wait)...\n')
    response = await ai.prompt('trivia_host_cli')()
    messages = response.messages
    show('Host (opening)', response)

    print('-' * 52)
    print('Your turn — reply to the host above, or type quit to leave.')
    print('-' * 52)

    while True:
        user_said = input('\nYou: ').strip()
        if user_said.lower() in quit_words:
            print('Goodbye.')
            return
        if not user_said:
            print('Empty line — type a message, or quit to exit.')
            continue

        response = await ai.generate(
            messages=messages,
            prompt=user_said,
            tools=DEMO_TOOLS,
        )
        messages = response.messages
        show('Host', response)

        while response.interrupts:
            interrupt = response.interrupts[0]
            name = interrupt.tool_request.name

            if name != present_questions.name:
                print(f'Unexpected tool: {name!r}')
                return

            payload = interrupt.tool_request.input
            if payload is None:
                print('Interrupt with no tool input.')
                return
            trivia = TriviaQuestions.model_validate(payload)

            n = len(trivia.answers)
            print('\n' + bar)
            print('  QUESTION — answer with a number')
            print(bar)
            print(trivia.question)
            for i, ans in enumerate(trivia.answers, start=1):
                print(f'  {i}. {ans}')
            print(f'Enter 1–{n}.')

            pick = input('Your choice (number): ').strip()
            if pick.lower() in quit_words:
                print('Goodbye.')
                return

            interrupt_response = respond_to_interrupt(
                pick,
                interrupt=interrupt,
                metadata={'source': 'cli', 'path': 'respond'},
            )
            response = await ai.generate(
                messages=messages,
                resume_respond=[interrupt_response],
                tools=DEMO_TOOLS,
            )
            messages = response.messages
            show('Host (after your answer)', response)


async def main() -> None:
    await interactive_trivia_cli()


if __name__ == '__main__':
    ai.run_main(main())
