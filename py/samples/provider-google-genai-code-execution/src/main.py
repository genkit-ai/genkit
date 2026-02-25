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

"""Code execution sample - Running code with Gemini.

This sample demonstrates how to use Gemini's server-side code execution
feature, which allows the model to write and execute Python code.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Code Execution      │ AI writes Python code AND runs it. Ask "what's     │
    │                     │ 2+2?" and it actually calculates, not guesses.     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Server-side         │ Code runs on Google's servers, not your computer.  │
    │                     │ Safe sandbox environment.                          │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Executable Code     │ The Python code Gemini writes to solve your        │
    │                     │ problem. You can see what it created.              │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Execution Result    │ The output from running the code. Numbers,         │
    │                     │ data, or error messages.                           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ CustomPart          │ Special response parts containing code or          │
    │                     │ results. Parse them to show the user.              │
    └─────────────────────┴────────────────────────────────────────────────────┘

Key Features
============
| Feature Description                     | Example Function / Code Snippet     |
|-----------------------------------------|-------------------------------------|
| Code Execution Config                   | `code_execution=True`               |
| Executable Code Part Handling           | `PartConverter.EXECUTABLE_CODE`     |
| Code Execution Result Handling          | `PartConverter.CODE_EXECUTION_RESULT`|
| Custom Part Parsing                     | `CustomPart` processing             |

See README.md for testing instructions.
"""

import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.blocks.model import MessageWrapper
from genkit.core.logging import get_logger
from genkit.core.typing import CustomPart, Message, TextPart
from genkit.plugins.google_genai import GeminiConfigSchema, GoogleAI
from genkit.plugins.google_genai.models.utils import PartConverter
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-pro-preview',
)


DEFAULT_CODE_TASK = 'What is the sum of the first 50 prime numbers?'


class CodeExecutionInput(BaseModel):
    """Input for code execution flow."""

    task: str = Field(default=DEFAULT_CODE_TASK, description='Task to execute code for')


@ai.flow()
async def execute_code(input: CodeExecutionInput) -> MessageWrapper:
    """Execute code for the given task.

    Args:
        input: Input with task to execute code for.

    Returns:
        The generated response enclosed in a MessageWrapper. The content field should contain the following:
        1. CustomPart with executableCode(language and code).
        2. CustomPart with codeExecutionResult(outcome and output).
        3. Textpart describing code and output generated.
    """
    response = await ai.generate(
        prompt=f'Generate and run code for the task: {input.task}',
        config=GeminiConfigSchema.model_validate({'temperature': 1, 'code_execution': True}).model_dump(),
    )
    if not response.message:
        raise ValueError('No message returned from model')
    return response.message


def display_code_execution(message: Message) -> None:
    """Display the code execution results from a message.

    This function parses the message content and prints details about
    any code execution that occurred, including the code itself and
    the execution results.
    """
    print('\n=== CODE EXECUTION DETAILS ===')  # noqa: T201
    for part in message.content:
        if isinstance(part.root, CustomPart):
            if PartConverter.EXECUTABLE_CODE in part.root.custom:
                code_data = part.root.custom[PartConverter.EXECUTABLE_CODE]
                lang = code_data.get(PartConverter.LANGUAGE, 'unknown')
                code = code_data.get(PartConverter.CODE, '')
                print(f'Language: {lang}')  # noqa: T201
                print(f'```{lang}\n{code}\n```')  # noqa: T201
            elif PartConverter.CODE_EXECUTION_RESULT in part.root.custom:
                result_data = part.root.custom[PartConverter.CODE_EXECUTION_RESULT]
                outcome = result_data.get(PartConverter.OUTCOME, 'unknown')
                output = result_data.get(PartConverter.OUTPUT, '')
                print(f'\nExecution Status: {outcome}')  # noqa: T201
                print('Output:')  # noqa: T201
                if output.strip():
                    for line in output.splitlines():
                        print(f'  {line}')  # noqa: T201
                else:
                    print('  <no output>')  # noqa: T201
        elif isinstance(part.root, TextPart) and part.root.text.strip():
            print(f'\nExplanation:\n{part.root.text}')  # noqa: T201
    print('=== END CODE EXECUTION ===\n')  # noqa: T201


async def main() -> None:
    """Main entry point for the Google genai code execution sample - keep alive for Dev UI.

    This function demonstrates how to perform code execution using the
    Genkit framework.
    """
    response_msg = await execute_code(CodeExecutionInput(task='What is the sum of the first 50 prime numbers?'))
    display_code_execution(response_msg)
    await logger.ainfo(response_msg.text)


if __name__ == '__main__':
    ai.run_main(main())
