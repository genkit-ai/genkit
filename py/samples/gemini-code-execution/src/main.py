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

"""Code execution - let Gemini write and run Python for a task."""

from pydantic import BaseModel, Field

from genkit import Genkit, Message
from genkit.plugins.google_genai import GeminiConfigSchema, GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-3-pro-preview')


class CodeExecutionInput(BaseModel):
    """Input for code execution."""

    task: str = Field(default='What is the sum of the first 50 prime numbers?', description='Problem to solve')


@ai.flow()
async def execute_code(input: CodeExecutionInput) -> Message:
    """Ask Gemini to generate and execute code."""

    response = await ai.generate(
        prompt=f'Write code and run it to solve this task: {input.task}',
        config=GeminiConfigSchema.model_validate({'code_execution': True}).model_dump(),
    )
    if not response.message:
        raise ValueError('No message returned from model')
    return response.message


async def main() -> None:
    """Run the code execution sample once."""
    try:
        message = await execute_code(CodeExecutionInput())
        print(message.model_dump_json(indent=2))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
