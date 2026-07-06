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

"""Code execution - let Gemini write and run Python for a task. Requires GEMINI_API_KEY.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit with Google GenAI plugin and default Gemini model
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


async def main() -> None:
    """Run code execution directly without intermediate flow wrappers."""
    try:
        # 2. Instruct Gemini to generate and execute Python code inside a secure sandbox (`code_execution=True`)
        response = await ai.generate(
            prompt='Write code and run it to calculate the sum of the first 50 prime numbers.',
            config={'code_execution': True},
        )
        if response.message:
            print(response.message.model_dump_json(indent=2))
        # => Message containing both the generated Python script block and the execution stdout result:
        # => "```python\ndef is_prime(n): ...\n```\nOutput: 5117"
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
