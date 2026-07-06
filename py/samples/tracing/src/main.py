#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Realtime tracing demo - spans appear in DevUI as they start, not when they end.

Requires GEMINI_API_KEY. See README.md.
"""

from __future__ import annotations

import asyncio

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit (spans automatically broadcast live via the trace reflection server)
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


# 2. Define a multi-step flow where each `ai.run` block emits live trace spans instantly
@ai.flow(name='trace_steps_live')
async def realtime_demo(topic: str = 'Python') -> str:
    """Multi-step flow: watch spans appear in Dev UI as each step starts (not when they end)."""

    # 3. Intermediate steps (`ai.run`) emit child spans right as they start executing
    async def research() -> str:
        await asyncio.sleep(1)
        return f'Researched {topic}'

    async def summarize() -> str:
        await asyncio.sleep(0.5)
        return f'Summarized {topic}'

    step1 = await ai.run(name='research', fn=research)
    step2 = await ai.run(name='summarize', fn=summarize)

    # 4. Model generation (`ai.generate`) emits a child span tracking token counts and latency
    response = await ai.generate(prompt=f'Write one sentence about {topic}.', config={'max_output_tokens': 50})

    # => Dev UI live timeline:
    # =>   [trace_steps_live]
    # =>     ├── [research] (1.0s)
    # =>     ├── [summarize] (0.5s)
    # =>     └── [generate] (googleai/gemini-flash-latest)
    # => Return: "Researched Python → Summarized Python → Python is a versatile, high-level programming language."
    return f'{step1} → {step2} → {response.text}'


async def main() -> None:
    """Run the tracing demo once from the CLI."""
    try:
        print(await realtime_demo('Python'))
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
