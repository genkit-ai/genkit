# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Realtime tracing demo - spans appear in DevUI as they start, not when they end. See README.md."""

import asyncio

from genkit import Genkit
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.0-flash')


async def _run_realtime_demo(topic: str) -> str:
    """Shared tracing demo implementation for the flow and direct CLI run."""

    async def research() -> str:
        await asyncio.sleep(2)
        return f'Researched {topic}'

    async def summarize() -> str:
        await asyncio.sleep(1)
        return f'Summarized {topic}'

    step1 = await ai.run(name='research', fn=research)
    step2 = await ai.run(name='summarize', fn=summarize)
    response = await ai.generate(prompt=f'One sentence about {topic}.', config={'max_output_tokens': 50})
    return f'{step1} → {step2} → {response.text}'


@ai.flow(name='trace_steps_live')
async def realtime_demo(topic: str = 'Python') -> str:
    """Multi-step flow: watch spans appear in DevUI as each step starts."""

    return await _run_realtime_demo(topic)


async def main() -> None:
    """Run the tracing demo once."""
    try:
        print(await _run_realtime_demo('Python'))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
