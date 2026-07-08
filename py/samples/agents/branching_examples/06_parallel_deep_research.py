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

"""Deep research that forks into parallel investigations and synthesizes.

One agent plans a question, then we fork that single checkpoint into several
branches that each research one angle concurrently with web grounding, and fork
it once more to synthesize the findings into a brief.

Branches share the plan but never see each other, so they explore independently
instead of collapsing into one averaged answer. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import asyncio

from genkit_google_genai import GoogleAI

from genkit import Genkit
from genkit.agent import InMemorySessionStore

QUESTION = 'How are AI coding agents changing the way software teams hire engineers in 2026?'
ANGLES = [
    'The optimistic case: where AI coding agents create leverage and new roles.',
    'The skeptical case: risks, hiring freezes, and where the hype breaks down.',
    'The hard data: concrete hiring numbers, surveys, and company decisions to cite.',
]

ai = Genkit(plugins=[GoogleAI()])

analyst = ai.define_agent(
    name='analyst',
    model='googleai/gemini-flash-latest',
    system=(
        'You are a sharp research analyst. Use web search to ground every claim. '
        'Be concise: 4-6 bullets with concrete facts, names, and numbers.'
    ),
    config={'google_search_retrieval': True},
    store=InMemorySessionStore(),
)


async def main() -> None:
    # Plan once. This snapshot is the shared starting point for every branch.
    plan = analyst.chat()
    await plan.send(f'Plan an approach to research: "{QUESTION}"')
    checkpoint = plan.snapshot_id
    assert checkpoint  # populated once the turn is store-backed

    # Fork the plan into independent investigations and run them concurrently.
    async def investigate(snapshot: str, angle: str) -> str:
        branch = await analyst.load_chat(snapshot_id=snapshot)
        return (await branch.send(f'Research this angle in depth: {angle}')).text

    findings = await asyncio.gather(*(investigate(checkpoint, angle) for angle in ANGLES))

    # Fork the plan once more to synthesize the independent findings into one brief.
    synth = await analyst.load_chat(snapshot_id=checkpoint)
    # → one tight brief: 3 key takeaways + 1 recommendation, merging every angle
    await synth.send(
        'Here are findings gathered independently from several angles. Synthesize them '
        'into a tight brief: 3 key takeaways and 1 recommendation, resolving any '
        'disagreements.\n\n' + '\n\n'.join(findings),
    )


if __name__ == '__main__':
    ai.run_main(main())
