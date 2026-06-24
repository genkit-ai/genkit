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

"""Backend: define_prompt_agent using AgentAPI."""

from __future__ import annotations

from genkit import Genkit
from genkit.agent import InMemoryLatestStateStore
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()])
store = InMemoryLatestStateStore()

ai.define_prompt(
    name='greeterPrompt',
    model='googleai/gemini-flash-latest',
    system='You are a greeter. Be warm and brief.',
)
agent = ai.define_prompt_agent(name='greeterPrompt', store=store)


async def main() -> None:
    session = agent.chat()
    print('--- SENDING TURN ---')
    async for chunk in session.send('Hello!'):
        print('chunk:', chunk)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
