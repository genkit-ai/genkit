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

"""Let the model write files into the session as artifacts.

With the Artifacts middleware the model can create named files (here poem.txt) and
they land on the session as structured artifacts you can read back — the basis for
agents that build up a workspace of documents. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from genkit_google_genai import GoogleAI
from genkit_middleware import Artifacts, Middleware

from genkit import Genkit
from genkit.agent import InMemorySessionStore

ai = Genkit(plugins=[GoogleAI(), Middleware()])
store = InMemorySessionStore()

agent = ai.define_agent(
    name='workspaceAgent',
    model='googleai/gemini-flash-latest',
    use=[Artifacts()],
    store=store,
)


async def main() -> None:
    chat = agent.chat()

    # The model writes a named file; the Artifacts middleware captures it on the chat.
    await chat.send('Write poem.txt with a short poem about Python agents.').response
    # → chat.artifacts now contains poem.txt holding the generated poem
    assert any(a.name == 'poem.txt' for a in chat.artifacts)


if __name__ == '__main__':
    ai.run_main(main())
