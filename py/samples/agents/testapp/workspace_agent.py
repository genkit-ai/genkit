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

"""An agent that builds up a workspace of files as it goes.

A ``write_artifact`` tool drops named files onto the session. Artifacts stream to
the client as ``artifact`` chunks as they're written and dedupe by name (rewriting
a file replaces it), so the client can render a live file tree. This is the
backbone of any "generate me a project" agent.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any

from _ai import ai
from pydantic import BaseModel

from genkit import ActionRunContext, Part, TextPart
from genkit.agent import Artifact


class WriteArtifactInput(BaseModel):
    name: str
    content: str


@ai.tool(name='write_artifact', description='Create or replace a named file in the workspace.')
async def write_artifact(input: WriteArtifactInput) -> dict[str, str]:
    # Adding to the session is what makes it stream out as an `artifact` chunk and
    # show up in chat.artifacts; same name replaces the prior version.
    if sess := ai.current_session():
        await sess.add_artifacts(Artifact(name=input.name, parts=[Part(TextPart(text=input.content))]))
    return {'name': input.name, 'status': 'written'}


workspace_agent = ai.define_agent(
    name='workspaceAgent',
    system=(
        'You are a helpful code generation assistant. Use the write_artifact tool to create '
        'files (pass the filename as "name" and full contents as "content"). You can create '
        'multiple files in a turn. After writing, briefly confirm what you created.'
    ),
    tools=[write_artifact],
)


@ai.flow()
async def test_workspace_agent(text: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Ask for a file; watch it arrive as an artifact chunk, then in chat.artifacts."""
    chat = workspace_agent.chat()
    turn = chat.send(text or 'Write poem.txt with a short poem about genkit')
    async for chunk in turn:
        if chunk.artifact is not None:
            ctx.send_chunk(f'[artifact] {chunk.artifact.name}')
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn
    return {'text': res.text, 'artifacts': [a.name for a in chat.artifacts]}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
