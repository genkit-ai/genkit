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

"""A coding assistant that edits a sandboxed workspace, asking before it writes.

The Filesystem middleware gives the agent list/read/write/edit tools scoped to
one directory it can't escape. ToolApproval auto-approves the read-only tools but
pauses on every write, so the human sees each change before it lands. Two little
browser flows let the web UI render the resulting file tree. A file store keeps
long coding sessions alive across restarts.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from _ai import ai

from genkit import ActionRunContext
from genkit.agent import FileSessionStore
from genkit.plugins.middleware import Filesystem, ToolApproval

WORKSPACE_DIR = Path(__file__).resolve().parent / 'workspace'
WORKSPACE_DIR.mkdir(exist_ok=True)


coding_agent = ai.define_agent(
    name='codingAgent',
    description='An AI coding assistant that reads, creates, and edits files in a sandboxed workspace.',
    system=(
        'You are an expert coding assistant working in a sandboxed workspace.\n'
        '- Use list_files and read_file to explore before changing anything.\n'
        '- Use write_file for new files, edit_file for surgical changes.\n'
        '- Explain what you are about to do, then confirm what you did.\n'
        '- Work one step at a time; use markdown and fenced code blocks.'
    ),
    use=[
        # Reads run freely; writes and edits pause for the user to approve — so
        # ToolApproval has to see the tool call before Filesystem executes it.
        ToolApproval(allowed_tools=['list_files', 'read_file']),
        Filesystem(root_dir=str(WORKSPACE_DIR), allow_write_access=True),
    ],
    store=FileSessionStore('./.snapshots-coding'),
    max_turns=30,
)


@ai.flow()
async def test_coding_agent(text: str, ctx: ActionRunContext) -> str:
    """Auto-approve every write so the agent can finish a task unattended."""
    chat = coding_agent.chat()
    turn = chat.send(text or 'Create a Python hello world file called hello.py in the workspace.')
    async for chunk in turn:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn

    # Approve pending writes in a loop until the agent runs out of them.
    for _ in range(10):
        if not res.interrupts:
            break
        ctx.send_chunk(f'[auto-approving] {", ".join(i.name for i in res.interrupts)}')
        restart = [i.restart(resumed_metadata={'tool_approved': True}) for i in res.interrupts]
        resume_turn = chat.resume(restart=restart)
        async for chunk in resume_turn:
            if chunk.text:
                ctx.send_chunk(chunk.text)
        res = await resume_turn

    return res.text


# --- Workspace browser flows (served at /api/workspace/files and /file) --------


def _walk(directory: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name)):
        if entry.name.startswith('.'):
            continue
        rel = str(entry.relative_to(WORKSPACE_DIR))
        if entry.is_dir():
            entries.append({'name': entry.name, 'path': rel, 'type': 'directory', 'children': _walk(entry)})
        else:
            entries.append({'name': entry.name, 'path': rel, 'type': 'file'})
    return entries


@ai.flow()
async def list_workspace_files(_: Any = None) -> dict[str, Any]:
    """The file tree the web UI renders beside the chat."""
    return {'files': _walk(WORKSPACE_DIR)}


@ai.flow()
async def read_workspace_file(path: str) -> dict[str, str]:
    """Read one file, refusing anything that tries to climb out of the workspace."""
    full = (WORKSPACE_DIR / path).resolve()
    if os.path.commonpath([str(full), str(WORKSPACE_DIR)]) != str(WORKSPACE_DIR):
        raise ValueError('Path outside workspace')
    return {'path': path, 'content': full.read_text()}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
