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

"""Agentic coding REPL — Filesystem + Skills + ToolApproval middleware.

An interactive coding agent that reads, edits, and writes files inside a
sandboxed ``workspace/`` directory. Read-only tools (``read_file``,
``list_files``, ``use_skill``) run automatically; everything that can
mutate the workspace (``write_file``, ``edit_file``) is gated by
``ToolApproval``, so the CLI pauses and asks ``y/N`` before each write.

The agent state — middleware instances and message history — is owned by a
``CodingAgent`` session object built once per ``main()`` invocation.
That ties every ``ai.generate()`` and resume in this REPL to the same
middleware stack. ``Filesystem`` itself keeps no cross-call cache; file
content reaches the model through enqueued messages inside each call.

Re-running cleanly:

* The agent mutates files in ``workspace/`` directly. To start over,
  ``rm -rf workspace/*`` — the directory itself is recreated on next run.
"""

from pathlib import Path

from genkit_google_genai import GoogleAI
from genkit_middleware import Filesystem, Middleware, Skills, ToolApproval

from genkit import Genkit, Message, ModelResponse, Part, Role, TextPart, ToolRequestPart, restart_tool

_HERE = Path(__file__).resolve().parent.parent
_WORKSPACE = _HERE / 'workspace'
_SKILLS = _HERE / 'skills'

ai = Genkit(
    plugins=[GoogleAI(), Middleware()],
    model='googleai/gemini-flash-latest',
)


SYSTEM_PROMPT = (
    'You are a helpful coding agent. Very terse but thoughtful and careful.\n'
    f'Your working directory is {_WORKSPACE}, you are not allowed to access anything outside it.\n'
    'Use plain filenames relative to the workspace root (e.g. ``foo.py``, not ``./foo.py`` '
    'or absolute paths). You must ``read_file`` an existing file before you can ``write_file`` '
    'or ``edit_file`` it — new files do not need a prior read.\n'
    'Use skills. ALWAYS start by analyzing the current state of the workspace, '
    'there might be something already there.'
)


class CodingAgent:
    """One agent session: owns the middleware stack and the running conversation."""

    def __init__(self) -> None:
        self.middleware = [
            ToolApproval(allowed_tools=['read_file', 'list_files', 'use_skill']),
            Skills(skill_paths=[str(_SKILLS)]),
            Filesystem(root_dir=str(_WORKSPACE), allow_write_access=True),
        ]
        self.messages: list[Message] = [
            Message(role=Role.SYSTEM, content=[Part(root=TextPart(text=SYSTEM_PROMPT))]),
        ]

    async def turn(self, user_input: str) -> ModelResponse:
        """Drive one user turn to completion across any number of approval prompts."""
        restart: list[ToolRequestPart] | None = None
        while True:
            response = await ai.generate(
                prompt=user_input if restart is None else None,
                messages=self.messages,
                resume_restart=restart,
                max_turns=20,
                use=self.middleware,
            )
            if not response.interrupts:
                self.messages = response.messages
                return response

            approved = await _ask_for_approvals(response.interrupts)
            if not approved:
                print('Tool denied.')  # noqa: T201
                self.messages = response.messages
                return response

            print('Resuming...')  # noqa: T201
            restart = approved
            self.messages = response.messages


async def _ask_for_approvals(interrupts: list[ToolRequestPart]) -> list[ToolRequestPart]:
    """Prompt the user y/N for each pending interrupt; return the approved restart parts."""
    approved: list[ToolRequestPart] = []
    for trp in interrupts:
        print('\n*** Tool Approval Required ***')  # noqa: T201
        print(f'Tool:  {trp.tool_request.name}')  # noqa: T201
        print(f'Input: {trp.tool_request.input}')  # noqa: T201
        if input('Approve? (y/N): ').strip().lower() in ('y', 'yes'):
            approved.append(
                restart_tool(interrupt=trp, resumed_metadata={'toolApproved': True}),
            )
    return approved


async def main() -> None:
    """Interactive REPL — one ``CodingAgent`` per process, one ``turn()`` per user line."""
    _WORKSPACE.mkdir(parents=True, exist_ok=True)

    print('--- Coding Agent ---')  # noqa: T201
    print('Type your request. To exit, type "exit".')  # noqa: T201

    agent = CodingAgent()

    while True:
        try:
            user_input = input('\n> ').strip()
        except EOFError:
            break
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue

        try:
            response = await agent.turn(user_input)
        except Exception as e:  # noqa: BLE001 - top-level REPL: surface, don't crash
            print(f'Error during generation: {e}')  # noqa: T201
            continue

        print(f'\nAI Response:\n{response.text}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
