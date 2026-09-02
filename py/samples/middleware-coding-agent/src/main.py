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

"""Read/write a sandboxed workspace/. Writes pause for y/N."""

from pathlib import Path

from genkit_google_genai import GoogleAI
from genkit_middleware import Filesystem, Middleware, Skills, ToolApproval

from genkit import Genkit, Message, Part, Role, TextPart, restart_tool

here = Path(__file__).resolve().parent.parent
workspace = here / 'workspace'
skills = here / 'skills'

ai = Genkit(
    plugins=[GoogleAI(), Middleware()],
    model=GoogleAI.gemini_model('gemini-flash-latest'),
)

# ToolApproval lets read-only tools run; write_file / edit_file interrupt.
middleware = [
    ToolApproval(allowed_tools=['read_file', 'list_files', 'use_skill']),
    Skills(skill_paths=[str(skills)]),
    Filesystem(root_dir=str(workspace), allow_write_access=True),
]


async def main() -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    messages = [
        Message(
            role=Role.SYSTEM,
            content=[
                Part(
                    root=TextPart(
                        text=(
                            f'You are a coding agent. Working directory is {workspace}. '
                            'Use plain filenames relative to that root. '
                            'Read a file before you edit it. Start by listing the workspace.'
                        )
                    )
                )
            ],
        ),
    ]
    print('Type a request. "exit" to quit.')

    while True:
        try:
            user_input = input('\n> ').strip()
        except EOFError:
            break
        if user_input.lower() == 'exit':
            break
        if not user_input:
            continue

        restart = None
        prompt = user_input
        while True:
            response = await ai.generate(
                prompt=prompt,
                messages=messages,
                resume_restart=restart,
                max_turns=20,
                use=middleware,
            )
            messages = response.messages
            if not response.interrupts:
                print(response.text)
                break

            # Each interrupt is a write. Approve restarts that tool.
            approved = []
            for interrupt in response.interrupts:
                print(f'{interrupt.tool_request.name}: {interrupt.tool_request.input}')
                if input('Approve? (y/N): ').strip().lower() in ('y', 'yes'):
                    approved.append(restart_tool(interrupt=interrupt, resumed_metadata={'tool_approved': True}))
            if not approved:
                print('Denied.')
                break
            restart = approved
            prompt = None


if __name__ == '__main__':
    ai.run_main(main())
