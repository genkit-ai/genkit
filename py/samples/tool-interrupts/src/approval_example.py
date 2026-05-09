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

"""**Bank transfer approval** — human-in-the-loop before a transfer tool finishes.

The model calls ``request_transfer``; the CLI asks **approve (y)** or **decline (n)**.
**Approve** → ``restart_tool(...)`` / ``resume_restart`` so the tool **runs again**
with ``ToolRunContext.is_resumed``. **Decline** → ``respond_to_interrupt`` /
``resume_respond`` (no second tool run).

Opening prompt, then one **canned user message** (no typing) so the model calls
``request_transfer``; you still answer **y/n** for approval. Run::

    uv run src/approval_example.py

For the trivia-only **respond** demo, see ``respond_example.py``. See README.md.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from genkit import (
    Genkit,
    Interrupt,
    ToolRunContext,
    respond_to_interrupt,
    restart_tool,
)
from genkit.model import ModelResponse
from genkit.plugins.google_genai import GoogleAI  # pyright: ignore[reportMissingImports]

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / 'prompts'

_BAR = '=' * 52
_RULE = '-' * 52

# Canned user line so the demo always triggers a transfer tool call without stdin.
USER_MESSAGE = 'Please wire $250.00 to Jane Doe (account ending in 4521) for April rent.'


def _print_intro() -> None:
    print(f'\n{_BAR}')
    print('  Bank transfer demo — outgoing wires need your approval in the CLI.')
    print(_BAR)
    print('  1) The banker speaks first.')
    print('  2) A scripted user message asks for a wire; the model calls the transfer tool.')
    print('  3) When asked y/n: yes = approve (tool runs again); no = decline.')
    print(f'{_BAR}\n')


def _print_scripted_user_turn() -> None:
    print(_RULE)
    print('Scripted user message (see USER_MESSAGE in source):')
    print(_RULE)


def _print_waiting_opening() -> None:
    print('Starting: banker opening (please wait)...\n')


def _print_model_turn(label: str, r: ModelResponse) -> None:
    print(f'\n[{label}]')
    if r.text:
        print(r.text)


def _print_transfer_approval_prompt(summary: str) -> None:
    print('\n' + _BAR)
    print('  TRANSFER APPROVAL — y = approve (rerun tool)  |  n = decline')
    print(_BAR)
    if summary:
        print(f'  {summary}')


def _print_unexpected_tool(name: str) -> None:
    print(f'Unexpected tool: {name!r}')


ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-3-flash-preview',
    prompt_dir=_PROMPTS_DIR,
)


class TransferRequest(BaseModel):
    """Wire transfer the user asked for; shown again before approval."""

    to_account: str = Field(description='recipient name or masked account identifier')
    amount_usd: str = Field(description='amount as a string, e.g. 250.00')
    memo: str = Field(default='', description='short reason (rent, invoice, gift, …)')


@ai.tool()
async def request_transfer(body: TransferRequest, ctx: ToolRunContext) -> dict:
    """First run: interrupt for approval. After approval: return confirmation with metadata."""
    if not ctx.is_resumed():
        line = f'Wire ${body.amount_usd} to {body.to_account}'
        if body.memo:
            line = f'{line} — {body.memo}'
        raise Interrupt({
            'summary': line,
            'to_account': body.to_account,
            'amount_usd': body.amount_usd,
            'memo': body.memo,
            'needs_approval': True,
        })
    return {'status': 'confirmed', 'resumed': ctx.resumed_metadata}


async def interactive_restart_cli() -> None:
    """Opening prompt, scripted user line, then transfer approval via ``request_transfer``."""

    _print_intro()

    _print_waiting_opening()
    response = await ai.prompt('bank_transfer_host_cli')()
    messages = response.messages
    _print_model_turn('Banker (opening)', response)

    _print_scripted_user_turn()
    user_said = USER_MESSAGE
    print(f'\nYou: {user_said}\n')

    response = await ai.generate(
        messages=messages,
        prompt=user_said,
        tools=[request_transfer],
    )
    messages = response.messages
    _print_model_turn('Banker', response)

    while response.interrupts:
        interrupt = response.interrupts[0]
        name = interrupt.tool_request.name
        if name != request_transfer.name:
            _print_unexpected_tool(name)
            return

        meta = interrupt.metadata.get('interrupt') if interrupt.metadata else True
        summary = meta.get('summary', '') if isinstance(meta, dict) else ''
        _print_transfer_approval_prompt(summary)
        ans = input('Approve transfer? [y/N]: ').strip().lower()

        if ans in ('y', 'yes'):
            restart = restart_tool(
                tool=request_transfer,
                interrupt=interrupt,
                resumed_metadata={'via': 'cli', 'path': 'restart'},
            )
            response = await ai.generate(
                messages=messages,
                resume_restart=restart,
                tools=[request_transfer],
            )
        else:
            decline_response = respond_to_interrupt(
                {'status': 'declined'},
                interrupt=interrupt,
                metadata={'source': 'cli', 'path': 'respond_decline'},
            )
            response = await ai.generate(
                messages=messages,
                resume_respond=decline_response,
                tools=[request_transfer],
            )
        messages = response.messages
        _print_model_turn('Banker (after your decision)', response)


async def main() -> None:
    await interactive_restart_cli()


if __name__ == '__main__':
    ai.run_main(main())
