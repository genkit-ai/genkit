#!/usr/bin/env python3
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

"""Tool interrupts (Restart path) — pause before a sensitive tool runs, then restart upon human approval.

Run:
    uv run src/approval_example.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from genkit import Genkit, Interrupt, ToolRunContext, restart_tool
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')


class TransferRequest(BaseModel):
    to_account: str = Field(description='Recipient account identifier')
    amount_usd: str = Field(description='Amount to transfer, e.g. 250.00')


# 1. Define a tool that pauses execution on first run using `raise Interrupt(...)`
@ai.tool()
async def request_transfer(body: TransferRequest, ctx: ToolRunContext) -> dict:
    """First run: interrupt for approval. Second run (when resumed via `restart_tool`): confirm wire."""
    if not ctx.is_resumed():
        # => Interrupt raised: execution pauses, returning `response.interrupts` to the caller
        raise Interrupt({'summary': f'Wire ${body.amount_usd} to {body.to_account}', 'needs_approval': True})
    # => Returned when `request_transfer` re-runs after human approval:
    return {'status': 'confirmed', 'resumed': ctx.resumed_metadata}


async def main() -> None:
    """Drive tool interruption and resume execution using `restart_tool` (`resume_restart`)."""
    try:
        # 2. First `generate` call triggers the tool and pauses at `raise Interrupt`
        res1 = await ai.generate(
            prompt='Please wire $250.00 to Jane Doe.',
            tools=[request_transfer],
        )
        interrupt = res1.interrupts[0]
        summary = interrupt.metadata.get('summary', '') if interrupt.metadata else ''
        print(f'Interrupt paused execution. Summary: {summary}')
        # => Interrupt paused execution. Summary: Wire $250.00 to Jane Doe

        # 3. Human approves -> `restart_tool(...)` re-runs the tool (`ctx.is_resumed() == True`)
        approved = restart_tool(interrupt=interrupt, resumed_metadata={'tool_approved': True})
        res2 = await ai.generate(
            messages=res1.messages,
            resume_restart=approved,
            tools=[request_transfer],
        )
        print(res2.text)
        # => "I have successfully wired $250.00 to Jane Doe."
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
