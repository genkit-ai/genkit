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

"""Pause a turn for human approval, then resume it.

ToolApproval interrupts the turn before a sensitive tool runs, so it ends with
finish_reason INTERRUPTED and a pending tool request instead of moving the money.
A human approves, you resume the same session, and the tool finally executes. The
store lets the paused turn survive between requests. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from genkit import Genkit, restart_tool
from genkit.agent import (
    AgentFinishReason,
    InMemoryLinearSessionStore,
    Resume,
    ToolRequest,
    ToolRequestPart,
)
from genkit.plugins.google_genai import GoogleAI
from genkit.plugins.middleware import Middleware, ToolApproval


class TransferInput(BaseModel):
    amount: float
    to_account: str = Field(alias='toAccount')

    model_config = {'populate_by_name': True}


class TransferOutput(BaseModel):
    success: bool
    transaction_id: str = Field(alias='transactionId')

    model_config = {'populate_by_name': True}


ai = Genkit(plugins=[GoogleAI(), Middleware()])
tool_approval = ToolApproval(allowed_tools=[])  # empty list ⇒ every tool needs approval


@ai.tool(name='transferMoney', description='Transfer money between accounts.')
async def transfer_money(_input: TransferInput) -> TransferOutput:
    return TransferOutput(success=True, transactionId=f'txn-{uuid4().hex[:12]}')


store = InMemoryLinearSessionStore()

agent = ai.define_agent(
    name='bankingAgent',
    model='googleai/gemini-flash-latest',
    system='Banking assistant. Call transferMoney when the user asks to transfer money.',
    tools=[transfer_money],
    use=[tool_approval],
    store=store,
)


async def main() -> None:
    session = agent.chat()

    # The model wants to move money, but ToolApproval pauses before the tool runs.
    turn1 = session.send('Transfer $500 to account 12345 for rent.')
    out1 = await turn1.output
    # → the turn stops with a pending tool request instead of executing it
    assert out1.finish_reason == AgentFinishReason.INTERRUPTED
    assert turn1.interrupt is not None  # the tool call awaiting a human decision

    # A human approves: rebuild the pending request, tag it approved, and resume.
    approved = restart_tool(
        interrupt=ToolRequestPart(
            tool_request=ToolRequest(
                name=turn1.interrupt.name,
                input=turn1.interrupt.input,
                ref=turn1.interrupt.ref,
            )
        ),
        resumed_metadata={'tool_approved': True},
    )
    # → resumes, runs transferMoney now that it's cleared, and finishes the turn
    out2 = await session.resume(Resume(restart=[approved]))
    assert out2.finish_reason == AgentFinishReason.STOP  # → the transfer runs and the turn completes
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
