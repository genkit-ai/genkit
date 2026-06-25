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

"""Interrupt + resume without a store — you carry the paused state.

Same human-in-the-loop approval as the stored version, but with no store the paused
turn lives only in this process. You inspect the interrupt, attach your approval as
resume metadata, and continue the same in-memory session. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from genkit import Genkit, restart_tool
from genkit.agent import AgentFinishReason, Resume, ToolRequest, ToolRequestPart
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
tool_approval = ToolApproval(allowed_tools=[])


@ai.tool(name='transferMoney', description='Transfer money.')
async def transfer_money(_input: TransferInput) -> TransferOutput:
    return TransferOutput(success=True, transactionId=f'txn-{uuid4().hex[:12]}')


agent = ai.define_agent(
    name='approvalNoStore',
    model='googleai/gemini-flash-latest',
    system='Banking assistant. Call transferMoney when the user asks to transfer money.',
    tools=[transfer_money],
    use=[tool_approval],
)


async def main() -> None:
    session = agent.chat()

    # ToolApproval pauses before transferMoney runs.
    turn1 = session.send('Transfer $100 to account 999 for lunch.')
    out1 = await turn1.output
    assert out1.finish_reason == AgentFinishReason.INTERRUPTED
    assert turn1.interrupt is not None  # → the pending request your UI would surface for approval

    # Approve it: rebuild the request, tag it approved, and resume in place.
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
    # → resumes and runs the transfer to completion
    await session.resume(Resume(restart=[approved]))
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
