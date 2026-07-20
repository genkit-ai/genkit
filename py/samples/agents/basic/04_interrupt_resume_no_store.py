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
turn lives only in this process. Inspect out.interrupts on the paused response, approve
them in one resume, and continue the same in-memory chat. Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from uuid import uuid4

from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware, ToolApproval
from pydantic import BaseModel, Field

from genkit import Genkit
from genkit.agent import AgentFinishReason


class TransferInput(BaseModel):
    amount: float
    to_account: str = Field(alias='toAccount')


class TransferOutput(BaseModel):
    success: bool
    transaction_id: str = Field(alias='transactionId')


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
    chat = agent.chat()

    out1 = await chat.send('Transfer $100 to account 999 for lunch.').response
    assert out1.finish_reason == AgentFinishReason.INTERRUPTED

    # Approve each pending tool call, then one resume continues the turn.
    restart_parts = [intr.restart(resumed_metadata={'tool_approved': True}) for intr in out1.interrupts]
    out2 = await chat.resume(restart=restart_parts).response
    assert out2.finish_reason == AgentFinishReason.STOP


if __name__ == '__main__':
    ai.run_main(main())
