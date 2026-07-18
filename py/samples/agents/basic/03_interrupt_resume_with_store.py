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
A human approves via out.interrupts, then one resume covers every pending tool call.
The store keeps the paused session alive between requests, and the tool finally executes.
Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from uuid import uuid4

from genkit_google_genai import GoogleAI
from genkit_middleware import Middleware, ToolApproval
from pydantic import BaseModel, Field

from genkit import Genkit, ToolRequestPart
from genkit.agent import (
    AgentFinishReason,
    InMemorySessionStore,
)


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


store = InMemorySessionStore()

agent = ai.define_agent(
    name='bankingAgent',
    model='googleai/gemini-flash-latest',
    system='Banking assistant. Call transferMoney when the user asks to transfer money.',
    tools=[transfer_money],
    use=[tool_approval],
    store=store,
)


async def main() -> None:
    chat = agent.chat()

    out1 = await chat.send('Transfer $500 to account 12345 for rent.')
    # → finish_reason INTERRUPTED; transferMoney is pending, not executed yet
    assert out1.finish_reason == AgentFinishReason.INTERRUPTED

    # Human approves each pending tool call, then one resume continues the turn.
    restart_parts: list[ToolRequestPart] = [
        intr.restart(resumed_metadata={'tool_approved': True}) for intr in out1.interrupts
    ]
    out2 = await chat.resume(restart=restart_parts)
    assert out2.finish_reason == AgentFinishReason.STOP


if __name__ == '__main__':
    ai.run_main(main())
