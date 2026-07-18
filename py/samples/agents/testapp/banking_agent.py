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

"""An agent that stops and asks before doing something irreversible.

The model calls ``userApproval`` first — that tool always interrupts, so the turn
ends with a pending approval request the client can show. After the human responds,
one ``resume`` with the answer continues and the model can call ``transferMoney``.
The store keeps the paused session alive across the two HTTP requests.

Requires GEMINI_API_KEY.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from _ai import ai
from pydantic import BaseModel, Field

from genkit import ActionRunContext
from genkit.agent import InMemorySessionStore


class UserApprovalInput(BaseModel):
    action: str
    details: str


user_approval = ai.define_interrupt(
    name='userApproval',
    description='Ask the user for approval before proceeding with a sensitive action.',
    input_schema=UserApprovalInput,
)


class TransferInput(BaseModel):
    amount: float
    to_account: str = Field(alias='toAccount')

    model_config = {'populate_by_name': True}


class TransferOutput(BaseModel):
    success: bool
    transaction_id: str = Field(alias='transactionId')

    model_config = {'populate_by_name': True}


@ai.tool(name='transferMoney', description='Transfer money to a specified account.')
async def transfer_money(_input: TransferInput) -> TransferOutput:
    return TransferOutput(success=True, transactionId=f'txn-{uuid4().hex[:12]}')


banking_agent = ai.define_agent(
    name='bankingAgent',
    system=(
        'You are a helpful banking assistant. If the user wants to transfer money, '
        'ALWAYS use the userApproval interrupt to confirm the details before executing '
        'the transferMoney tool.'
    ),
    tools=[user_approval, transfer_money],
    store=InMemorySessionStore(),
)


@ai.flow()
async def test_banking_agent(text: str, ctx: ActionRunContext) -> dict[str, Any]:
    """Run a turn that pauses for approval, approve it, then let the transfer land."""
    chat = banking_agent.chat()
    turn = chat.send(text or 'Transfer $500 to my savings account.')
    async for chunk in turn:
        if chunk.text:
            ctx.send_chunk(chunk.text)
    res = await turn

    paused_for_approval = bool(res.interrupts)
    if paused_for_approval:
        ctx.send_chunk('[interrupted] approving pending action…')
        approval = next((i for i in res.interrupts if i.name == 'userApproval'), None)
        if approval is not None:
            resume_turn = chat.resume(respond=[approval.respond({'approved': True, 'feedback': 'Looks good'})])
            async for chunk in resume_turn:
                if chunk.text:
                    ctx.send_chunk(chunk.text)
            res = await resume_turn

    return {'text': res.text, 'paused_for_approval': paused_for_approval}


if __name__ == '__main__':
    import asyncio

    ai.run_main(asyncio.sleep(0))
