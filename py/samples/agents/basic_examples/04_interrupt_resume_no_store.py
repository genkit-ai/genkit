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

"""Backend: interrupt + resume without store — client-managed state using AgentAPI."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field

from genkit import Genkit, restart_tool
from genkit._core._typing import ToolRequest, ToolRequestPart
from genkit.agent import Resume
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
    session = agent.connect()
    # --- Turn 1: user message → stream until interrupted ---
    print('--- SENDING TURN 1 ---')
    turn1 = session.send('Transfer $100 to account 999 for lunch.')
    async for chunk in turn1.stream:
        print('turn 1 chunk:', chunk)

    out1 = await turn1.output
    if out1.finish_reason != 'interrupted':
        raise RuntimeError(f'expected interrupted, got {out1.finish_reason}')

    # Inspect the interrupt to approve
    if turn1.interrupt:
        print(f'client would show approval UI for: {turn1.interrupt.name}')

        trp = ToolRequestPart(
            tool_request=ToolRequest(
                name=turn1.interrupt.name,
                input=turn1.interrupt.input_data,
                ref=turn1.interrupt.ref,
            )
        )
        restarts = [restart_tool(interrupt=trp, resumed_metadata={'tool_approved': True})]

        # --- Turn 2: resume within same session ---
        print('\n--- SENDING TURN 2 (RESUME) ---')
        turn2 = session.resume(Resume(restart=restarts))
        async for chunk in turn2.stream:
            print('turn 2 chunk:', chunk)

        out2 = await turn2.output
        print('turn 2 output:', out2)
    await session.close()


if __name__ == '__main__':
    ai.run_main(main())
