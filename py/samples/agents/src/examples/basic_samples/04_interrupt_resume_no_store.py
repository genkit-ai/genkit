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

"""Backend: interrupt + resume without store — carry out.state between stream_bidi calls.

Uses ToolApproval middleware: transferMoney is gated until the client resumes with
``restart_tool(..., resumed_metadata={'toolApproved': True})``.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from pydantic import BaseModel, Field

from genkit import Genkit, Tool, restart_tool
from genkit._ai._generate import resolve_tool
from genkit._core._typing import Resume, ToolRequestPart
from genkit.agent import AgentInit
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


async def main() -> None:
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

    conn = await agent.stream_bidi(AgentInit())
    await conn.send_text('Transfer $100 to account 999 for lunch.')
    await conn.close()

    interrupt_trp: ToolRequestPart | None = None
    async for chunk in conn.receive():
        mc = chunk.model_chunk
        if mc and mc.content:
            for part in mc.content:
                tr = part.tool_request
                if tr and tr.name == 'transferMoney':
                    interrupt_trp = part

    out1 = await conn.output()
    if out1.finish_reason != 'interrupted':
        raise RuntimeError(f'expected interrupted, got {out1.finish_reason}')
    if interrupt_trp is None:
        raise RuntimeError('expected transferMoney interrupt chunk')

    transfer_tool = Tool(await resolve_tool(ai.registry, 'transferMoney'))
    restart_part = restart_tool(
        tool=transfer_tool,
        interrupt=interrupt_trp,
        resumed_metadata={'toolApproved': True},
    )

    conn2 = await agent.stream_bidi(AgentInit(state=out1.state))
    await conn2.send_resume(Resume(restart=[restart_part]))
    await conn2.close()

    async for chunk in conn2.receive():
        print('chunk:', chunk.model_dump(by_alias=True, exclude_none=True))

    out2 = await conn2.output()
    print('output:', out2.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    asyncio.run(main())
