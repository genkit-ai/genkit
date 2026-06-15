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

"""ToolApproval interrupt — two invocations, like HTTP+SSE.

Turn 1 streams until ToolApproval interrupts. The client surfaces those tool
calls for human approval; the connection ends (finishReason interrupted) —
nothing auto-approves on the server.

Turn 2 is a fresh stream_bidi: init with sessionId or snapshotId from turn 1,
and data.resume only (no new user message). That matches a second POST after
the user clicks approve in the UI.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from pydantic import BaseModel, Field

from genkit import Genkit, Tool, restart_tool
from genkit._ai._generate import resolve_tool
from genkit._core._typing import Resume, ToolRequestPart
from genkit.agent import AgentInit, InMemorySessionStore
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


class BalanceInput(BaseModel):
    account: str


class BalanceOutput(BaseModel):
    balance: float


def _as_tool_request_part(part) -> ToolRequestPart | None:
    root = getattr(part, 'root', part)
    if not getattr(root, 'tool_request', None):
        return None
    if isinstance(root, ToolRequestPart):
        return root
    if not hasattr(root, 'model_dump'):
        return None
    return ToolRequestPart.model_validate(root.model_dump())


def _tool_request_key(trp: ToolRequestPart) -> str:
    ref = trp.tool_request.ref
    if ref:
        return ref
    inp = trp.tool_request.input
    if isinstance(inp, BaseModel):
        inp = inp.model_dump()
    return f'{trp.tool_request.name}:{json.dumps(inp, sort_keys=True, default=str)}'


def _collect_tool_request(pending: dict[str, ToolRequestPart], part) -> None:
    trp = _as_tool_request_part(part)
    if trp is None:
        return
    key = _tool_request_key(trp)
    existing = pending.get(key)
    if existing is None:
        pending[key] = trp
        return
    if (trp.metadata or {}).get('interrupt') and not (existing.metadata or {}).get('interrupt'):
        pending[key] = trp


def _interrupted_tool_requests(pending: dict[str, ToolRequestPart]) -> list[ToolRequestPart]:
    marked = [trp for trp in pending.values() if (trp.metadata or {}).get('interrupt')]
    return marked if marked else list(pending.values())


async def _build_restarts(ai: Genkit, interrupted: list[ToolRequestPart]) -> list[ToolRequestPart]:
    restarts = []
    for trp in interrupted:
        tool = Tool(await resolve_tool(ai.registry, trp.tool_request.name))
        restarts.append(
            restart_tool(
                tool=tool,
                interrupt=trp,
                resumed_metadata={'toolApproved': True},
            )
        )
    return restarts


async def main() -> None:
    ai = Genkit(plugins=[GoogleAI(), Middleware()])
    tool_approval = ToolApproval(allowed_tools=[])

    @ai.tool(name='transferMoney', description='Transfer money between accounts.')
    async def transfer_money(_input: TransferInput) -> TransferOutput:
        return TransferOutput(success=True, transactionId=f'txn-{uuid4().hex[:12]}')

    @ai.tool(name='checkBalance', description='Check the balance of an account.')
    async def check_balance(_input: BalanceInput) -> BalanceOutput:
        return BalanceOutput(balance=1_234.56)

    store = InMemorySessionStore()

    agent = ai.define_agent(
        name='bankingAgent',
        model='googleai/gemini-flash-latest',
        system=(
            'Banking assistant. When the user asks to transfer money, call '
            'transferMoney and checkBalance for the destination account in the '
            'same turn (both tools).'
        ),
        tools=[transfer_money, check_balance],
        use=[tool_approval],
        store=store,
    )

    session_id = str(uuid4())

    # --- Turn 1 (HTTP POST #1): user message → stream until interrupted ---
    conn1 = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn1.send_text('Transfer $500 to account 12345 for rent and check the balance in account 12345.')
    await conn1.close()

    pending: dict[str, ToolRequestPart] = {}
    async for chunk in conn1.receive():
        print('turn 1 chunk:', chunk.model_dump(by_alias=True, exclude_none=True))
        mc = chunk.model_chunk
        if mc and mc.content:
            for part in mc.content:
                _collect_tool_request(pending, part)

    out1 = await conn1.output()
    if out1.finish_reason != 'interrupted':
        raise RuntimeError(f'expected interrupted, got {out1.finish_reason}')
    if not out1.snapshot_id:
        raise RuntimeError('expected snapshotId on interrupted turn with store')

    interrupted = _interrupted_tool_requests(pending)
    if not interrupted:
        raise RuntimeError('expected at least one tool-request chunk to approve')
    print(
        'client would show approval UI for:',
        [trp.tool_request.name for trp in interrupted],  # pyright: ignore[reportUnknownMemberType]
    )

    restarts = await _build_restarts(ai, interrupted)

    # --- Turn 2 (HTTP POST #2): resume only — user approved in the UI ---
    # sessionId is enough on a linear chat; snapshotId from out1 works too.
    conn2 = await agent.stream_bidi(AgentInit(session_id=session_id))
    await conn2.send_resume(Resume(restart=restarts))
    await conn2.close()

    async for chunk in conn2.receive():
        print('turn 2 chunk (sessionId init):', chunk.model_dump(by_alias=True, exclude_none=True))

    out2 = await conn2.output()
    print('turn 2 output (sessionId init):', out2.model_dump(by_alias=True, exclude_none=True))


if __name__ == '__main__':
    asyncio.run(main())
