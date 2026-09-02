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

"""A tool can stop and ask a human before it finishes. Then you resume."""

from genkit_google_genai import GoogleAI
from pydantic import BaseModel, Field

from genkit import Genkit, Interrupt, ToolRunContext, respond_to_interrupt, restart_tool

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


class TransferRequest(BaseModel):
    to_account: str = Field(description='recipient name or masked account')
    amount_usd: str = Field(description='amount as a string, e.g. 250.00')
    memo: str = ''


@ai.tool()
async def request_transfer(body: TransferRequest, ctx: ToolRunContext) -> dict:
    # First call pauses. After you resume, is_resumed() is True and the
    # wire actually runs.
    if not ctx.is_resumed():
        raise Interrupt({'summary': f'Wire ${body.amount_usd} to {body.to_account} — {body.memo}'})
    return {'status': 'confirmed', 'resumed': ctx.resumed_metadata}


async def main() -> None:
    response = await ai.generate(
        prompt='Please wire $250.00 to Jane Doe (account ending in 4521) for April rent.',
        system='You are a treasury desk. Call request_transfer to send money.',
        tools=[request_transfer],
    )
    print(response.text)

    if not response.interrupts:
        return

    interrupt = response.interrupts[0]
    print(f'paused: {interrupt.metadata}')

    # restart_tool re-runs the tool after the human says yes.
    approved = await ai.generate(
        messages=response.messages,
        resume_restart=restart_tool(interrupt=interrupt, resumed_metadata={'approved': True}),
        tools=[request_transfer],
    )
    print(approved.text)

    # respond_to_interrupt injects a result instead — decline without
    # sending the wire.
    declined = await ai.generate(
        prompt='Please wire $80.00 to Sam Lee (account ending in 9910) for lunch.',
        system='You are a treasury desk. Call request_transfer to send money.',
        tools=[request_transfer],
    )
    if not declined.interrupts:
        return
    print(f'paused: {declined.interrupts[0].metadata}')

    done = await ai.generate(
        messages=declined.messages,
        resume_respond=respond_to_interrupt({'status': 'declined'}, interrupt=declined.interrupts[0]),
        tools=[request_transfer],
    )
    print(done.text)


if __name__ == '__main__':
    ai.run_main(main())
