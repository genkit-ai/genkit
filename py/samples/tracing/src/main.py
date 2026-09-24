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

"""Every step is a span. Open Dev UI after this runs and click the trace."""

from typing import Literal

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import Genkit

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))


class Ticket(BaseModel):
    account_id: str = 'acc_42'
    message: str = 'Charged twice for ORD-8891 and it still has not arrived.'


class Classification(BaseModel):
    priority: Literal['low', 'normal', 'urgent']
    category: Literal['billing', 'shipping', 'account', 'other']


class Triage(BaseModel):
    priority: Literal['low', 'normal', 'urgent']
    category: Literal['billing', 'shipping', 'account', 'other']
    reply: str


@ai.flow()
async def triage_ticket(ticket: Ticket) -> Triage:
    # The lookup is a real step on the ticket path, so it gets its own
    # span next to the model calls — that's what you inspect when a
    # reply looks wrong.
    async def lookup_account() -> dict[str, str]:
        if ticket.account_id != 'acc_42':
            return {'name': 'Unknown'}
        return {
            'name': 'Maya Chen',
            'plan': 'pro',
            'last_order': 'ORD-8891',
            'last_order_status': 'in_transit',
        }

    account = await ai.run(name='lookup_account', fn=lookup_account)

    classified = await ai.generate(
        prompt=f'Classify this support ticket.\nAccount: {account}\nMessage: {ticket.message}',
        output_schema=Classification,
    )
    classification = classified.output
    if classification is None:
        raise RuntimeError('Model did not return a classification')

    drafted = await ai.generate(
        prompt=(
            'Draft a short, specific support reply. Use the account facts. '
            'Do not invent order numbers.\n'
            f'Account: {account}\n'
            f'Classification: {classification}\n'
            f'Message: {ticket.message}'
        ),
    )
    return Triage(
        priority=classification.priority,
        category=classification.category,
        reply=drafted.text,
    )


async def main() -> None:
    print((await triage_ticket(Ticket())).model_dump_json(indent=2))


if __name__ == '__main__':
    ai.run_main(main())
