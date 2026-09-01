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

"""Per-request user/tenant on generate() and tools — not in the prompt."""

from genkit_google_genai import GoogleAI
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit

ai = Genkit(plugins=[GoogleAI()], model=GoogleAI.gemini_model('gemini-flash-latest'))

# What your auth middleware would have already resolved for this request.
ACCOUNTS = {
    ('user-42', 'acme'): 'Ada at Acme (enterprise)',
    ('user-99', 'globex'): 'Ben at Globex (starter)',
}


class AccountNeed(BaseModel):
    need: str


@ai.tool()
async def account_record(input: AccountNeed, ctx: ActionRunContext) -> str:
    # Tenant is not a tool argument, so a prompt cannot hop to another customer.
    key = (str(ctx.context.get('user_id', '')), str(ctx.context.get('tenant_id', '')))
    account = ACCOUNTS.get(key, 'unknown')
    return f'{account}; asked for {input.need}'


async def main() -> None:
    for user_id, tenant_id in (('user-42', 'acme'), ('user-99', 'globex')):
        # context= is the same dict your auth middleware puts on the request.
        # generate() and every tool call inherit it.
        response = await ai.generate(
            prompt='What is included in my plan?',
            system='Call account_record for the signed-in user. You cannot choose a tenant.',
            tools=['account_record'],
            context={'user_id': user_id, 'tenant_id': tenant_id},
        )
        print(f'{tenant_id}/{user_id}: {response.text}')


if __name__ == '__main__':
    ai.run_main(main())
