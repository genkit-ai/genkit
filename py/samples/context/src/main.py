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

"""Context - pass request data through `generate()`, flows, and tools. Requires GEMINI_API_KEY.

Run directly:
    uv run src/main.py
Or inspect live execution and traces in Dev UI:
    genkit start -- uv run src/main.py
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from genkit import ActionRunContext, Genkit
from genkit.plugins.google_genai import GoogleAI

# 1. Initialize Genkit and mock database
ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-flash-latest')

USERS: dict[int, dict[str, str]] = {
    42: {'name': 'Arthur Dent', 'plan': 'premium'},
    123: {'name': 'Jane Doe', 'plan': 'enterprise'},
}


class ContextInput(BaseModel):
    user_id: int = Field(default=42, description='Try 42 or 123')


# 2. Define tools that read request-scoped context via Genkit.current_context()
@ai.tool()
async def get_user_info() -> str:
    """Read user info from `Genkit.current_context()` without passing parameters."""
    ctx = Genkit.current_context() or {}
    user_id = int(ctx.get('user_id', 0))  # type: ignore[arg-type]
    user = USERS.get(user_id, {'name': 'Unknown', 'plan': 'none'})
    return f'{user["name"]} ({user["plan"]} plan)'


# 3. Pass context into generate() — tools inspect and inherit it automatically across turns
@ai.flow()
async def context_in_generate(input: ContextInput) -> str:
    """Pass context into `ai.generate()` and let a tool read it."""
    response = await ai.generate(
        prompt='Look up the current user and state their name and plan.',
        tools=[get_user_info],
        context={'user_id': input.user_id},
    )
    # => The current user is Arthur Dent, who is currently on the premium plan.
    return response.text


@ai.flow()
async def context_in_flow(input: ContextInput, ctx: ActionRunContext) -> str:
    """Access request context directly inside a flow."""
    # => Flow context: {'user_id': 42}. Requested user: 42.
    return f'Flow context: {ctx.context}. Requested user: {input.user_id}.'


async def main() -> None:
    """Run the context demo from the CLI."""
    try:
        # 1. Pass request context into `ai.generate(...)` (`context_in_generate`)
        print(await context_in_generate(ContextInput(user_id=42)))
        # => The current user is Arthur Dent, who is currently on the premium plan.

        # 2. Invoke a flow (`.run(...)`) explicitly passing `context={'user_id': 42}` so `ctx.context` receives it
        flow_res = await context_in_flow.run(ContextInput(user_id=42), context={'user_id': 42})
        print(flow_res.response)
        # => Flow context: {'user_id': 42}. Requested user: 42.
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')


if __name__ == '__main__':
    ai.run_main(main())
