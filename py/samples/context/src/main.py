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

"""Context - pass request data through `generate()`, flows, and tools."""

from pydantic import BaseModel, Field

from genkit import Genkit
from genkit._core._action import ActionRunContext
from genkit.plugins.google_genai import GoogleAI

ai = Genkit(plugins=[GoogleAI()], model='googleai/gemini-2.5-flash')

USERS: dict[int, dict[str, str]] = {
    42: {'name': 'Arthur Dent', 'plan': 'premium'},
    123: {'name': 'Jane Doe', 'plan': 'enterprise'},
    999: {'name': 'Guest User', 'plan': 'free'},
}


class ContextInput(BaseModel):
    """Input for context flows."""

    user_id: int = Field(default=42, description='Try 42, 123, or 999')


def _current_user() -> dict[str, str]:
    """Read the current user record from execution context."""

    context = Genkit.current_context() or {}
    raw_user = context.get('user')
    if not isinstance(raw_user, dict):
        return {'name': 'Unknown', 'plan': 'none'}
    user_id = int(raw_user.get('id', 0))  # type: ignore[arg-type]
    return USERS.get(user_id, {'name': 'Unknown', 'plan': 'none'})


@ai.tool()
async def get_user_info() -> str:
    """Read user info from `Genkit.current_context()`."""

    user = _current_user()
    return f'{user["name"]} ({user["plan"]} plan)'


@ai.tool()
async def get_user_permissions() -> str:
    """Read plan-based permissions from execution context."""

    plan = _current_user()['plan']
    permissions = {
        'free': 'read-only access',
        'premium': 'read-write access',
        'enterprise': 'admin access',
        'none': 'no access',
    }
    return permissions.get(plan, 'unknown access')


@ai.flow()
async def context_in_generate(input: ContextInput) -> str:
    """Pass context into `ai.generate()` and let a tool read it."""

    response = await ai.generate(
        prompt='Look up the current user.',
        tools=['get_user_info'],
        context={'user': {'id': input.user_id}},
    )
    return response.text


@ai.flow()
async def context_in_flow(input: ContextInput, ctx: ActionRunContext) -> str:
    """Access request context directly inside a flow."""

    return f'Flow context: {ctx.context}. Requested user: {input.user_id}.'


@ai.flow()
async def context_propagation_chain(input: ContextInput) -> str:
    """Show that nested `generate()` calls inherit context automatically."""

    first_response = await ai.generate(
        prompt='Look up the current user.',
        tools=['get_user_info'],
        context={'user': {'id': input.user_id}},
    )
    second_response = await ai.generate(
        prompt=f'The user is {first_response.text}. What permissions do they have?',
        tools=['get_user_permissions'],
    )
    return f'User: {first_response.text}\nPermissions: {second_response.text}'


async def main() -> None:
    """Run the context demos once."""
    try:
        print(await context_in_generate(ContextInput()))  # noqa: T201
        print(await context_propagation_chain(ContextInput()))  # noqa: T201
    except Exception as error:
        print(f'Set GEMINI_API_KEY to a valid value before running this sample directly.\n{error}')  # noqa: T201


if __name__ == '__main__':
    ai.run_main(main())
