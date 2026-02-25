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

"""Context propagation demo - How context flows through Genkit.

This sample demonstrates the four main ways to use context in Genkit:

1. Passing context to ``ai.generate(context=...)`` so tools can read it.
2. Accessing context inside a flow via ``ActionRunContext.context``.
3. Reading context from anywhere via the static ``Genkit.current_context()``.
4. Verifying context propagates through nested generate/tool chains.

Key Concepts (ELI5)::

    ┌─────────────────────┬────────────────────────────────────────────────────┐
    │ Concept             │ ELI5 Explanation                                   │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ Context             │ A dictionary of data (like user info or auth)      │
    │                     │ that follows a request through the system.         │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ContextVar          │ Python's way to store data per-task. Like a        │
    │                     │ backpack each async task carries around.           │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ActionRunContext     │ The object flows/tools receive with context,       │
    │                     │ streaming info, etc. The "execution envelope."     │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ ToolRunContext       │ Same as ActionRunContext but for tools. Also       │
    │                     │ has .interrupt() for human-in-the-loop.            │
    ├─────────────────────┼────────────────────────────────────────────────────┤
    │ current_context()   │ Static method to read context from anywhere.       │
    │                     │ No need to pass ctx around -- just call it.        │
    └─────────────────────┴────────────────────────────────────────────────────┘

Data Flow::

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                  HOW CONTEXT PROPAGATES IN GENKIT                       │
    │                                                                         │
    │   ai.generate(context={'user': {'id': 42}})                             │
    │        │                                                                │
    │        │  (1) Context stored in ContextVar                              │
    │        ▼                                                                │
    │   ┌──────────────┐                                                      │
    │   │ Model Call    │  Model decides to call a tool                        │
    │   └──────┬───────┘                                                      │
    │          │                                                              │
    │          │  (2) Tool receives ToolRunContext with same context           │
    │          ▼                                                              │
    │   ┌──────────────┐                                                      │
    │   │ Tool          │  ctx.context['user']['id'] == 42                     │
    │   │ (get_user)    │  Genkit.current_context()['user']['id'] == 42       │
    │   └──────┬───────┘                                                      │
    │          │                                                              │
    │          │  (3) Nested generate inherits context automatically          │
    │          ▼                                                              │
    │   ┌──────────────┐                                                      │
    │   │ Nested Tool   │  Still sees context['user']['id'] == 42             │
    │   └──────────────┘                                                      │
    └─────────────────────────────────────────────────────────────────────────┘

Testing Instructions
====================
1. Set ``GEMINI_API_KEY`` environment variable.
2. Run ``./run.sh`` from this sample directory.
3. Open the DevUI at http://localhost:4000.
4. Run each flow and verify context-dependent behavior.

See README.md for the full testing checklist.
"""

import asyncio
import os

from pydantic import BaseModel, Field

from genkit.ai import Genkit
from genkit.core.action import ActionRunContext
from genkit.core.logging import get_logger
from genkit.plugins.google_genai import GoogleAI
from samples.shared.logging import setup_sample

setup_sample()

if 'GEMINI_API_KEY' not in os.environ:
    os.environ['GEMINI_API_KEY'] = input('Please enter your GEMINI_API_KEY: ')

logger = get_logger(__name__)

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.5-flash',
)


def _user_from_context(context: dict[str, object]) -> tuple[int, dict[str, str]]:
    """Extract user_id and record from a context dict.

    Args:
        context: The Genkit action context dict.

    Returns:
        A (user_id, record) tuple with defaults for unknown users.
    """
    default_record: dict[str, str] = {'name': 'Unknown', 'role': 'unknown', 'plan': 'none'}
    raw_user = context.get('user')
    if not isinstance(raw_user, dict):
        return 0, default_record
    user_id = int(raw_user.get('id', 0))  # type: ignore[arg-type]
    return user_id, MOCK_USER_DB.get(user_id, default_record)


MOCK_USER_DB: dict[int, dict[str, str]] = {
    42: {'name': 'Arthur Dent', 'role': 'intergalactic traveler', 'plan': 'premium'},
    123: {'name': 'Jane Doe', 'role': 'engineer', 'plan': 'enterprise'},
    999: {'name': 'Guest User', 'role': 'visitor', 'plan': 'free'},
}


class ContextInput(BaseModel):
    """Input for context demo flows."""

    user_id: int = Field(default=42, description='User ID to look up (try 42, 123, or 999)')


@ai.tool()
def get_user_info() -> str:
    """Look up the current user from context.

    This tool takes no explicit input -- it reads the user ID from the
    execution context that was passed to ai.generate(context=...).

    Returns:
        A description of the user.
    """
    context = Genkit.current_context() or {}
    _, record = _user_from_context(context)
    return f'{record["name"]} ({record["role"]}, {record["plan"]} plan)'


@ai.tool()
def get_user_via_static() -> str:
    """Look up the current user using Genkit.current_context().

    Demonstrates the static method approach -- useful when context is needed
    deep in a call stack where ctx isn't easily threaded through.

    Returns:
        A description of the user from the static context accessor.
    """
    context = Genkit.current_context()
    if context is None:
        return 'No context available (not running inside an action).'
    user = context.get('user', {})
    user_id = user.get('id')
    record = MOCK_USER_DB.get(user_id, {'name': 'Unknown', 'role': 'unknown', 'plan': 'none'})
    return f'[via current_context()] {record["name"]} ({record["role"]}, {record["plan"]} plan)'


@ai.tool()
def get_user_permissions() -> str:
    """Return permissions based on the user's plan from context.

    Used in the propagation chain demo to verify context survives
    through nested generate calls.

    Returns:
        Permission level description.
    """
    context = Genkit.current_context() or {}
    _, record = _user_from_context(context)
    permissions = {
        'free': 'read-only access',
        'premium': 'read-write access with priority support',
        'enterprise': 'full admin access with SLA guarantees',
        'none': 'no access',
    }
    return f'{record["name"]} has {permissions.get(record["plan"], "unknown")} ({record["plan"]} plan)'


@ai.flow()
async def context_in_generate(input: ContextInput) -> str:
    """Pass context to ai.generate() and let a tool read it.

    This is the simplest context pattern: the caller provides context
    as a dictionary, and tools receive it via ``ctx.context``.

    Args:
        input: Input with user ID.

    Returns:
        Model response incorporating user-specific tool output.
    """
    response = await ai.generate(
        prompt='Look up the current user and describe who they are.',
        tools=['get_user_info'],
        context={'user': {'id': input.user_id}},
    )
    return response.text


@ai.flow()
async def context_in_flow(input: ContextInput, ctx: ActionRunContext) -> str:
    """Access context directly inside a flow.

    When a flow is invoked with context (e.g., from the DevUI or another
    flow), the ``ActionRunContext`` parameter provides access to it.

    Args:
        input: Input with user ID.
        ctx: Execution context provided by Genkit.

    Returns:
        Description of what context the flow sees.
    """
    flow_context = ctx.context
    await logger.ainfo('Flow context received', context=flow_context)

    response = await ai.generate(
        prompt=(f'The flow received this context: {flow_context}. Also look up the user info using the tool.'),
        tools=['get_user_info'],
        context={'user': {'id': input.user_id}},
    )
    return response.text


@ai.flow()
async def context_current_context(input: ContextInput) -> str:
    """Demonstrate Genkit.current_context() static method.

    The tool in this flow uses ``Genkit.current_context()`` instead of
    ``ctx.context`` to read the execution context. This is useful when
    context is needed deep in a call stack where the ToolRunContext
    object isn't directly available.

    Args:
        input: Input with user ID.

    Returns:
        Model response using the static context accessor.
    """
    response = await ai.generate(
        prompt='Look up the current user using the static context method and describe them.',
        tools=['get_user_via_static'],
        context={'user': {'id': input.user_id}},
    )
    return response.text


@ai.flow()
async def context_propagation_chain(input: ContextInput) -> str:
    """Verify context propagates through nested generate/tool chains.

    This flow calls ai.generate() with context, which triggers a tool.
    That tool's response is then fed into a second ai.generate() call
    (without explicitly passing context again) to verify that context
    is automatically inherited through the ContextVar mechanism.

    Args:
        input: Input with user ID.

    Returns:
        Combined response showing context survived multiple levels.
    """
    first_response = await ai.generate(
        prompt='Look up the current user info.',
        tools=['get_user_info'],
        context={'user': {'id': input.user_id}},
    )

    second_response = await ai.generate(
        prompt=(
            f'The user was identified as: {first_response.text}. '
            'Now check their permissions using the permissions tool.'
        ),
        tools=['get_user_permissions'],
        context={'user': {'id': input.user_id}},
    )

    return f'User info: {first_response.text}\nPermissions: {second_response.text}'


async def main() -> None:
    """Main function -- keep alive for Dev UI."""
    await logger.ainfo('Context demo started. Open http://localhost:4000 to test flows.')
    while True:
        await asyncio.sleep(3600)


if __name__ == '__main__':
    ai.run_main(main())
