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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Keyword-only constructor contracts for Action, Prompt, and Context classes.

pyright, pyrefly and ty must accept the documented positional prefixes with
keyword-only options, and reject extra positional arguments past ``*``. Each
checker fails on an unused ignore comment here, so dropping ``*`` breaks CI.
"""

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable

import pytest

from genkit import FlowContext, Genkit, Prompt, ToolContext
from genkit._core._action import ActionRunContext, BidiAction
from genkit.exp.agent import AgentContext
from genkit.middleware import MiddlewareContext
from genkit.model import ModelContext
from genkit.plugin_api import Action, ActionKind


async def _check_allergens(dish: str) -> str:
    return f'{dish}: nut-free'


async def _seat_table(
    init: str | None,
    turns: AsyncIterator[str],
    send_chunk: Callable[[str], None],
) -> str:
    async for turn in turns:
        send_chunk(turn)
    return init or 'walk-in'


def test_keyword_only_constructors_accept_documented_shapes() -> None:
    """Positional identity args plus keyword options type-check and construct cleanly."""
    ai = Genkit()
    abort = asyncio.Event()

    action = Action(
        ActionKind.CUSTOM,
        'check_allergens',
        _check_allergens,
        description='Check dish allergens.',
    )
    assert action.name == 'check_allergens'
    assert action.description == 'Check dish allergens.'

    bidi = BidiAction(
        ActionKind.AGENT,
        'table_host',
        _seat_table,
        description='Seat guests.',
    )
    assert bidi.name == 'table_host'

    prompt = Prompt(
        ai,
        name='menu_summary',
        variant='dinner',
        prompt='Summarize {{dish}}.',
    )
    assert prompt.ref['name'] == 'menu_summary.dinner'

    run_ctx = ActionRunContext(context={'table': 4}, abort_signal=abort)
    flow_ctx = FlowContext(context={'table': 4})
    model_ctx: ModelContext = ModelContext(context={'table': 4})
    agent_ctx: AgentContext = AgentContext(context={'table': 4})
    tool_ctx = ToolContext(
        flow_ctx,
        resumed_metadata={'approved': True},
        original_input={'dish': 'salmon'},
    )
    mw_ctx = MiddlewareContext(
        ai,
        abort_signal=abort,
        context={'table': 4},
        telemetry_labels={'service': 'dining'},
    )

    assert run_ctx.context == {'table': 4}
    assert model_ctx.context == {'table': 4}
    assert agent_ctx.context == {'table': 4}
    assert tool_ctx.is_resumed() is True
    assert mw_ctx.telemetry_labels == {'service': 'dining'}


def test_positional_arguments_past_star_fail_type_checkers_and_runtime() -> None:
    """Passing optional configuration positionally is both a static type error and a runtime TypeError."""
    ai = Genkit()
    abort = asyncio.Event()
    flow_ctx = FlowContext()

    with pytest.raises(TypeError, match='positional argument'):
        Action(ActionKind.CUSTOM, 'check_allergens', _check_allergens, _check_allergens)  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        BidiAction(ActionKind.AGENT, 'table_host', _seat_table, _seat_table)  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        Prompt(ai, 'dinner')  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        ActionRunContext({'table': 4})  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        FlowContext({'table': 4})  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        ModelContext({'table': 4})  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        AgentContext({'table': 4})  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        ToolContext(flow_ctx, {'approved': True})  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    with pytest.raises(TypeError, match='positional argument'):
        MiddlewareContext(ai, abort)  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]


@pytest.mark.parametrize(
    ('cls', 'positional_names'),
    [
        (Action, ['self', 'kind', 'name', 'fn']),
        (BidiAction, ['self', 'kind', 'name', 'bidi_fn']),
        (Prompt, ['self', 'ai']),
        (ActionRunContext, ['self']),
        (ToolContext, ['self', 'ctx']),
        (MiddlewareContext, ['self', 'ai']),
    ],
)
def test_every_non_identity_init_param_is_keyword_only(
    cls: type,
    positional_names: list[str],
) -> None:
    """Pin the exact positional prefix so new parameters cannot slip in before ``*``."""
    params = list(inspect.signature(cls.__init__).parameters.values())
    pos = [p.name for p in params if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    kw_only = [p for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY]

    assert pos == positional_names
    assert len(pos) + len(kw_only) == len(params)
    assert len(kw_only) >= 1
