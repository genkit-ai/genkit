# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Every action takes at most one input; the context goes to the parameter annotated ActionRunContext."""

import sys
from collections.abc import Awaitable, Callable
from typing import Optional, cast

import pytest

from genkit import ActionRunContext, Genkit, Message, ModelResponse, Operation, Part
from genkit._core._action import Action, ActionKind
from genkit._core._background import CheckModelOpFn, StartModelOpFn
from genkit._core._typing import Role
from genkit.model import ModelRequest

_REPLY = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))


@pytest.mark.asyncio
async def test_model_with_context_first_gets_request_and_context() -> None:
    """A model fn `(ctx, request)` gets each by annotation, not by position."""
    seen: list[object] = []

    async def chef(ctx: ActionRunContext, request: ModelRequest) -> ModelResponse:
        seen.extend([type(request), ctx.context])
        return _REPLY

    action = Action(ActionKind.MODEL, 'chef', chef)
    result = await action.run(ModelRequest(messages=[]), context={'table': 4})

    assert result.response.text == 'Smoked Salmon Tartine'
    assert seen == [ModelRequest, {'table': 4}]


@pytest.mark.asyncio
async def test_action_with_optional_context_union_gets_context() -> None:
    """`ctx: ActionRunContext | None` and `Optional[ActionRunContext]` both mark the context."""

    async def by_pipe(order: str, ctx: ActionRunContext | None = None) -> str:
        return f'{order} for table {ctx.context["table"] if ctx else "?"}'

    async def by_optional(ctx: Optional[ActionRunContext], order: str) -> str:  # noqa: UP045 - the old spelling is the case under test
        return f'{order} for table {ctx.context["table"] if ctx else "?"}'

    for fn in (by_pipe, by_optional):
        action = Action(ActionKind.CUSTOM, fn.__name__, fn)
        result = await action.run('soup', context={'table': 4})
        assert result.response == 'soup for table 4'


@pytest.mark.asyncio
async def test_action_with_postponed_context_annotations_gets_context() -> None:
    """Postponed annotations mark the context, parameterized or not."""
    seen: list[object] = []

    # Postponed annotations are stored as these strings.
    async def plain(order: str, ctx: 'ActionRunContext') -> None:
        seen.append(type(ctx))

    async def parameterized(ctx: 'ActionRunContext[str]', order: str) -> None:
        seen.append(type(ctx))

    for fn in (plain, parameterized):
        await Action(ActionKind.CUSTOM, fn.__name__, fn).run('soup')

    assert seen == [ActionRunContext, ActionRunContext]


# A user module that imports its context type only for type checkers, so the
# annotation stays the string 'ToolRunContext | None' at runtime.
_TYPE_CHECKING_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit import ToolRunContext


async def plate(order: str, ctx: ToolRunContext | None) -> str:
    return f'{order} for table {ctx.context["table"]}'
"""


@pytest.mark.asyncio
async def test_action_with_unresolvable_context_subclass_union_gets_context() -> None:
    """An unresolvable `'ToolRunContext | None'` still marks the context, by class name."""
    module_globals: dict[str, object] = {}
    exec(_TYPE_CHECKING_CONTEXT_MODULE, module_globals)  # noqa: S102 - builds a module with postponed annotations
    plate = cast(Callable[[str, ActionRunContext], Awaitable[str]], module_globals['plate'])

    result = await Action(ActionKind.CUSTOM, 'plate', plate).run('soup', context={'table': 4})

    assert result.response == 'soup for table 4'


@pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from Python 3.14')
@pytest.mark.asyncio
async def test_action_with_type_checking_only_context_runs_without_future_import() -> None:
    """On 3.14, a `TYPE_CHECKING`-only `ToolRunContext` without the `__future__` import still runs."""
    module_globals: dict[str, object] = {}
    source = _TYPE_CHECKING_CONTEXT_MODULE.replace('from __future__ import annotations\n', '')
    exec(source, module_globals)  # noqa: S102 - builds a module whose annotations are evaluated lazily
    plate = cast(Callable[[str, ActionRunContext], Awaitable[str]], module_globals['plate'])

    result = await Action(ActionKind.CUSTOM, 'plate', plate).run('soup', context={'table': 4})

    assert result.response == 'soup for table 4'


@pytest.mark.asyncio
async def test_action_with_defaulted_input_after_context_uses_default() -> None:
    """Running without an input leaves a defaulted input out, even when it comes after the context."""

    async def special(ctx: ActionRunContext, dish: str = 'risotto') -> str:
        return dish

    assert (await Action(ActionKind.CUSTOM, 'special', special).run()).response == 'risotto'


@pytest.mark.asyncio
async def test_custom_action_allows_unannotated_input() -> None:
    """Only tools and flows require an input annotation; other actions accept anything."""

    async def echo(order, ctx: ActionRunContext) -> object:  # noqa: ANN001
        return order

    action = Action(ActionKind.CUSTOM, 'echo', echo)

    assert action.input_schema == {}
    assert (await action.run({'dish': 'soup'})).response == {'dish': 'soup'}


def test_action_with_unannotated_second_parameter_raises_type_error() -> None:
    """`(request, ctx)` with no ctx annotation raises at definition, saying to annotate ActionRunContext."""

    async def chef(request: ModelRequest, ctx) -> ModelResponse:  # noqa: ANN001
        return _REPLY

    with pytest.raises(TypeError, match="model 'chef' takes one input, but 'ctx' is a second parameter") as exc:
        Action(ActionKind.MODEL, 'chef', chef)
    assert "annotate 'ctx' as ActionRunContext" in str(exc.value)


def test_action_with_positional_only_parameter_raises_type_error() -> None:
    """A positional-only parameter can't be passed by name, so it raises at definition."""

    async def chef(request: ModelRequest, /, ctx: ActionRunContext) -> ModelResponse:
        return _REPLY

    with pytest.raises(TypeError, match="model 'chef' parameter 'request' is positional-only"):
        Action(ActionKind.MODEL, 'chef', chef)


@pytest.mark.asyncio
async def test_background_model_with_context_first_start_gets_both() -> None:
    """`start(ctx, request)` gets each by name, and the handle has the start action key."""
    ai = Genkit()
    seen: list[object] = []

    async def start(ctx: ActionRunContext, request: ModelRequest) -> Operation:
        seen.extend([type(request), ctx.context])
        return Operation(id='render-1', done=False)

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        return op

    # StartModelOpFn types (request, ctx) positionally; the runtime takes either order.
    action = ai.define_background_model(name='menu-video', start=cast(StartModelOpFn, start), check=check)
    result = await action.start_action.run(ModelRequest(messages=[]), context={'table': 4})

    assert seen == [ModelRequest, {'table': 4}]
    assert result.response.id == 'render-1'
    assert result.response.action == '/background-model/menu-video'


@pytest.mark.asyncio
async def test_background_model_with_context_first_check_gets_operation() -> None:
    """`check(ctx, op)` gets each by name, same as start."""
    ai = Genkit()
    seen: list[object] = []

    async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id='render-1', done=False)

    async def check(ctx: ActionRunContext, op: Operation) -> Operation:
        seen.extend([op.id, ctx.context])
        return op

    action = ai.define_background_model(name='menu-video', start=start, check=cast(CheckModelOpFn, check))
    started = await action.start_action.run(ModelRequest(messages=[]), context={'table': 4})
    result = await action.check_action.run(started.response, context={'table': 4})

    assert seen == ['render-1', {'table': 4}]
    assert result.response.action == '/background-model/menu-video'


def test_background_model_actions_lists_start_check_and_cancel() -> None:
    """`actions` is every action a background model registers; cancel only when given."""
    ai = Genkit()

    async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id='render-1')

    async def check(op: Operation, ctx: ActionRunContext) -> Operation:
        return op

    video = ai.define_background_model(name='menu-video', start=start, check=check, cancel=check)
    photo = ai.define_background_model(name='menu-photo', start=start, check=check)

    assert [a.name for a in video.actions] == ['menu-video', 'menu-video/check', 'menu-video/cancel']
    assert [a.name for a in photo.actions] == ['menu-photo', 'menu-photo/check']


def test_action_params_names_input_and_context() -> None:
    """`Action.params` exposes which parameter gets the input and which gets the context."""

    async def chef(ctx: ActionRunContext, request: ModelRequest) -> ModelResponse:
        return _REPLY

    params = Action(ActionKind.MODEL, 'chef', chef).params

    assert params.input is not None and params.input.name == 'request'
    assert params.context is not None and params.context.name == 'ctx'
    assert not params.input_optional
