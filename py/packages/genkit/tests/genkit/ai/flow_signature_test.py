# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A flow takes one input; the context arrives only on an ActionRunContext-annotated parameter."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit


@pytest.mark.asyncio
async def test_flow_with_one_input_returns_result() -> None:
    """A one-input flow runs and returns, as before."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str) -> str:
        return f'hello {name}'

    assert greet.input_schema == {'type': 'string'}
    assert await greet('ada') == 'hello ada'


@pytest.mark.asyncio
async def test_flow_with_input_then_action_run_context_gets_both() -> None:
    """`(input, ctx: ActionRunContext)` receives both."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str, ctx: ActionRunContext) -> str:
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


@pytest.mark.asyncio
async def test_flow_with_action_run_context_first_then_input_gets_both() -> None:
    """`(ctx: ActionRunContext, input)` receives both, and `flow.input_schema` is the input's."""
    ai = Genkit()

    @ai.flow()
    async def greet(ctx: ActionRunContext, name: str) -> str:
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


@pytest.mark.asyncio
async def test_flow_with_only_action_run_context_runs_with_context_and_no_input() -> None:
    """`(ctx: ActionRunContext)` gets the context instead of the input."""
    ai = Genkit()

    @ai.flow()
    async def whoami(ctx: ActionRunContext) -> str:
        return f'you are {ctx.context["user"]}'

    result = await whoami.run(context={'user': 'u1'})

    assert whoami.input_schema == {}
    assert result.response == 'you are u1'


def test_flow_with_two_plain_parameters_raises_type_error_at_definition() -> None:
    """`(a: str, b: str)` raises TypeError naming `b`."""
    ai = Genkit()

    async def pair(a: str, b: str) -> str:
        return a + b

    with pytest.raises(TypeError, match="flow 'pair' takes one input, but 'b' is a second parameter") as exc:
        ai.flow()(pair)
    assert "annotate 'b' as ActionRunContext" in str(exc.value)


def test_flow_with_unannotated_ctx_parameter_raises_type_error_at_definition() -> None:
    """`(input, ctx)` with no annotation raises TypeError telling them to annotate ActionRunContext."""
    ai = Genkit()

    async def greet(name: str, ctx) -> str:  # noqa: ANN001
        return name

    with pytest.raises(TypeError, match="'ctx' is a second parameter") as exc:
        ai.flow()(greet)
    assert "annotate 'ctx' as ActionRunContext" in str(exc.value)


def test_flow_with_unannotated_input_raises_type_error_at_definition() -> None:
    """`greet(name)` with no annotation raises TypeError naming `name` and the `Any` option."""
    ai = Genkit()

    async def greet(name) -> str:  # noqa: ANN001
        return f'hello {name}'

    with pytest.raises(TypeError, match="flow 'greet' input 'name' has no type annotation") as exc:
        ai.flow()(greet)
    assert 'or use Any to accept anything' in str(exc.value)


def test_flow_with_plain_class_input_raises_type_error_naming_input() -> None:
    """`read(t: Thermometer)` raises TypeError naming the flow, the input, and the kinds of types it can be."""
    ai = Genkit()

    class Thermometer:
        pass

    async def read(t: Thermometer) -> str:
        return ''

    with pytest.raises(TypeError, match="flow 'read' input 't' has type Thermometer, which has no JSON schema") as exc:
        ai.flow()(read)
    assert 'Use a Pydantic model, dataclass, TypedDict, or a basic type' in str(exc.value)


def test_flow_with_input_model_defined_in_function_raises_type_error_naming_it() -> None:
    """A flow whose input model is defined in the same function raises TypeError saying to move it to module level."""
    ai = Genkit()

    class StepInput(BaseModel):
        step: int

    # Postponed annotations are stored as this string, which can't see StepInput.
    async def run_step(input: 'StepInput') -> int:
        return input.step

    with pytest.raises(
        TypeError, match="flow 'run_step' input 'input' has type 'StepInput', which can't be found"
    ) as exc:
        ai.flow()(run_step)
    assert 'Define or import it at module level' in str(exc.value)


@pytest.mark.asyncio
async def test_flow_with_postponed_annotations_finds_action_run_context() -> None:
    """The string annotation `'ActionRunContext'` under `from __future__ import annotations` still marks the context."""
    ai = Genkit()

    # Postponed annotations are stored as these strings.
    @ai.flow()
    async def greet(ctx: 'ActionRunContext', name: 'str') -> 'str':
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


class Greeting(BaseModel):
    name: str


# A user module that imports the context type only for type checkers, so
# 'ActionRunContext' can't be resolved at runtime but 'Greeting' can.
_TYPE_CHECKING_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit import ActionRunContext


async def greet(input: Greeting, ctx: ActionRunContext) -> str:
    return f'hello {input.name} as {ctx.context["user"]}'
"""


@pytest.mark.asyncio
async def test_flow_with_type_checking_only_context_still_resolves_module_level_input() -> None:
    """A `TYPE_CHECKING`-only `ActionRunContext` doesn't stop the module-level `Greeting` from resolving."""
    ai = Genkit()
    module_globals: dict[str, Any] = {'Greeting': Greeting}
    exec(_TYPE_CHECKING_CONTEXT_MODULE, module_globals)  # noqa: S102 - builds a module with postponed annotations

    greet_fn: Callable[[Greeting, ActionRunContext], Awaitable[str]] = module_globals['greet']
    greet = ai.flow()(greet_fn)
    result = await greet.run(Greeting(name='ada'), context={'user': 'u1'})

    assert greet.input_schema == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}},
        'required': ['name'],
        'title': 'Greeting',
        'type': 'object',
    }
    assert result.response == 'hello ada as u1'
