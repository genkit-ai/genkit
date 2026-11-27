# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A flow takes one input; the context arrives only on a FlowContext-annotated parameter."""

import sys
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Flow, FlowContext, Genkit


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
async def test_flow_with_input_then_flow_context_gets_both() -> None:
    """`(input, ctx: FlowContext)` receives both."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str, ctx: FlowContext) -> str:
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


@pytest.mark.asyncio
async def test_flow_with_flow_context_first_then_input_gets_both() -> None:
    """`(ctx: FlowContext, input)` receives both, and `flow.input_schema` is the input's."""
    ai = Genkit()

    @ai.flow()
    async def greet(ctx: FlowContext, name: str) -> str:
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


@pytest.mark.asyncio
async def test_flow_with_only_flow_context_runs_with_context_and_no_input() -> None:
    """`(ctx: FlowContext)` gets the context instead of the input."""
    ai = Genkit()

    @ai.flow()
    async def whoami(ctx: FlowContext) -> str:
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
    assert "annotate 'b' as FlowContext" in str(exc.value)


def test_flow_with_unannotated_ctx_parameter_raises_type_error_at_definition() -> None:
    """`(input, ctx)` with no annotation raises TypeError telling them to annotate FlowContext."""
    ai = Genkit()

    async def greet(name: str, ctx) -> str:  # noqa: ANN001
        return name

    with pytest.raises(TypeError, match="'ctx' is a second parameter") as exc:
        ai.flow()(greet)
    assert "annotate 'ctx' as FlowContext" in str(exc.value)


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
async def test_flow_with_postponed_annotations_finds_flow_context() -> None:
    """The string annotation `'FlowContext'` under `from __future__ import annotations` still marks the context."""
    ai = Genkit()

    # Postponed annotations are stored as these strings.
    @ai.flow()
    async def greet(ctx: 'FlowContext', name: 'str') -> 'str':
        return f'hello {name} as {ctx.context["user"]}'

    result = await greet.run('ada', context={'user': 'u1'})

    assert greet.input_schema == {'type': 'string'}
    assert result.response == 'hello ada as u1'


class Greeting(BaseModel):
    name: str


# A user module that imports the context type only for type checkers, so
# 'FlowContext' can't be resolved at runtime but 'Greeting' can.
_TYPE_CHECKING_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit import FlowContext


async def greet(input: Greeting, ctx: FlowContext) -> str:
    return f'hello {input.name} as {ctx.context["user"]}'
"""


@pytest.mark.asyncio
async def test_flow_with_type_checking_only_context_still_resolves_module_level_input() -> None:
    """A `TYPE_CHECKING`-only `FlowContext` doesn't stop the module-level `Greeting` from resolving."""
    ai = Genkit()
    module_globals: dict[str, Any] = {'Greeting': Greeting}
    exec(_TYPE_CHECKING_CONTEXT_MODULE, module_globals)  # noqa: S102 - builds a module with postponed annotations

    greet_fn: Callable[[Greeting, FlowContext], Awaitable[str]] = module_globals['greet']
    greet = ai.flow()(greet_fn)
    result = await greet.run(Greeting(name='ada'), context={'user': 'u1'})

    assert greet.input_schema == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}},
        'required': ['name'],
        'title': 'Greeting',
        'type': 'object',
    }
    assert result.response == 'hello ada as u1'


@pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from Python 3.14')
@pytest.mark.asyncio
async def test_flow_with_type_checking_only_context_runs_without_future_import() -> None:
    """On 3.14, a `TYPE_CHECKING`-only `FlowContext` without the `__future__` import still runs."""
    ai = Genkit()
    module_globals: dict[str, Any] = {'Greeting': Greeting}
    source = _TYPE_CHECKING_CONTEXT_MODULE.replace('from __future__ import annotations\n', '')
    exec(source, module_globals)  # noqa: S102 - builds a module whose annotations are evaluated lazily

    greet_fn: Callable[[Greeting, FlowContext], Awaitable[str]] = module_globals['greet']
    greet = ai.flow()(greet_fn)
    result = await greet.run(Greeting(name='ada'), context={'user': 'u1'})

    assert greet.input_schema == {
        'properties': {'name': {'title': 'Name', 'type': 'string'}},
        'required': ['name'],
        'title': 'Greeting',
        'type': 'object',
    }
    assert result.response == 'hello ada as u1'


@pytest.mark.asyncio
async def test_flow_with_parameterized_flow_context_streams_typed_chunks() -> None:
    """`ctx: FlowContext[str]` marks the context and its `send_chunk` streams to the caller."""
    ai = Genkit()

    @ai.flow()
    async def plate(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('searing')
        return f'{dish} for {ctx.context["table"]}'

    chunks: list[str] = []
    result = await plate.run('salmon', on_chunk=chunks.append, context={'table': 4})

    assert plate.input_schema == {'type': 'string'}
    assert chunks == ['searing']
    assert result.response == 'salmon for 4'


_TYPE_CHECKING_PARAMETERIZED_CONTEXT_MODULE = """
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from genkit import FlowContext


async def plate(dish: str, ctx: FlowContext[str]) -> str:
    ctx.send_chunk('searing')
    return f'{dish} for {ctx.context["table"]}'
"""


@pytest.mark.asyncio
async def test_flow_with_type_checking_only_parameterized_flow_context_gets_context() -> None:
    """A `TYPE_CHECKING`-only `FlowContext[str]` stays the string `'FlowContext[str]'` and still marks the context."""
    ai = Genkit()
    module_globals: dict[str, Any] = {}
    exec(_TYPE_CHECKING_PARAMETERIZED_CONTEXT_MODULE, module_globals)  # noqa: S102 - builds a module with postponed annotations

    plate_fn: Callable[[str, FlowContext[str]], Awaitable[str]] = module_globals['plate']
    plate = ai.flow()(plate_fn)
    chunks: list[str] = []
    result = await plate.run('salmon', on_chunk=chunks.append, context={'table': 4})

    assert plate.input_schema == {'type': 'string'}
    assert chunks == ['searing']
    assert result.response == 'salmon for 4'


@pytest.mark.asyncio
async def test_bare_flow_decorator_without_parentheses_registers_and_streams() -> None:
    """Bare `@ai.flow` (without parentheses) registers the flow by function name and infers `FlowContext[ChunkT]`."""
    ai = Genkit()

    @ai.flow
    async def greet(name: str) -> str:
        return f'hello {name}'

    @ai.flow
    async def plate(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('searing')
        return f'{dish} for {ctx.context["table"]}'

    assert isinstance(greet, Flow)
    assert isinstance(plate, Flow)
    assert_type(greet, Flow[str, str])
    assert_type(plate, Flow[str, str, str])
    assert greet.name == 'greet'
    assert await greet('ada') == 'hello ada'

    chunks: list[str] = []
    result = await plate.run('salmon', on_chunk=chunks.append, context={'table': 4})
    assert plate.name == 'plate'
    assert chunks == ['searing']
    assert result.response == 'salmon for 4'


@pytest.mark.asyncio
async def test_standalone_flow_class_runs_and_streams_without_registry() -> None:
    """`Flow(fn)` constructs a standalone typed flow that runs, streams, and enforces strict I/O without a registry."""

    async def plate(dish: str, ctx: FlowContext[str]) -> str:
        ctx.send_chunk('searing')
        ctx.send_chunk('plating')
        return f'{dish} for {ctx.context["table"]}'

    standalone = Flow(plate, name='plate_dish', description='Plate a dish.')
    assert isinstance(standalone, Flow)
    assert_type(standalone, Flow[str, str, str])
    assert standalone.name == 'plate_dish'
    assert standalone.description == 'Plate a dish.'
    assert standalone.input_schema == {'type': 'string'}
    assert standalone.output_schema == {'type': 'string'}
    assert await standalone('halibut', context={'table': 7}) == 'halibut for 7'

    stream = standalone.stream('salmon', context={'table': 4})
    got = [chunk async for chunk in stream.stream]
    assert got == ['searing', 'plating']
    assert await stream.response == 'salmon for 4'

    async def reversed_flow(ctx: FlowContext[int], dish: str) -> str:
        ctx.send_chunk(1)
        return dish.upper()

    rev = Flow(reversed_flow)
    assert_type(rev, Flow[str, str, int])
    assert await rev('risotto') == 'RISOTTO'

    async def simple_flow(dish: str) -> int:
        return len(dish)

    simple = Flow(simple_flow)
    assert_type(simple, Flow[str, int])
    assert await simple('taco') == 4

    async def untyped(name) -> str:  # noqa: ANN001
        return f'hello {name}'

    with pytest.raises(TypeError, match="flow 'untyped' input 'name' has no type annotation"):
        Flow(untyped)

    def sync_fn(x: str) -> str:
        return x

    with pytest.raises(TypeError, match='Flow must be async'):
        Flow(sync_fn)  # type: ignore[arg-type]


class _Order(BaseModel):
    dish: str


@pytest.mark.asyncio
async def test_flow_input_type_inference_across_shapes() -> None:
    """Each handler shape types the flow's input as written, in pyright, pyrefly and ty.

    `(x: Any)` and `(x: object)` also accept a FlowContext, so they must not
    match the ctx-only overload and come out as Flow[None, ...].
    """
    ai = Genkit()

    async def echo_any(x: Any) -> str:  # noqa: ANN401
        return str(x)

    async def echo_object(x: object) -> str:
        return str(x)

    async def ping(ctx: FlowContext) -> str:
        return 'pong'

    async def plate(ctx: FlowContext[str]) -> str:
        ctx.send_chunk('searing')
        return 'plated'

    async def take_order(x: _Order) -> str:
        return x.dish

    async def stream_order(x: _Order, ctx: FlowContext[str]) -> str:
        ctx.send_chunk(x.dish)
        return x.dish

    async def stream_any(ctx: FlowContext[str], x: Any) -> str:  # noqa: ANN401
        return str(x)

    async def specials() -> str:
        return 'risotto'

    async def with_default(dish: str = 'soup') -> str:
        return dish

    # 1. Decorator, bare and called.
    # ty types `(x: Any)` as Unknown: it matches overloads with different
    # returns, and the typing spec's overload rule makes that ambiguous. No
    # arrangement fixes it without breaking `(ctx: FlowContext)`.
    assert_type(ai.flow(echo_any), Flow[Any, str])  # ty: ignore[type-assertion-failure]
    assert_type(ai.flow()(echo_any), Flow[Any, str])  # ty: ignore[type-assertion-failure]
    assert_type(ai.flow(echo_object), Flow[Any, str])
    assert_type(ai.flow()(echo_object), Flow[Any, str])
    assert_type(ai.flow(ping), Flow[None, str])
    assert_type(ai.flow()(plate), Flow[None, str, str])
    assert_type(ai.flow(take_order), Flow[_Order, str])
    assert_type(ai.flow(stream_order), Flow[_Order, str, str])
    assert_type(ai.flow(stream_any), Flow[Any, str, str])  # ty: ignore[type-assertion-failure]
    assert_type(ai.flow(specials), Flow[None, str])
    assert_type(ai.flow(with_default), Flow[str, str])

    # 2. Standalone Flow(fn). An __init__ overload can't pin InputT to None,
    # so ctx-only and no-arg flows leave it at its Any default.
    assert_type(Flow(echo_any), Flow[Any, str])
    assert_type(Flow(echo_object), Flow[Any, str])
    assert_type(Flow(ping), Flow[Any, str])
    assert_type(Flow(plate), Flow[Any, str, str])
    assert_type(Flow(take_order), Flow[_Order, str])
    assert_type(Flow(stream_order), Flow[_Order, str, str])
    assert_type(Flow(stream_any), Flow[Any, str, str])
    assert_type(Flow(specials), Flow[Any, str])
    assert_type(Flow(with_default), Flow[str, str])

    # 3. An Any/object flow takes any input at the call site.
    echo = Flow(echo_object)
    assert await echo(5) == '5'
    assert await Flow(echo_any)({'dish': 'soup'}) == "{'dish': 'soup'}"
    assert await Flow(ping)() == 'pong'
