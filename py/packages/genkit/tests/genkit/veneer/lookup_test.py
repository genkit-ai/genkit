#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What app code and middleware get back from lookup_model, lookup_tool, lookup_value, and define_value."""

import re
from collections.abc import Awaitable, Callable

import pytest

from genkit import FinishReason, Genkit, Message, ModelResponse, Operation, Part, Role, Tool
from genkit._core._typing import MiddlewareRef
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import GenerateActionOptions, Model, ModelContext, ModelRequest
from genkit.plugin_api import Action, ActionKind, Plugin

NextModel = Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]]


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _define_answering_model(ai: Genkit, name: str, text: str) -> None:
    async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return _answer(text)

    ai.define_model(name=name, fn=fn)


class LazyPlugin(Plugin):
    """Serves `lazy/fast` only through resolve, never from init."""

    name = 'lazy'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.MODEL or name != 'fast':
            return None

        async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
            return _answer('from lazy fast')

        return Model(name, fn)

    async def list_actions(self) -> list[Action]:
        return []


@pytest.mark.asyncio
async def test_lookup_model_returns_registered_model_action() -> None:
    """`await ai.lookup_model('test/echo')` returns the model action, and running it gets that model's answer."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'echoed')

    action = await ai.lookup_model('test/echo')

    assert isinstance(action, Model)
    assert action.name == 'test/echo'
    result = await action.run(ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])]))
    assert isinstance(result.response, ModelResponse)
    assert result.response.text == 'echoed'


@pytest.mark.asyncio
async def test_lookup_model_unknown_name_returns_none() -> None:
    """`await ai.lookup_model('nope')` returns None instead of raising."""
    ai = Genkit()

    assert await ai.lookup_model('nope') is None


@pytest.mark.asyncio
async def test_lookup_model_resolves_plugin_model_not_yet_loaded() -> None:
    """A plugin model only reachable through the plugin's resolve comes back from `lookup_model` before any generate."""
    ai = Genkit(plugins=[LazyPlugin()])

    action = await ai.lookup_model('lazy/fast')

    assert action is not None
    assert action.name == 'lazy/fast'
    result = await action.run(ModelRequest(messages=[]))
    assert result.response.text == 'from lazy fast'


@pytest.mark.asyncio
async def test_lookup_model_does_not_return_background_models() -> None:
    """A background model runs through `generate_operation`, so `lookup_model` returns None for it."""
    ai = Genkit()

    async def start(request: ModelRequest, ctx: ModelContext) -> Operation:
        return Operation(id='op-1')

    async def check(operation: Operation, ctx: ModelContext) -> Operation:
        return operation

    ai.define_background_model('veo', start=start, check=check)

    assert await ai.lookup_model('veo') is None


@pytest.mark.asyncio
async def test_lookup_tool_returns_a_callable_tool_with_the_same_definition() -> None:
    """`ai.lookup_tool('menu_price')` gives back a Tool that runs and describes itself like the decorated one."""
    ai = Genkit()

    @ai.tool()
    async def menu_price(dish: str) -> float:
        """Price of a dish in dollars."""
        return 14.5

    found = await ai.lookup_tool('menu_price')

    assert isinstance(found, Tool)
    assert found.definition() == menu_price.definition()
    assert (await found('ramen')).output == 14.5


@pytest.mark.asyncio
async def test_lookup_tool_unknown_name_returns_none() -> None:
    """A name nothing answers to returns None."""
    ai = Genkit()

    assert await ai.lookup_tool('missing') is None


@pytest.mark.asyncio
async def test_lookup_tool_finds_interrupts() -> None:
    """An interrupt is a tool, so `lookup_tool` finds it."""
    ai = Genkit()
    ai.define_interrupt('confirm_order', description='Ask the diner to confirm.')

    found = await ai.lookup_tool('confirm_order')

    assert found is not None
    assert found.name == 'confirm_order'


def test_define_value_then_lookup_value_returns_it() -> None:
    """A value stored with `ai.define_value(...)` comes back from `ai.lookup_value(...)` for the same kind and name."""
    ai = Genkit()
    catalog = {'id': 'banner', 'components': []}

    ai.define_value(kind='catalog', name='banner', value=catalog)

    assert ai.lookup_value(kind='catalog', name='banner') is catalog


def test_lookup_value_unknown_returns_none() -> None:
    """`ai.lookup_value(kind=..., name=...)` for nothing defined returns None."""
    ai = Genkit()
    ai.define_value(kind='catalog', name='banner', value={'id': 'banner'})

    assert ai.lookup_value(kind='catalog', name='other') is None
    assert ai.lookup_value(kind='other-kind', name='banner') is None


def test_define_value_same_name_twice_raises() -> None:
    """Defining the same kind and name twice raises ValueError and the first value stays."""
    ai = Genkit()
    ai.define_value(kind='catalog', name='banner', value='first')

    with pytest.raises(ValueError, match='already registered'):
        ai.define_value(kind='catalog', name='banner', value='second')

    assert ai.lookup_value(kind='catalog', name='banner') == 'first'


def test_define_value_requires_keywords() -> None:
    """`ai.define_value('k', 'n', v)` raises TypeError, so swapped kind and name can't store under the wrong key."""
    ai = Genkit()

    with pytest.raises(TypeError):
        ai.define_value('catalog', 'banner', 'value')  # type: ignore[misc]  # pyright: ignore[reportCallIssue]

    assert ai.lookup_value(kind='catalog', name='banner') is None
    assert ai.lookup_value(kind='banner', name='catalog') is None


@pytest.mark.parametrize(
    ('kind', 'api'),
    [('middleware', 'define_middleware'), ('format', 'define_format'), ('defaultModel', 'Genkit(model=...)')],
)
def test_define_value_rejects_kinds_genkit_owns(kind: str, api: str) -> None:
    """`define_value` on a kind core reads with a fixed type raises and names the API to use."""
    ai = Genkit()

    with pytest.raises(ValueError, match=f'reserved; use {re.escape(api)}'):
        ai.define_value(kind=kind, name='audit', value={'not': 'the right type'})

    assert ai.lookup_value(kind=kind, name='audit') is None


def test_define_value_rejects_none() -> None:
    """`define_value(value=None)` raises, so None from `lookup_value` always means "not defined"."""
    ai = Genkit()

    with pytest.raises(ValueError, match='must not be None'):
        ai.define_value(kind='catalog', name='banner', value=None)

    ai.define_value(kind='catalog', name='banner', value='later')
    assert ai.lookup_value(kind='catalog', name='banner') == 'later'


def test_genkit_has_no_registry_attribute() -> None:
    """`ai.registry` raises AttributeError; app code uses the lookup and define methods."""
    ai = Genkit()

    with pytest.raises(AttributeError):
        _ = ai.registry  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_middleware_ctx_ai_is_the_app_on_every_generate_path() -> None:
    """`ctx.ai` is the app's Genkit for `ai.generate`, a prompt call, and Dev UI's `/util/generate`."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    seen: list[object] = []

    class PeekAi(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai)
            return await next_fn(params, ctx)

    ai.define_middleware(PeekAi, name='peek_ai')

    # 1. Veneer
    await ai.generate(model='test/echo', prompt='hi', use=[PeekAi()])

    # 2. Prompt
    prompt = ai.define_prompt(model='test/echo', prompt='hi', use=[PeekAi()])
    await prompt()

    # 3. Dev UI / reflection
    util = await ai._registry.resolve_action(ActionKind.UTIL, 'generate')  # pyright: ignore[reportPrivateUsage]
    assert util is not None
    await util.run(
        GenerateActionOptions(
            model='test/echo',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='peek_ai')],
        )
    )

    assert len(seen) == 3
    assert all(s is ai for s in seen)


@pytest.mark.asyncio
async def test_middleware_ctx_ai_lookup_model_finds_app_model() -> None:
    """Middleware calling `await ctx.ai.lookup_model('test/other')` gets the app's model and can answer with it."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    _define_answering_model(ai, 'test/other', 'from other')

    class SwapModel(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            other = await ctx.ai.lookup_model('test/other')
            assert other is not None
            result = await other.run(params.request)
            return result.response

    response = await ai.generate(model='test/echo', prompt='hi', use=[SwapModel()])

    assert response.text == 'from other'


@pytest.mark.asyncio
async def test_middleware_ctx_ai_lookup_value_sees_app_value() -> None:
    """A value from `ai.define_value` is what middleware gets from `ctx.ai.lookup_value`."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    ai.define_value(kind='greeting', name='default', value='hello')
    seen: list[object | None] = []

    class ReadValue(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai.lookup_value(kind='greeting', name='default'))
            seen.append(ctx.ai.lookup_value(kind='greeting', name='missing'))
            return await next_fn(params, ctx)

    await ai.generate(model='test/echo', prompt='hi', use=[ReadValue()])

    assert seen == ['hello', None]
