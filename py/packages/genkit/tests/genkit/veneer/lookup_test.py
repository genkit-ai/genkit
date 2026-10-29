#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What app code and middleware get back from lookup_model, lookup_tool, list_tools, lookup_value, and define_value."""

import re
from collections.abc import Awaitable, Callable

import pytest

from genkit import (
    ActionRunContext,
    FinishReason,
    Genkit,
    Message,
    ModelResponse,
    Operation,
    Part,
    Role,
    Tool,
    tool as genkit_tool,
)
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, MiddlewareRef, ModelHookParams
from genkit.model import GenerateActionOptions, ModelRequest, model
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin

NextModel = Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]]


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _define_answering_model(ai: Genkit, name: str, text: str) -> None:
    async def fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
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

        async def fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
            return _answer('from lazy fast')

        return model(name, fn)

    async def list_actions(self) -> list[ActionMetadata]:
        return []


@pytest.mark.asyncio
async def test_lookup_model_returns_registered_model_action() -> None:
    """`await ai.lookup_model('test/echo')` returns the model action, and running it gets that model's answer."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'echoed')

    action = await ai.lookup_model('test/echo')

    assert action is not None
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

    async def start(request: ModelRequest, ctx: ActionRunContext) -> Operation:
        return Operation(id='op-1')

    async def check(operation: Operation, ctx: ActionRunContext) -> Operation:
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


class ToolPlugin(Plugin):
    """Registers `catalog/stock` from init, the way a plugin ships tools."""

    name = 'catalog'

    async def init(self) -> list[Action]:
        async def stock(sku: str) -> int:
            return 3

        return [genkit_tool(stock, name='catalog/stock').action()]

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []


@pytest.mark.asyncio
async def test_list_tools_returns_every_defined_tool_as_a_callable_tool() -> None:
    """`ai.list_tools()` gives one Tool per defined tool, each describing and running like the decorated one."""
    ai = Genkit()

    @ai.tool()
    async def menu_price(dish: str) -> float:
        """Price of a dish in dollars."""
        return 14.5

    @ai.tool()
    async def opening_hours() -> str:
        """When the restaurant is open."""
        return '9-5'

    tools = {tool.name: tool for tool in await ai.list_tools()}

    assert sorted(tools) == ['menu_price', 'opening_hours']
    assert all(isinstance(tool, Tool) for tool in tools.values())
    assert tools['menu_price'].definition() == menu_price.definition()
    assert (await tools['menu_price']('ramen')).output == 14.5


@pytest.mark.asyncio
async def test_list_tools_on_an_app_without_tools_is_empty() -> None:
    """An app that defines no tools lists none."""
    assert await Genkit().list_tools() == []


@pytest.mark.asyncio
async def test_list_tools_reads_the_app_again_on_every_call() -> None:
    """A tool defined after one listing is in the next one."""
    ai = Genkit()
    assert await ai.list_tools() == []

    @ai.tool()
    async def menu_price(dish: str) -> float:
        return 14.5

    assert [tool.name for tool in await ai.list_tools()] == ['menu_price']


@pytest.mark.asyncio
async def test_list_tools_includes_interrupts() -> None:
    """An interrupt is a tool, so `list_tools` lists it, as `lookup_tool` finds it."""
    ai = Genkit()
    ai.define_interrupt('confirm_order', description='Ask the diner to confirm.')

    assert [tool.name for tool in await ai.list_tools()] == ['confirm_order']


@pytest.mark.asyncio
async def test_list_tools_includes_tools_a_plugin_registers_at_init() -> None:
    """A plugin's tools are listed before anything has used the plugin."""
    ai = Genkit(plugins=[ToolPlugin()])

    assert [tool.name for tool in await ai.list_tools()] == ['catalog/stock']


class FailingPlugin(Plugin):
    """Fails to start, as a plugin with a bad API key does."""

    name = 'failing'

    async def init(self) -> list[Action]:
        raise RuntimeError('no API key')

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[ActionMetadata]:
        return []


@pytest.mark.asyncio
async def test_list_tools_raises_when_a_plugin_fails_to_start() -> None:
    """A listing that silently dropped a plugin's tools would look complete, so a failed start raises."""
    ai = Genkit(plugins=[FailingPlugin()])

    @ai.tool()
    async def menu_price(dish: str) -> float:
        return 14.5

    with pytest.raises(RuntimeError, match='no API key'):
        await ai.list_tools()


class AdvertisingToolPlugin(Plugin):
    """Advertises `shelf/count` and builds it only when resolved."""

    name = 'shelf'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.TOOL or name != 'count':
            return None

        async def count(sku: str) -> int:
            return 7

        return genkit_tool(count, name='shelf/count').action()

    async def list_actions(self) -> list[ActionMetadata]:
        return [ActionMetadata(name='shelf/count', action_type=ActionKind.TOOL)]


@pytest.mark.asyncio
async def test_list_tools_includes_an_advertised_plugin_tool_only_once_looked_up() -> None:
    """Listing does not build tools a plugin only advertises; a lookup registers them."""
    ai = Genkit(plugins=[AdvertisingToolPlugin()])

    assert await ai.list_tools() == []
    assert await ai.lookup_tool('shelf/count') is not None
    assert [tool.name for tool in await ai.list_tools()] == ['shelf/count']


@pytest.mark.asyncio
async def test_list_tools_leaves_out_dynamic_action_provider_tools() -> None:
    """A provider's tools are not defined on the app, so listing never reaches the provider."""
    ai = Genkit()
    listed = 0

    async def provide() -> dict[str, list[Action]]:
        nonlocal listed
        listed += 1

        async def remote_search(query: str) -> str:
            return query

        return {'tool': [genkit_tool(remote_search, name='remote_search').action()]}

    ai.define_dynamic_action_provider('remote', provide)

    assert await ai.list_tools() == []
    assert listed == 0


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

    class PeekAi(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
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

    class SwapModel(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
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

    class ReadValue(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai.lookup_value(kind='greeting', name='default'))
            seen.append(ctx.ai.lookup_value(kind='greeting', name='missing'))
            return await next_fn(params, ctx)

    await ai.generate(model='test/echo', prompt='hi', use=[ReadValue()])

    assert seen == ['hello', None]
