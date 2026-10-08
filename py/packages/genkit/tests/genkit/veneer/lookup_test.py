#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What app code and middleware get back from lookup_model, lookup_value, and define_value."""

from collections.abc import Awaitable, Callable

import pytest

from genkit import FinishReason, Genkit, Message, ModelResponse, Part, Role
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams
from genkit.model import ModelRequest, model
from genkit.plugin_api import Action, ActionKind, ActionMetadata, Plugin

NextModel = Callable[[ModelHookParams, GenerateMiddlewareContext], Awaitable[ModelResponse]]


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _define_answering_model(ai: Genkit, name: str, text: str) -> None:
    async def fn(request: ModelRequest) -> ModelResponse:
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

        async def fn(request: ModelRequest) -> ModelResponse:
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


def test_genkit_has_no_registry_attribute() -> None:
    """`ai.registry` raises AttributeError; app code uses the lookup and define methods."""
    ai = Genkit()

    with pytest.raises(AttributeError):
        _ = ai.registry  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_middleware_ctx_lookup_model_finds_app_model() -> None:
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

    assert response.finish_reason == FinishReason.STOP
    assert response.message is not None
    assert response.text == 'from other'


@pytest.mark.asyncio
async def test_middleware_ctx_lookup_value_sees_app_value() -> None:
    """A value from `ai.define_value` is what middleware gets from `ctx.ai.lookup_value`."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    ai.define_value(kind='greeting', name='default', value='hello from the app')
    seen: list[object | None] = []

    class ReadGreeting(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai.lookup_value(kind='greeting', name='default'))
            seen.append(ctx.ai.lookup_value(kind='greeting', name='missing'))
            return await next_fn(params, ctx)

    response = await ai.generate(model='test/echo', prompt='hi', use=[ReadGreeting()])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from echo'
    assert seen == ['hello from the app', None]


@pytest.mark.asyncio
async def test_middleware_ctx_has_no_registry_attribute() -> None:
    """`ctx.ai.registry` raises AttributeError inside middleware; the call still finishes normally."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    errors: list[BaseException] = []

    class PeekRegistry(BaseMiddleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: GenerateMiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            try:
                _ = ctx.ai.registry  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
            except AttributeError as exc:
                errors.append(exc)
            return await next_fn(params, ctx)

    response = await ai.generate(model='test/echo', prompt='hi', use=[PeekRegistry()])

    assert response.finish_reason == FinishReason.STOP
    assert response.text == 'from echo'
    assert len(errors) == 1
    assert isinstance(errors[0], AttributeError)
