#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Middleware sees the config as the model's class, the same object the model gets.

generate builds the config once. Middleware changes fields on it; putting a
dict, None, or another class back into ``request.config`` fails the turn,
naming the middleware, before anything reaches the next layer.
"""

from collections.abc import Callable
from typing import Any, TypedDict

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator

from genkit import Genkit, GenkitError, Message, ModelResponse, Part
from genkit._core._action import ActionRunContext
from genkit._core._middleware import BaseMiddleware
from genkit._core._model import ModelRequest
from genkit._core._typing import FinishReason, Role
from genkit.model import ModelConfig


class StrictConfig(ModelConfig):
    """A plugin class with one setting of its own."""

    safe_prompt: bool | None = None


class TaskBudget(BaseModel):
    """A nested setting sent whole."""

    model_config = ConfigDict(extra='forbid')
    total: int


class NestedConfig(ModelConfig):
    """A Claude-shaped class with a nested object."""

    task_budget: TaskBudget | None = None


class _Model:
    """Records the config each request carried and answers 'ok'."""

    def __init__(self) -> None:
        self.configs: list[object] = []

    async def typed(self, request: ModelRequest[StrictConfig], _ctx: ActionRunContext) -> ModelResponse:
        self.configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    async def untyped(self, request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        self.configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


def _middleware(name: str, seen: list[object], change: Callable[[Any], None] | None = None) -> BaseMiddleware:
    """A wrap_model middleware named `name` that records the config it got, then applies `change`."""

    class _Mw(BaseMiddleware):
        async def wrap_model(self, params: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
            seen.append(params.request.config)
            if change is not None:
                change(params)
            return await next_fn(params, ctx)

    _Mw.__name__ = _Mw.__qualname__ = name
    return _Mw()


def _ai(*, typed: bool = True) -> tuple[Genkit, _Model]:
    ai = Genkit()
    model = _Model()
    ai.define_model(name='strict', fn=model.typed if typed else model.untyped, config_schema=StrictConfig)
    return ai, model


def _nested_ai() -> tuple[Genkit, _Model]:
    ai = Genkit()
    model = _Model()

    async def fn(request: ModelRequest[NestedConfig], _ctx: ActionRunContext) -> ModelResponse:
        model.configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='nested', fn=fn, config_schema=NestedConfig)
    return ai, model


def _set_field(name: str, value: object) -> Callable[[Any], None]:
    return lambda params: setattr(params.request.config, name, value)


def _copy_with(update: dict[str, Any]) -> Callable[[Any], None]:
    def change(params: Any) -> None:  # noqa: ANN401
        params.request.config = params.request.config.model_copy(update=update)

    return change


def _replace_with(value: object) -> Callable[[Any], None]:
    return lambda params: setattr(params.request, 'config', value)


@pytest.mark.asyncio
async def test_middleware_and_model_see_the_same_config_class() -> None:
    """With no change, both middleware layers and the model get a StrictConfig."""
    ai, model = _ai()
    outer: list[object] = []
    inner: list[object] = []

    await ai.generate(
        model='strict',
        prompt='hi',
        config={'temperature': 0.2},
        use=[_middleware('Outer', outer), _middleware('Inner', inner)],
    )

    assert [type(c) for c in (*outer, *inner, *model.configs)] == [StrictConfig] * 3
    assert model.configs[0].temperature == 0.2  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change',
    [_set_field('temperature', 0.1), _copy_with({'temperature': 0.1})],
    ids=['set_field', 'model_copy'],
)
async def test_middleware_config_change_reaches_inner_middleware_and_model_as_the_class(
    change: Callable[[Any], None],
) -> None:
    """temperature=0.1, set in place or via model_copy, reaches inner middleware and the model as StrictConfig."""
    ai, model = _ai()
    inner: list[object] = []

    await ai.generate(
        model='strict',
        prompt='hi',
        config={'temperature': 0.2, 'safe_prompt': True},
        use=[_middleware('Rewrite', [], change), _middleware('Inner', inner)],
    )

    for config in (*inner, *model.configs):
        assert isinstance(config, StrictConfig)
        assert config.temperature == 0.1
        assert config.safe_prompt is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('value', 'got'),
    [
        ({'temperature': 0.1}, 'dict'),
        (None, 'None'),
        (ModelConfig(temperature=0.1), 'genkit._core._model.ModelConfig'),
    ],
    ids=['dict', 'none', 'other_class'],
)
async def test_middleware_replacing_config_fails_naming_middleware_and_what_it_put(value: object, got: str) -> None:
    """Putting a dict, None, or another class into request.config fails before inner middleware or the model run."""
    ai, model = _ai()
    inner: list[object] = []

    response = await ai.generate(
        model='strict',
        prompt='hi',
        use=[_middleware('Rewrite', [], _replace_with(value)), _middleware('Inner', inner)],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message == (
        f"strict: middleware 'Rewrite' replaced request.config with {got}; change fields on request.config instead"
    )
    assert response.error is not None
    assert response.error.status == 'INVALID_ARGUMENT'
    assert inner == []
    assert model.configs == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change',
    [_set_field('temperature', 'hot'), _copy_with({'temperature': 'hot'})],
    ids=['set_field', 'model_copy'],
)
async def test_middleware_bad_config_value_fails_naming_middleware_and_field(change: Callable[[Any], None]) -> None:
    """temperature='hot', set in place or via model_copy, fails naming Rewrite and temperature; the model never runs."""
    ai, model = _ai()

    response = await ai.generate(model='strict', prompt='hi', use=[_middleware('Rewrite', [], change)])

    assert response.finish_reason == FinishReason.FAILED
    assert response.finish_message is not None
    assert response.finish_message.startswith("strict: middleware 'Rewrite' set config 'temperature': ")
    assert model.configs == []


@pytest.mark.asyncio
async def test_middleware_nested_dict_reaches_model_as_the_nested_class() -> None:
    """`config.task_budget = {'total': 1}` reaches the model as TaskBudget(total=1), not a dict."""
    ai, model = _nested_ai()

    await ai.generate(
        model='nested', prompt='hi', use=[_middleware('Rewrite', [], _set_field('task_budget', {'total': 1}))]
    )

    config = model.configs[0]
    assert isinstance(config, NestedConfig)
    assert config.task_budget == TaskBudget(total=1)


@pytest.mark.asyncio
async def test_middleware_incomplete_nested_dict_fails_naming_the_field() -> None:
    """`config.task_budget = {}` fails naming task_budget; the model never runs."""
    ai, model = _nested_ai()

    response = await ai.generate(
        model='nested', prompt='hi', use=[_middleware('Rewrite', [], _set_field('task_budget', {}))]
    )

    assert response.finish_message is not None
    assert response.finish_message.startswith("nested: middleware 'Rewrite' set config 'task_budget': ")
    assert model.configs == []


@pytest.mark.asyncio
async def test_middleware_unknown_config_key_via_model_copy_fails_naming_the_key() -> None:
    """model_copy(update={'temprature': 0.1}) fails naming the key and pointing at config['extra']."""
    ai, model = _ai()

    response = await ai.generate(
        model='strict', prompt='hi', use=[_middleware('Rewrite', [], _copy_with({'temprature': 0.1}))]
    )

    assert response.finish_message == (
        "strict: middleware 'Rewrite' set unknown config key 'temprature'; "
        "put provider-only settings in config['extra']"
    )
    assert model.configs == []


@pytest.mark.asyncio
async def test_middleware_replacing_config_blames_the_layer_that_did_it() -> None:
    """When the inner of two middleware puts a dict in, the error names Inner, not Outer."""
    ai, model = _ai()

    response = await ai.generate(
        model='strict',
        prompt='hi',
        use=[_middleware('Outer', []), _middleware('Inner', [], _replace_with({'temperature': 0.1}))],
    )

    assert response.finish_message is not None
    assert "middleware 'Inner' replaced request.config with dict" in response.finish_message
    assert model.configs == []


@pytest.mark.asyncio
async def test_model_that_takes_plain_model_request_still_gets_a_dict_middleware_put_in() -> None:
    """A model typed as plain ModelRequest takes dicts, so a dict from middleware reaches it unchanged."""
    ai, model = _ai(typed=False)
    outer: list[object] = []

    await ai.generate(
        model='strict',
        prompt='hi',
        config={'temperature': 0.2},
        use=[_middleware('Rewrite', outer, _replace_with({'temperature': 0.1}))],
    )

    assert outer == [{'temperature': 0.2}]
    assert model.configs == [{'temperature': 0.1}]


class TypedDictConfig(TypedDict, total=False):
    """A model whose request config is a TypedDict, not a pydantic class."""

    temperature: float


@pytest.mark.asyncio
async def test_model_typed_with_typeddict_config_runs_through_middleware() -> None:
    """ModelRequest[TypedDict] skips the check: the model runs and gets the dict."""
    ai = Genkit()
    configs: list[object] = []

    async def fn(request: ModelRequest[TypedDictConfig], _ctx: ActionRunContext) -> ModelResponse:
        configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='td', fn=fn)

    response = await ai.generate(
        model='td', prompt='hi', config={'temperature': 0.1}, use=[_middleware('Passthrough', [])]
    )

    assert response.text == 'ok'
    assert configs == [{'temperature': 0.1}]


class PrefixedConfig(ModelConfig):
    """A config whose validator isn't idempotent, plus a frozen field."""

    region: str | None = Field(default=None, frozen=True)

    @field_validator('version')
    @classmethod
    def _prefix(cls, value: str | None) -> str | None:
        return None if value is None else f'models/{value}'


@pytest.mark.asyncio
async def test_fields_middleware_did_not_touch_are_not_revalidated() -> None:
    """Three pass-through layers: version stays 'models/v1', frozen region passes, the stop list is the same object."""
    ai = Genkit()
    configs: list[Any] = []

    async def fn(request: ModelRequest[PrefixedConfig], _ctx: ActionRunContext) -> ModelResponse:
        configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='prefixed', fn=fn, config_schema=PrefixedConfig)
    outer: list[Any] = []

    response = await ai.generate(
        model='prefixed',
        prompt='hi',
        config={'version': 'v1', 'region': 'us', 'stop_sequences': ['END']},
        use=[_middleware('A', outer), _middleware('B', []), _middleware('C', [])],
    )

    assert response.text == 'ok'
    assert configs[0].version == 'models/v1'
    assert configs[0].region == 'us'
    assert configs[0].stop_sequences is outer[0].stop_sequences


@pytest.mark.asyncio
async def test_middleware_changed_field_is_validated_once() -> None:
    """A middleware setting version='v2' reaches the model as 'models/v2', through later layers too."""
    ai = Genkit()
    configs: list[Any] = []

    async def fn(request: ModelRequest[PrefixedConfig], _ctx: ActionRunContext) -> ModelResponse:
        configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='prefixed', fn=fn, config_schema=PrefixedConfig)

    await ai.generate(
        model='prefixed',
        prompt='hi',
        use=[_middleware('Rewrite', [], _set_field('version', 'v2')), _middleware('B', []), _middleware('C', [])],
    )

    assert configs[0].version == 'models/v2'


class _RetryOnce(BaseMiddleware):
    """Calls next again when it raises, like a broad user retry."""

    async def wrap_model(self, params: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
        try:
            return await next_fn(params, ctx)
        except GenkitError:
            return await next_fn(params, ctx)


@pytest.mark.asyncio
async def test_outer_retry_does_not_take_the_blame_for_inner_swap() -> None:
    """When an outer layer retries after Inner put a dict in, the error still names Inner."""
    ai, model = _ai()

    response = await ai.generate(
        model='strict',
        prompt='hi',
        use=[_RetryOnce(), _middleware('Inner', [], _replace_with({'temperature': 0.1}))],
    )

    assert response.finish_message == (
        "strict: middleware 'Inner' replaced request.config with dict; change fields on request.config instead"
    )
    assert model.configs == []
