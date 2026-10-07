#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A model that registered a config class receives that class as request.config."""

from collections.abc import Callable
from typing import Any

import pytest

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._action import Action, ActionKind, ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import ModelRequest
from genkit._core._typing import ActionMetadata, Operation, Role
from genkit.middleware import BaseMiddleware, GenerateMiddlewareContext, ModelHookParams
from genkit.model import ModelConfig, model
from genkit.plugin_api import Plugin


class MistralConfig(ModelConfig):
    """Stand-in for a plugin config class."""


class OtherConfig(ModelConfig):
    """Some other plugin's class."""


class GeminiConfig(ModelConfig):
    """A child of ModelConfig, used where the annotation must be the exact class."""

    safety_settings: dict[str, str] | None = None


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


def _user_request(*, config: object = None) -> ModelRequest:
    kwargs: dict[str, object] = {
        'messages': [Message(role=Role.USER, content=[Part.from_text('hi')])],
    }
    if config is not None:
        kwargs['config'] = config
    return ModelRequest(**kwargs)


def _define(
    *,
    fn: Callable[..., Any],
    name: str = 'mistral/bare',
    config_schema: type | dict[str, object] | None = MistralConfig,
) -> tuple[Genkit, Action, list[object]]:
    seen: list[object] = []

    async def record(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        seen.append(request.config)
        return OK

    record.__annotations__.update(fn.__annotations__)
    ai = Genkit()
    action = ai.define_model(name=name, fn=record, config_schema=config_schema)
    return ai, action, seen


@pytest.mark.asyncio
async def test_define_model_with_config_class_and_plain_annotation_gets_that_class() -> None:
    """`request.config` is a `MistralConfig` and `request.config.temperature == 0.3`."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    await ai.generate(model='mistral/bare', prompt='hi', config={'temperature': 0.3})

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


@pytest.mark.asyncio
async def test_define_model_with_config_class_and_no_config_gets_default_instance() -> None:
    """With no `config=`, `request.config` is `MistralConfig()`, the same as a typed annotation gets."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    await ai.generate(model='mistral/bare', prompt='hi')

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1] == MistralConfig()


@pytest.mark.asyncio
async def test_define_model_with_config_class_and_model_config_raises() -> None:
    """`config=ModelConfig(...)` on a model registered as `MistralConfig` raises before the model runs."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    with pytest.raises(GenkitError) as err:
        await ai.generate(model='mistral/bare', prompt='hi', config=ModelConfig(temperature=0.3))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert 'MistralConfig' in str(err.value)
    assert 'ModelConfig' in str(err.value)
    assert seen == []


@pytest.mark.asyncio
async def test_define_model_with_config_class_and_any_annotation_gets_that_class() -> None:
    """`request: ModelRequest[Any]` is treated like plain `ModelRequest`."""

    async def generate(request: ModelRequest[Any], _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    await ai.generate(model='mistral/bare', prompt='hi', config={'temperature': 0.3})

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


@pytest.mark.asyncio
async def test_define_model_with_typed_annotation_is_unchanged() -> None:
    """`ModelRequest[MistralConfig]` still gets `MistralConfig` (control)."""

    async def generate(request: ModelRequest[MistralConfig], _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    await ai.generate(model='mistral/bare', prompt='hi', config={'temperature': 0.3})

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


@pytest.mark.asyncio
async def test_define_model_without_config_class_keeps_dict() -> None:
    """No `config_schema` means `request.config` stays the dict (control)."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate, name='loose', config_schema=None)
    await ai.generate(model='loose', prompt='hi', config={'temperature': 0.3})

    assert seen[-1] == {'temperature': 0.3}


@pytest.mark.asyncio
async def test_define_model_with_json_schema_dict_keeps_dict() -> None:
    """A JSON-schema dict `config_schema` has no Python class, so the dict is kept."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    schema = {'type': 'object', 'properties': {'temperature': {'type': 'number'}}}
    ai, _action, seen = _define(fn=generate, name='json-schema', config_schema=schema)
    await ai.generate(model='json-schema', prompt='hi', config={'temperature': 0.3})

    assert seen[-1] == {'temperature': 0.3}


@pytest.mark.asyncio
async def test_plugin_model_builder_with_plain_annotation_gets_config_class() -> None:
    """A plugin's `model(name, fn, config_schema=X)` returned from `resolve` gives `fn` an `X`."""
    seen: list[object] = []

    class Acme(Plugin):
        name = 'acme'

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            if action_type != ActionKind.MODEL or name != 'bare':
                return None

            async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
                seen.append(request.config)
                return OK

            return model(name, generate, config_schema=MistralConfig)

        async def list_actions(self) -> list[ActionMetadata]:
            return [ActionMetadata(action_type=ActionKind.MODEL, name='acme/bare')]

    ai = Genkit(plugins=[Acme()])
    await ai.generate(model='acme/bare', prompt='hi', config={'temperature': 0.3})

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


@pytest.mark.asyncio
async def test_background_model_with_config_class_and_plain_annotation_gets_that_class() -> None:
    """`generate_operation` on a `background_model(..., config_schema=X)` passes `start` an `X`."""
    seen: list[object] = []

    async def start(request: ModelRequest, _ctx: ActionRunContext) -> Operation:
        seen.append(request.config)
        return Operation(id='job-1', done=False)

    async def check(op: Operation, _ctx: ActionRunContext) -> Operation:
        return op

    ai = Genkit()
    ai.define_background_model(name='bg', start=start, check=check, config_schema=MistralConfig)
    await ai.generate_operation(model='bg', prompt='hi', config={'temperature': 0.3})

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


def test_define_model_annotation_and_config_class_disagree_raises() -> None:
    """`ModelRequest[OtherConfig]` with `config_schema=MistralConfig` raises when the model is defined."""

    async def generate(request: ModelRequest[OtherConfig], _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai = Genkit()
    with pytest.raises(GenkitError) as err:
        ai.define_model(name='mistral/bare', fn=generate, config_schema=MistralConfig)

    assert err.value.status == 'INVALID_ARGUMENT'
    assert 'OtherConfig' in str(err.value)
    assert 'MistralConfig' in str(err.value)


@pytest.mark.asyncio
async def test_define_model_action_run_with_dict_config_gets_that_class() -> None:
    """A direct model run with `config={'temperature': 0.3}` still gives the model `MistralConfig`."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    _ai, action, seen = _define(fn=generate)
    await action.run(_user_request(config={'temperature': 0.3}))

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.3


@pytest.mark.asyncio
async def test_define_model_action_run_with_no_config_gets_empty_instance() -> None:
    """A direct model run with no config, or `config=None`, gives the model `MistralConfig()`."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    _ai, action, seen = _define(fn=generate)
    await action.run(_user_request())
    await action.run(ModelRequest(messages=_user_request().messages, config=None))

    assert len(seen) == 2
    assert isinstance(seen[0], MistralConfig)
    assert seen[0] == MistralConfig()
    assert isinstance(seen[1], MistralConfig)
    assert seen[1] == MistralConfig()


@pytest.mark.asyncio
async def test_generate_middleware_rewritten_dict_config_still_gets_the_class() -> None:
    """Middleware that rewrites `request.config` to a dict still gives the model `MistralConfig`."""

    class RewriteToDict(BaseMiddleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: GenerateMiddlewareContext,
            next_fn: Callable[[ModelHookParams, GenerateMiddlewareContext], Any],
        ) -> ModelResponse:
            params.request.config = {'temperature': 0.4}
            return await next_fn(params, ctx)

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai, _action, seen = _define(fn=generate)
    await ai.generate(
        model='mistral/bare',
        prompt='hi',
        config={'temperature': 0.1},
        use=[RewriteToDict()],
    )

    assert isinstance(seen[-1], MistralConfig)
    assert seen[-1].temperature == 0.4


@pytest.mark.asyncio
async def test_define_model_action_run_with_other_plugin_class_raises() -> None:
    """A direct model run with another plugin's class raises INVALID_ARGUMENT."""

    async def generate(request: ModelRequest, _ctx: ActionRunContext) -> ModelResponse:
        return OK

    _ai, action, seen = _define(fn=generate)
    with pytest.raises(GenkitError) as err:
        await action.run(_user_request(config=OtherConfig(temperature=0.2)))

    assert err.value.status == 'INVALID_ARGUMENT'
    assert 'MistralConfig' in str(err.value)
    assert 'OtherConfig' in str(err.value)
    assert seen == []


def test_define_model_dict_annotation_with_config_class_raises() -> None:
    """`ModelRequest[dict]` with a config class raises when the model is defined."""

    async def generate(request: ModelRequest[dict], _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai = Genkit()
    with pytest.raises(GenkitError) as err:
        ai.define_model(name='mistral/bare', fn=generate, config_schema=MistralConfig)

    assert err.value.status == 'INVALID_ARGUMENT'
    assert 'dict' in str(err.value)
    assert 'MistralConfig' in str(err.value)


def test_define_model_parent_config_annotation_with_child_class_raises() -> None:
    """`ModelRequest[ModelConfig]` with `config_schema=GeminiConfig` raises; the annotation must be the exact class."""

    async def generate(request: ModelRequest[ModelConfig], _ctx: ActionRunContext) -> ModelResponse:
        return OK

    ai = Genkit()
    with pytest.raises(GenkitError) as err:
        ai.define_model(name='gemini/bare', fn=generate, config_schema=GeminiConfig)

    assert err.value.status == 'INVALID_ARGUMENT'
    assert 'ModelConfig' in str(err.value)
    assert 'GeminiConfig' in str(err.value)
