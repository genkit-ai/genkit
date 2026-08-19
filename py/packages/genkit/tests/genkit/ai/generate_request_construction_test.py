#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Typed-request fast path through generate: happy path, error contract, escape hatch."""

import pytest
from pydantic import BaseModel

from genkit import Genkit
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelConfig, ModelRequest, ModelResponse
from genkit._core._typing import Part, Role, TextPart


class ConformingCfg(BaseModel):
    """Follows the extra='allow' convention: the escape hatch stays open."""

    model_config = {'extra': 'allow'}
    temperature: float | None = None


class StrictCfg(BaseModel):
    """Deliberately closes the hatch — a legitimate plugin choice."""

    model_config = {'extra': 'forbid'}
    temperature: float | None = None


class PluginOnlyCfg(ModelConfig):
    """A field the common config does not have. The plugin schema owns it."""

    duration_seconds: int | None = None


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='ok'))]))


@pytest.fixture
def ai_and_seen() -> tuple[Genkit, dict]:
    ai = Genkit()
    seen: dict = {}

    async def conforming(request: ModelRequest[ConformingCfg], ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        return OK

    async def strict(request: ModelRequest[StrictCfg], ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        return OK

    async def bare(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        return OK

    async def plugin_only(request: ModelRequest[PluginOnlyCfg], ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        return OK

    ai.define_model(name='conforming', fn=conforming)
    ai.define_model(name='strict', fn=strict)
    ai.define_model(name='bare', fn=bare)
    ai.define_model(name='plugin_only', fn=plugin_only)
    return ai, seen


@pytest.mark.asyncio
async def test_typed_plugin_receives_typed_config(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Fast path: the dict config arrives parsed into the plugin's class."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config={'temperature': 0.7})
    assert isinstance(seen['config'], ConformingCfg)
    assert seen['config'].temperature == 0.7


@pytest.mark.asyncio
async def test_typed_plugin_receives_plugin_only_fields_from_instance(
    ai_and_seen: tuple[Genkit, dict],
) -> None:
    """Plugin-only knobs are legal because the action parses its schema.

    ``duration_seconds`` is not on the common config. Passing the object or a
    dict both land as ``PluginOnlyCfg.duration_seconds`` — not because generate
    kept the instance, because ``ModelRequest[PluginOnlyCfg]`` rehydrated it.
    """
    ai, seen = ai_and_seen
    cfg = PluginOnlyCfg(duration_seconds=8)

    await ai.generate(model='plugin_only', prompt='hi', config=cfg)
    assert type(seen['config']) is PluginOnlyCfg
    assert seen['config'].duration_seconds == 8

    await ai.generate(model='plugin_only', prompt='hi', config={'duration_seconds': 8})
    assert type(seen['config']) is PluginOnlyCfg
    assert seen['config'].duration_seconds == 8


@pytest.mark.asyncio
async def test_bare_plugin_receives_raw_dict(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Pass-through carrier: bare ModelRequest never transforms config."""
    ai, seen = ai_and_seen
    await ai.generate(model='bare', prompt='hi', config={'temperature': 0.7, 'anything': 1})
    assert seen['config'] == {'temperature': 0.7, 'anything': 1}
    assert type(seen['config']) is dict


@pytest.mark.asyncio
async def test_omitted_config_yields_empty_typed_config(ai_and_seen: tuple[Genkit, dict]) -> None:
    """The request is never None; absent config means an empty typed config."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi')
    assert isinstance(seen['config'], ConformingCfg)
    assert seen['config'].temperature is None


@pytest.mark.asyncio
async def test_unknown_keys_reach_plugin_via_model_extra(ai_and_seen: tuple[Genkit, dict]) -> None:
    """D1 escape hatch, end to end: unknown keys survive to model_extra."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config={'thinking': {'budget': 8192}})
    assert seen['config'].model_extra == {'thinking': {'budget': 8192}}


@pytest.mark.asyncio
async def test_invalid_value_raises_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Error contract: bad values in declared fields surface as GenkitError, never raw ValidationError."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'conforming'"):
        await ai.generate(model='conforming', prompt='hi', config={'temperature': 'high'})


@pytest.mark.asyncio
async def test_strict_config_rejects_unknown_keys_as_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A plugin that closes the hatch (extra='forbid') gets a loud, wrapped error — its choice."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'strict'"):
        await ai.generate(model='strict', prompt='hi', config={'thinking': True})
