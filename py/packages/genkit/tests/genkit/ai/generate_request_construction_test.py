#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What a plugin handler sees after ai.generate: typed config, extras, errors, output."""

from typing import Any, cast

import pytest
from genkit_openai import OpenAIConfig
from pydantic import BaseModel

from genkit import Document, Genkit, GenkitError, Part
from genkit._core._action import ActionRunContext
from genkit._core._model import Message, ModelConfig, ModelRequest, ModelResponse
from genkit._core._typing import Role


class ConformingCfg(BaseModel):
    """Unknown keys are kept."""

    model_config = {'extra': 'allow'}
    temperature: float | None = None


class StrictCfg(BaseModel):
    """Unknown keys are rejected."""

    model_config = {'extra': 'forbid'}
    temperature: float | None = None


class PluginOnlyCfg(ModelConfig):
    """A knob the common config does not have."""

    duration_seconds: int | None = None


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


@pytest.fixture
def ai_and_seen() -> tuple[Genkit, dict]:
    ai = Genkit()
    seen: dict = {}

    async def conforming(request: ModelRequest[ConformingCfg], ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        seen['request'] = request
        if request.output_format == 'json':
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{}')]))
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
    """ai.generate(config={'temperature': 0.7}) arrives as MyConfig.temperature."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config={'temperature': 0.7})
    assert isinstance(seen['config'], ConformingCfg)
    assert seen['config'].temperature == 0.7


@pytest.mark.asyncio
async def test_matching_config_instance_passes_through(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(config=MyConfig(...)) arrives as that same class. MyConfig subclasses ModelConfig."""
    ai, seen = ai_and_seen
    await ai.generate(model='plugin_only', prompt='hi', config=PluginOnlyCfg(duration_seconds=8))
    assert type(seen['config']) is PluginOnlyCfg
    assert seen['config'].duration_seconds == 8


@pytest.mark.asyncio
async def test_plain_basemodel_config_accepted_at_generate(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(config=) accepts a dict or a BaseModel instance."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config=ConformingCfg(temperature=0.7))
    assert type(seen['config']) is ConformingCfg
    assert seen['config'].temperature == 0.7


@pytest.mark.asyncio
async def test_typed_plugin_receives_plugin_only_fields_from_instance(
    ai_and_seen: tuple[Genkit, dict],
) -> None:
    """A field only the plugin schema owns still arrives, from an instance or a dict."""
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
    """A handler annotated ModelRequest (no type param) still sees the raw dict."""
    ai, seen = ai_and_seen
    await ai.generate(model='bare', prompt='hi', config={'temperature': 0.7, 'anything': 1})
    assert seen['config'] == {'temperature': 0.7, 'anything': 1}
    assert type(seen['config']) is dict


@pytest.mark.asyncio
async def test_omitted_config_yields_empty_typed_config(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Omitting config still gives the plugin an empty MyConfig, not None."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi')
    assert isinstance(seen['config'], ConformingCfg)
    assert seen['config'].temperature is None


@pytest.mark.asyncio
async def test_unknown_keys_reach_plugin_via_model_extra(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A key the plugin schema does not declare still reaches model_extra."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config={'thinking': {'budget': 8192}})
    assert seen['config'].model_extra == {'thinking': {'budget': 8192}}


@pytest.mark.asyncio
async def test_invalid_config_value_raises_before_the_model_runs(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ModelRequest[Cfg] alone sets the model's config class, so generate checks the call config."""
    ai, seen = ai_and_seen
    with pytest.raises(GenkitError, match=r"conforming: config 'temperature'") as excinfo:
        await ai.generate(model='conforming', prompt='hi', config={'temperature': 'high'})
    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert 'config' not in seen


@pytest.mark.asyncio
async def test_invalid_config_with_docs_raises_before_the_model_runs(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(docs=..., config={'temperature': 'high'}) is the same call-time error."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match=r"conforming: config 'temperature'"):
        await ai.generate(
            model='conforming',
            prompt='hi',
            docs=[Document.from_text('ctx')],
            config={'temperature': 'high'},
        )


@pytest.mark.asyncio
async def test_strict_config_rejects_unknown_keys_at_call_time(ai_and_seen: tuple[Genkit, dict]) -> None:
    """extra='forbid' on the annotated class rejects unknown keys before the model runs."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match=r"strict: unknown config key 'thinking'"):
        await ai.generate(model='strict', prompt='hi', config={'thinking': True})


@pytest.mark.asyncio
async def test_invalid_config_from_a_raw_run_raises_at_the_boundary(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A Dev UI-shaped run skips generate's check; the action boundary still names the model."""
    ai, _ = ai_and_seen
    action = await ai.registry.resolve_model('strict')
    assert action is not None
    with pytest.raises(GenkitError, match="Invalid input for model 'strict'"):
        await action.run(cast(Any, {'messages': [], 'config': {'thinking': True}}))


@pytest.mark.asyncio
async def test_foreign_config_class_raises(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(config=OpenAIConfig(...)) on a ModelRequest[ConformingCfg] model is a caller mistake."""
    ai, seen = ai_and_seen
    with pytest.raises(GenkitError, match=r'config must be .*ConformingCfg or a mapping, got .*OpenAIConfig'):
        await ai.generate(model='conforming', prompt='hi', config=OpenAIConfig(temperature=0.7))
    assert 'config' not in seen


@pytest.mark.asyncio
async def test_output_format_reaches_the_plugin(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(output_format='json') is what the plugin reads as request.output_format."""
    ai, seen = ai_and_seen
    await ai.generate(
        model='conforming',
        prompt='hi',
        output_format='json',
        output_schema={'type': 'object'},
    )
    assert seen['request'].output_format == 'json'
    assert seen['request'].output_schema == {'type': 'object'}
    assert seen['request'].output.format == 'json'
