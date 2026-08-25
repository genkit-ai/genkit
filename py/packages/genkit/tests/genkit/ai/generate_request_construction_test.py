#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What a plugin handler sees after ai.generate: typed config, extras, errors, output."""

import pytest
from genkit_openai import OpenAIConfig
from pydantic import BaseModel

from genkit import Document, Genkit
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelConfig, ModelRequest, ModelResponse
from genkit._core._typing import Part, Role, TextPart


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


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part(root=TextPart(text='ok'))]))


@pytest.fixture
def ai_and_seen() -> tuple[Genkit, dict]:
    ai = Genkit()
    seen: dict = {}

    async def conforming(request: ModelRequest[ConformingCfg], ctx: ActionRunContext) -> ModelResponse:
        seen['config'] = request.config
        seen['request'] = request
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
async def test_invalid_value_raises_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A bad value on a declared field is GenkitError, not a raw ValidationError."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'conforming'"):
        await ai.generate(model='conforming', prompt='hi', config={'temperature': 'high'})


@pytest.mark.asyncio
async def test_invalid_config_with_docs_is_still_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(docs=..., config={'temperature': 'high'}) is GenkitError, not AttributeError."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'conforming'"):
        await ai.generate(
            model='conforming',
            prompt='hi',
            docs=[Document.from_text('ctx')],
            config={'temperature': 'high'},
        )


@pytest.mark.asyncio
async def test_strict_config_rejects_unknown_keys_as_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """extra='forbid' rejects unknown keys as GenkitError — the plugin opted in."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'strict'"):
        await ai.generate(model='strict', prompt='hi', config={'thinking': True})


@pytest.mark.asyncio
async def test_foreign_config_class_normalizes_to_conforming(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.generate(config=OpenAIConfig(...)) normalizes through veneer to the target plugin's config."""
    ai, seen = ai_and_seen
    await ai.generate(model='conforming', prompt='hi', config=OpenAIConfig(temperature=0.7))
    assert type(seen['config']) is ConformingCfg
    assert seen['config'].temperature == 0.7


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
