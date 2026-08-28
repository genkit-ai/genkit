#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What a plugin handler sees after ai.embed: typed options, extras, errors."""

import pytest
from genkit_openai import OpenAIConfig
from pydantic import BaseModel

from genkit import Genkit
from genkit._core._error import GenkitError
from genkit._core._model import EmbedRequest
from genkit._core._typing import Embedding, EmbedResponse


class ConformingCfg(BaseModel):
    """Unknown keys are kept."""

    model_config = {'extra': 'allow'}
    task_type: str | None = None
    temperature: float | None = None


class StrictCfg(BaseModel):
    """Unknown keys are rejected."""

    model_config = {'extra': 'forbid'}
    task_type: str | None = None


class PluginOnlyCfg(BaseModel):
    """A knob the common options bag does not have."""

    output_dimensionality: int | None = None


OK = EmbedResponse(embeddings=[Embedding(embedding=[1.0])])


@pytest.fixture
def ai_and_seen() -> tuple[Genkit, dict]:
    ai = Genkit()
    seen: dict = {}

    async def conforming(request: EmbedRequest[ConformingCfg]) -> EmbedResponse:
        seen['options'] = request.options
        seen['request'] = request
        return OK

    async def strict(request: EmbedRequest[StrictCfg]) -> EmbedResponse:
        seen['options'] = request.options
        return OK

    async def bare(request: EmbedRequest) -> EmbedResponse:
        seen['options'] = request.options
        return OK

    async def plugin_only(request: EmbedRequest[PluginOnlyCfg]) -> EmbedResponse:
        seen['options'] = request.options
        return OK

    ai.define_embedder(name='conforming', fn=conforming)
    ai.define_embedder(name='strict', fn=strict)
    ai.define_embedder(name='bare', fn=bare)
    ai.define_embedder(name='plugin_only', fn=plugin_only)
    return ai, seen


@pytest.mark.asyncio
async def test_typed_plugin_receives_typed_options(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.embed(options={'task_type': 'retrieval'}) arrives as MyOptions.task_type."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='conforming', content='hi', options={'task_type': 'retrieval'})
    assert isinstance(seen['options'], ConformingCfg)
    assert seen['options'].task_type == 'retrieval'


@pytest.mark.asyncio
async def test_matching_options_instance_passes_through(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.embed(options=MyOptions(...)) arrives as that same class."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='plugin_only', content='hi', options=PluginOnlyCfg(output_dimensionality=8))
    assert type(seen['options']) is PluginOnlyCfg
    assert seen['options'].output_dimensionality == 8


@pytest.mark.asyncio
async def test_plain_basemodel_options_accepted_at_embed(ai_and_seen: tuple[Genkit, dict]) -> None:
    """ai.embed(options=) accepts a dict or a BaseModel instance."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='conforming', content='hi', options=ConformingCfg(task_type='retrieval'))
    assert type(seen['options']) is ConformingCfg
    assert seen['options'].task_type == 'retrieval'


@pytest.mark.asyncio
async def test_typed_plugin_receives_plugin_only_fields_from_instance(
    ai_and_seen: tuple[Genkit, dict],
) -> None:
    """A field only the plugin schema owns still arrives, from an instance or a dict."""
    ai, seen = ai_and_seen
    cfg = PluginOnlyCfg(output_dimensionality=8)

    await ai.embed(embedder='plugin_only', content='hi', options=cfg)
    assert type(seen['options']) is PluginOnlyCfg
    assert seen['options'].output_dimensionality == 8

    await ai.embed(embedder='plugin_only', content='hi', options={'output_dimensionality': 8})
    assert type(seen['options']) is PluginOnlyCfg
    assert seen['options'].output_dimensionality == 8


@pytest.mark.asyncio
async def test_bare_plugin_receives_raw_dict(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A handler annotated EmbedRequest (no type param) still sees the raw dict."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='bare', content='hi', options={'task_type': 'retrieval', 'anything': 1})
    assert seen['options'] == {'task_type': 'retrieval', 'anything': 1}
    assert type(seen['options']) is dict


@pytest.mark.asyncio
async def test_omitted_options_yields_empty_typed_options(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Omitting options still gives the plugin an empty MyOptions, not None."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='conforming', content='hi')
    assert isinstance(seen['options'], ConformingCfg)
    assert seen['options'].task_type is None


@pytest.mark.asyncio
async def test_unknown_keys_reach_plugin_via_model_extra(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A key the plugin schema does not declare still reaches model_extra."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='conforming', content='hi', options={'thinking': {'budget': 8192}})
    assert seen['options'].model_extra == {'thinking': {'budget': 8192}}


@pytest.mark.asyncio
async def test_invalid_value_raises_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """A bad value on a declared field is GenkitError, not a raw ValidationError."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'conforming'"):
        await ai.embed(embedder='conforming', content='hi', options={'temperature': 'high'})


@pytest.mark.asyncio
async def test_strict_options_rejects_unknown_keys_as_genkit_error(ai_and_seen: tuple[Genkit, dict]) -> None:
    """extra='forbid' rejects unknown keys as GenkitError — the plugin opted in."""
    ai, _ = ai_and_seen
    with pytest.raises(GenkitError, match="Invalid input for action 'strict'"):
        await ai.embed(embedder='strict', content='hi', options={'thinking': True})


@pytest.mark.asyncio
async def test_foreign_options_class_normalizes_to_conforming(ai_and_seen: tuple[Genkit, dict]) -> None:
    """Without config_schema= on the action, a foreign instance dumps and coerces."""
    ai, seen = ai_and_seen
    await ai.embed(embedder='conforming', content='hi', options=OpenAIConfig(temperature=0.7))
    assert type(seen['options']) is ConformingCfg
