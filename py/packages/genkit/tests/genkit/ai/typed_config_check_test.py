#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A model typed as ModelRequest[Cfg] checks call config against Cfg up front.

Without a class passed as ``config_schema=``, a value the annotation's class
can't take used to come back as an empty response with
``finish_reason='failed'``. It now raises INVALID_ARGUMENT before the model
runs, the same as when ``config_schema=`` is a class.
"""

from collections.abc import Awaitable, Callable
from typing import Literal

import pytest
from pydantic import BaseModel

from genkit import Genkit, Part
from genkit._core._action import ActionRunContext
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import Role

QUALITY_SCHEMA = {
    'type': 'object',
    'properties': {'quality': {'type': 'string', 'enum': ['low', 'high']}},
}


class SketchConfig(BaseModel):
    """Config the sketch model's function is typed with."""

    quality: Literal['low', 'high'] = 'low'


class OtherConfig(BaseModel):
    """A config class that belongs to some other model."""

    quality: str = 'low'


def sketch_model() -> tuple[
    Callable[[ModelRequest[SketchConfig], ActionRunContext], Awaitable[ModelResponse]], list[object]
]:
    """A typed model function, plus the configs it was called with."""
    configs: list[object] = []

    async def sketch(request: ModelRequest[SketchConfig], ctx: ActionRunContext) -> ModelResponse:
        configs.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('sketch')]))

    return sketch, configs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'define_kwargs',
    [
        pytest.param({}, id='no-config-schema'),
        pytest.param({'config_schema': QUALITY_SCHEMA}, id='dict-config-schema'),
    ],
)
async def test_generate_with_bad_config_raises_invalid_argument(define_kwargs: dict[str, object]) -> None:
    """A value the annotation's class rejects raises before the model runs."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch, **define_kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='acme/sketch', prompt='a lighthouse', config={'quality': 'ultra'})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert "config 'quality'" in str(exc_info.value)
    assert configs == []


@pytest.mark.asyncio
async def test_generate_with_good_config_reaches_model_as_annotation_class() -> None:
    """Valid config is still coerced into the annotation's class."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch, config_schema=QUALITY_SCHEMA)

    response = await ai.generate(model='acme/sketch', prompt='a lighthouse', config={'quality': 'high'})

    assert response.text == 'sketch'
    assert configs == [SketchConfig(quality='high')]


@pytest.mark.asyncio
async def test_generate_with_other_models_config_object_converts_when_values_fit() -> None:
    """Another class's config object converts into the annotation's class when its values fit."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch)

    await ai.generate(model='acme/sketch', prompt='a lighthouse', config=OtherConfig(quality='high'))

    assert configs == [SketchConfig(quality='high')]


@pytest.mark.asyncio
async def test_generate_with_other_models_config_object_raises_when_values_dont_fit() -> None:
    """Another class's config object with a value the annotation's class rejects raises up front."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='acme/sketch', prompt='a lighthouse', config=OtherConfig(quality='ultra'))

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert "config 'quality'" in str(exc_info.value)
    assert configs == []


@pytest.mark.asyncio
async def test_generate_stream_with_bad_config_raises_invalid_argument() -> None:
    """generate_stream fails the same way as generate."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch)

    stream = ai.generate_stream(model='acme/sketch', prompt='a lighthouse', config={'quality': 'ultra'})
    with pytest.raises(GenkitError) as exc_info:
        await stream.response

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert configs == []


@pytest.mark.asyncio
async def test_prompt_with_bad_call_config_raises_invalid_argument() -> None:
    """A prompt call's config override is checked against the annotation's class too."""
    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch)
    lighthouse = ai.define_prompt(model='acme/sketch', prompt='a lighthouse')

    with pytest.raises(GenkitError) as exc_info:
        await lighthouse(config={'quality': 'ultra'})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert configs == []


@pytest.mark.asyncio
async def test_config_schema_class_still_wins_over_annotation() -> None:
    """With both, the call is checked against the class passed as config_schema=."""

    class StrictSketchConfig(BaseModel):
        quality: Literal['high'] = 'high'

    ai = Genkit()
    sketch, configs = sketch_model()
    ai.define_model('acme/sketch', sketch, config_schema=StrictSketchConfig)

    with pytest.raises(GenkitError) as exc_info:
        await ai.generate(model='acme/sketch', prompt='a lighthouse', config={'quality': 'low'})

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert configs == []
