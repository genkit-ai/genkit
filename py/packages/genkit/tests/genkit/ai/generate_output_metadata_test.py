#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Output-format annotations do not mutate model-owned metadata."""

import copy

import pytest

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit.model import ModelContext, ModelRequest


def cached_response_model() -> tuple[Genkit, ModelResponse]:
    ai = Genkit()
    response = ModelResponse(
        message=Message(
            role=Role.MODEL,
            content=[Part.from_text('{"answer":"yes"}')],
            metadata={'generate': {'custom': 'keep'}, 'source': 'fixture'},
        )
    )

    async def model(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return response

    ai.define_model(name='cachedResponse', fn=model)
    return ai, response


@pytest.mark.asyncio
@pytest.mark.parametrize('output_format', ['json', 'text'])
async def test_output_format_preserves_model_metadata(output_format: str) -> None:
    ai, shared = cached_response_model()
    assert shared.message is not None
    original_metadata = copy.deepcopy(shared.message.metadata)

    result = await ai.generate(model='cachedResponse', prompt='answer', output_format=output_format)

    assert result.message is not None
    assert result.message.metadata is not None
    assert result.message.metadata['generate']['output']['format'] == output_format
    assert result.message.metadata['generate']['custom'] == 'keep'
    assert shared.message.metadata == original_metadata


@pytest.mark.asyncio
async def test_later_generation_does_not_change_an_earlier_output_format() -> None:
    ai, _ = cached_response_model()
    first = await ai.generate(model='cachedResponse', prompt='answer', output_format='json')
    assert first.message is not None
    first_metadata = copy.deepcopy(first.message.metadata)

    second = await ai.generate(model='cachedResponse', prompt='answer again', output_format='text')

    assert second.message is not None
    assert second.message.metadata is not None
    assert second.message.metadata['generate']['output']['format'] == 'text'
    assert first.message.metadata == first_metadata
