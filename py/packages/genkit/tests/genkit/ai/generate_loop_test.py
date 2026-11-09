#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Turn-loop regressions: usage totals, max_turns validation, and friends."""

import pytest

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._typing import FinishReason, GenerationUsage, Role, ToolRequest
from genkit.testing import define_scripted_model


def _calls_tool(*, ref: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='lookup', input={}, ref=ref))]),
        usage=usage,
    )


def _answers(text: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
        usage=usage,
    )


@pytest.mark.asyncio
async def test_total_usage_sums_every_turn_of_a_tool_loop() -> None:
    """usage is the last call; total_usage is what the whole loop billed."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'in stock'

    pm.responses = [
        _calls_tool(
            ref='1',
            usage=GenerationUsage(input_tokens=10, output_tokens=5, thoughts_tokens=3, custom={'calls': 1, 'id': 'a'}),
        ),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=20, output_tokens=6, custom={'calls': 1, 'id': 'b'})),
        _answers('done', usage=GenerationUsage(input_tokens=30, output_tokens=7)),
    ]

    response = await ai.generate(prompt='check stock', tools=['lookup'])

    assert response.text == 'done'
    assert response.usage is not None
    assert response.usage.input_tokens == 30
    assert response.total_usage == GenerationUsage(
        input_tokens=60,
        output_tokens=18,
        thoughts_tokens=3,
        custom={'calls': 2, 'id': 'b'},
    )


@pytest.mark.asyncio
async def test_total_usage_matches_usage_on_a_single_turn() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi', usage=GenerationUsage(input_tokens=4, output_tokens=2))]

    response = await ai.generate(prompt='hello')

    assert response.total_usage == response.usage


@pytest.mark.asyncio
async def test_total_usage_survives_a_max_turns_abort() -> None:
    """The rounds billed before the cap still show up on the aborted response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'again'

    pm.responses = [
        _calls_tool(ref='1', usage=GenerationUsage(input_tokens=10)),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=11)),
    ]

    response = await ai.generate(prompt='loop', tools=['lookup'], max_turns=1)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.total_usage is not None
    assert response.total_usage.input_tokens == 21


@pytest.mark.asyncio
async def test_total_usage_stays_off_when_no_model_reported_usage() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi')]

    response = await ai.generate(prompt='hello')

    assert response.total_usage is None
