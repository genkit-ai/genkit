#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""to_sse unboxes a FAILED ModelResponse whatever handle carried it."""

import json
from collections.abc import AsyncIterator

import pytest

from genkit import Genkit, ModelResponse
from genkit._core._typing import FinishReason
from genkit.testing import define_scripted_model
from genkit.web import to_sse


async def _frames(handle: object) -> list[dict[str, object]]:
    return [json.loads(line.removeprefix('data: ')) async for line in to_sse(handle)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_flow_stream_returning_a_failed_response_sends_an_error_frame() -> None:
    """A flow that returns the whole ModelResponse no longer serves {"result": ...} on failure."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = []  # The scripted model raises when it runs out of responses.

    @ai.flow()
    async def suggest_dish(order: str) -> ModelResponse:
        return await ai.generate(prompt=order)

    frames = await _frames(suggest_dish.stream('tartine'))

    assert len(frames) == 1
    assert 'error' in frames[0]


class _Handle:
    def __init__(self, result: object) -> None:
        self._result = result

    @property
    def stream(self) -> AsyncIterator[object]:
        async def chunks() -> AsyncIterator[object]:
            yield 'chunk'

        return chunks()

    @property
    def response(self) -> object:
        async def result() -> object:
            return self._result

        return result()


@pytest.mark.asyncio
async def test_any_handle_with_a_failed_response_ends_in_an_error_frame() -> None:
    failed = ModelResponse(finish_reason=FinishReason.FAILED, finish_message='quota exceeded')

    frames = await _frames(_Handle(failed))

    assert frames[0] == {'message': 'chunk'}
    assert 'error' in frames[-1]
    assert 'result' not in frames[-1]


@pytest.mark.asyncio
async def test_successful_response_still_sends_the_result() -> None:
    ok = ModelResponse(finish_reason=FinishReason.STOP)

    frames = await _frames(_Handle(ok))

    assert frames[-1]['result']['finishReason'] == 'stop'  # type: ignore[index]
