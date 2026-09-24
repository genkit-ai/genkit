# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""L1 pins: a finished ``ai.generate(..., use=[Surfaces()])`` returns data parts, not fences."""

from __future__ import annotations

import json
from typing import Any

import pytest
from genkit_a2ui import A2uiParseError, Surfaces
from helpers import (
    WEATHER_PROMPT,
    a2ui_parts,
    assert_finished_message,
    assert_no_a2ui_parts,
    assert_no_fence_in_text,
    bad_component_fence,
    broken_fence,
    create_surface_ids,
    envelopes,
    fence_only,
    joined_text,
    model_ok,
    no_root_fence,
    request_system_text,
    setup,
    text_part,
    weather_fence,
)
from pydantic import ValidationError

from genkit import Message, ModelResponse, ModelResponseChunk
from genkit._core._typing import Candidate, FinishReason, Role


@pytest.mark.asyncio
async def test_generate_a2ui_rewrites_fence_to_data_part() -> None:
    """A finished weather turn returns a data part; the fence is gone from text."""
    ai, pm = setup()
    fence = weather_fence()
    pm.responses = [model_ok(fence)]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert envelopes(message.content)
    assert create_surface_ids(message.content)
    assert 'Here is the weather:' in joined_text(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_leaves_plain_prose_untouched() -> None:
    """A turn with no A2UI fence stays plain text and grows no data part."""
    ai, pm = setup()
    pm.responses = [model_ok('just chatting')]

    response = await ai.generate(model='programmableModel', prompt='hi', use=[Surfaces()])
    message = assert_finished_message(response)
    assert_no_a2ui_parts(message.content)
    assert joined_text(message.content) == 'just chatting'
    assert response.text == 'just chatting'


@pytest.mark.asyncio
async def test_generate_a2ui_fence_only_returns_data_part() -> None:
    """A fence with no surrounding prose still becomes a data part."""
    ai, pm = setup()
    pm.responses = [model_ok(fence_only())]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert envelopes(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_stitches_fence_split_across_parts() -> None:
    """A fence split across several final text parts becomes one data part."""
    ai, pm = setup()
    fence = weather_fence()
    mid = len(fence) // 2
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[text_part(fence[:mid]), text_part(fence[mid:])],
            ),
        )
    ]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert len(a2ui_parts(message.content)) == 1
    assert envelopes(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_keeps_prose_on_both_sides_of_part() -> None:
    """Prose before and after the fence stays in order around the data part."""
    ai, pm = setup()
    body = weather_fence().replace('Here is the weather:\n', '')
    pm.responses = [model_ok(f'intro\n{body}outro')]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    content = message.content
    assert len(content) == 3
    assert 'intro' in joined_text([content[0]])
    assert a2ui_parts([content[1]])
    assert 'outro' in joined_text([content[2]])
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_stream_and_final_share_surface_id() -> None:
    """Streamed chunks and the final message mint the same surface id."""
    ai, pm = setup()
    fence = weather_fence()
    pm.responses = [model_ok(fence)]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(fence)])]]

    stream = ai.generate_stream(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    streamed: list[str] = []
    async for chunk in stream.stream:
        streamed.extend(create_surface_ids(chunk.content))
    response = await stream.response
    message = assert_finished_message(response)
    final_ids = create_surface_ids(message.content)
    assert streamed
    assert final_ids
    assert streamed[0] == final_ids[0]


@pytest.mark.asyncio
async def test_generate_a2ui_flushes_open_fence_at_end_of_turn() -> None:
    """An unterminated fence still reaches the stream when the turn ends."""
    ai, pm = setup()
    unterminated = '```a2ui\n' + json.dumps([
        {
            'updateComponents': {
                'surfaceId': 'SURFACE_ID',
                'components': [{'id': 'root', 'component': 'Text', 'text': 'hi'}],
            }
        }
    ])
    pm.responses = [model_ok(unterminated)]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(unterminated)])]]

    stream = ai.generate_stream(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(surface_id='sfc')],
    )
    streamed_envs: list[dict[str, Any]] = []
    async for chunk in stream.stream:
        streamed_envs.extend(envelopes(chunk.content))
    response = await stream.response
    assert_finished_message(response)
    assert any('updateComponents' in env for env in streamed_envs)


@pytest.mark.asyncio
async def test_generate_a2ui_flushes_withheld_prose_at_end_of_turn() -> None:
    """Trailing prose held back for a possible fence still reaches the stream."""
    ai, pm = setup()
    full = 'Hello there, friend!'
    pm.responses = [model_ok(full)]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(ch)]) for ch in full]]

    stream = ai.generate_stream(model='programmableModel', prompt='hi', use=[Surfaces()])
    streamed = ''
    async for chunk in stream.stream:
        streamed += joined_text(chunk.content)
    response = await stream.response
    assert_finished_message(response)
    assert streamed == full
    assert response.text == full


@pytest.mark.asyncio
async def test_generate_a2ui_drops_bad_block_and_keeps_prose() -> None:
    """Default warn drops an unknown component and keeps the surrounding prose."""
    ai, pm = setup()
    pm.responses = [model_ok(bad_component_fence())]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert_no_a2ui_parts(message.content)
    assert_no_fence_in_text(message.content)
    assert 'oops' in joined_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_strict_raises_on_bad_block() -> None:
    """Strict mode fails the generate call when the fence names an unknown component."""
    ai, pm = setup()
    pm.responses = [model_ok(bad_component_fence())]

    with pytest.raises(A2uiParseError):
        await ai.generate(
            model='programmableModel',
            prompt=WEATHER_PROMPT,
            use=[Surfaces(validate='strict')],
        )


@pytest.mark.asyncio
async def test_generate_stream_strict_raises_on_bad_block() -> None:
    """Strict mode fails generate_stream with A2uiParseError when the fence names an unknown component."""
    ai, pm = setup()
    fence = bad_component_fence()
    pm.responses = [model_ok(fence)]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(fence)])]]

    stream = ai.generate_stream(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(validate='strict')],
    )
    with pytest.raises(A2uiParseError, match='NotAThing'):
        async for _chunk in stream.stream:
            pass
        await stream.response


@pytest.mark.asyncio
async def test_generate_a2ui_skips_rewrite_when_blocked_or_aborted() -> None:
    """A stopped turn keeps the finish reason and does not salvage a card."""
    for reason in (
        FinishReason.BLOCKED,
        FinishReason.ABORTED,
        FinishReason.INTERRUPTED,
        FinishReason.FAILED,
        FinishReason.UNKNOWN,
    ):
        ai, pm = setup()
        raw = broken_fence()
        pm.responses = [
            ModelResponse(
                finish_reason=reason,
                finish_message='safety',
                message=Message(role=Role.MODEL, content=[text_part(raw)]),
            )
        ]

        response = await ai.generate(
            model='programmableModel',
            prompt=WEATHER_PROMPT,
            use=[Surfaces(validate='strict')],
        )
        message = assert_finished_message(response, finish_reason=reason)
        assert response.finish_message == 'safety'
        assert_no_a2ui_parts(message.content)
        assert '```a2ui' in joined_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_other_keeps_raw_fence() -> None:
    """A turn that finished other keeps the finish reason and does not salvage a card."""
    ai, pm = setup()
    raw = broken_fence()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.OTHER,
            finish_message='provider other',
            message=Message(role=Role.MODEL, content=[text_part(raw)]),
        )
    ]

    response = await ai.generate(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(validate='strict')],
    )
    message = assert_finished_message(response, finish_reason=FinishReason.OTHER)
    assert response.finish_message == 'provider other'
    assert_no_a2ui_parts(message.content)
    assert '```a2ui' in joined_text(message.content)


@pytest.mark.asyncio
async def test_generate_stream_blocked_paints_card_but_response_keeps_fence() -> None:
    """A blocked stream may already have painted a card; the response they persist still has the fence."""
    ai, pm = setup()
    fence = weather_fence()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.BLOCKED,
            finish_message='stop',
            message=Message(role=Role.MODEL, content=[text_part(fence)]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(fence)])]]

    stream = ai.generate_stream(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(validate='strict')],
    )
    streamed_envs: list[dict[str, Any]] = []
    async for chunk in stream.stream:
        streamed_envs.extend(envelopes(chunk.content))
    response = await stream.response
    message = assert_finished_message(response, finish_reason=FinishReason.BLOCKED)
    assert response.finish_message == 'stop'
    assert streamed_envs
    assert_no_a2ui_parts(message.content)
    assert '```a2ui' in joined_text(message.content)


@pytest.mark.asyncio
async def test_generate_stream_other_paints_card_but_response_keeps_fence() -> None:
    """A stream that finished other may already have painted a card; the response they persist still has the fence."""
    ai, pm = setup()
    fence = weather_fence()
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.OTHER,
            finish_message='stop',
            message=Message(role=Role.MODEL, content=[text_part(fence)]),
        )
    ]
    pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[text_part(fence)])]]

    stream = ai.generate_stream(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(validate='strict')],
    )
    streamed_envs: list[dict[str, Any]] = []
    async for chunk in stream.stream:
        streamed_envs.extend(envelopes(chunk.content))
    response = await stream.response
    message = assert_finished_message(response, finish_reason=FinishReason.OTHER)
    assert response.finish_message == 'stop'
    assert streamed_envs
    assert_no_a2ui_parts(message.content)
    assert '```a2ui' in joined_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_mints_new_surface_id() -> None:
    """A new createSurface does not keep the model's SURFACE_ID placeholder."""
    ai, pm = setup()
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    ids = create_surface_ids(message.content)
    assert ids
    assert 'SURFACE_ID' not in ids


@pytest.mark.asyncio
async def test_generate_a2ui_injects_catalog_instructions_by_default() -> None:
    """Default Surfaces() tells the model how to emit an A2UI fence."""
    ai, pm = setup()
    pm.responses = [model_ok('ok')]

    await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    assert 'Rendering UI with A2UI' in request_system_text(pm)


@pytest.mark.asyncio
async def test_generate_a2ui_instructions_none_does_not_inject_catalog() -> None:
    """Surfaces(instructions='none') leaves the system prompt without catalog instructions."""
    ai, pm = setup()
    pm.responses = [model_ok('ok')]

    await ai.generate(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(instructions='none')],
    )
    assert 'Rendering UI with A2UI' not in request_system_text(pm)


@pytest.mark.asyncio
async def test_generate_a2ui_rewrites_candidates_message_too() -> None:
    """A finished turn rewrites candidates[0].message, not only the top-level message."""
    ai, pm = setup()
    fence = weather_fence()
    message = Message(role=Role.MODEL, content=[text_part(fence)])
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=message,
            candidates=[
                Candidate(index=0, message=message, finish_reason=FinishReason.STOP),
            ],
        )
    ]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    finished = assert_finished_message(response)
    assert envelopes(finished.content)
    assert_no_fence_in_text(finished.content)
    assert response.candidates
    candidate_message = response.candidates[0].message
    assert envelopes(candidate_message.content)
    assert_no_fence_in_text(candidate_message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_rewrites_candidates_when_message_missing() -> None:
    """A finished turn that only carries candidates[0] still becomes a data part."""
    ai, pm = setup()
    fence = weather_fence()
    message = Message(role=Role.MODEL, content=[text_part(fence)])
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=None,
            candidates=[
                Candidate(index=0, message=message, finish_reason=FinishReason.STOP),
            ],
        )
    ]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    finished = assert_finished_message(response)
    assert envelopes(finished.content)
    assert_no_fence_in_text(finished.content)
    assert response.candidates
    candidate_message = response.candidates[0].message
    assert envelopes(candidate_message.content)
    assert_no_fence_in_text(candidate_message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_off_keeps_surface_without_root() -> None:
    """validate='off' still returns the surface when the component list has no root."""
    ai, pm = setup()
    pm.responses = [model_ok(no_root_fence())]

    response = await ai.generate(
        model='programmableModel',
        prompt=WEATHER_PROMPT,
        use=[Surfaces(validate='off')],
    )
    message = assert_finished_message(response)
    assert envelopes(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_warn_drops_surface_without_root() -> None:
    """Default warn drops a surface that has no root and keeps no fence."""
    ai, pm = setup()
    pm.responses = [model_ok(no_root_fence())]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert_no_a2ui_parts(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_empty_text_part_mid_fence_still_returns_a_card() -> None:
    """An empty text part in the middle of a fence does not drop the card."""
    ai, pm = setup()
    fence = weather_fence()
    mid = len(fence) // 2
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[text_part(fence[:mid]), text_part(''), text_part(fence[mid:])],
            ),
        )
    ]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response)
    assert envelopes(message.content)
    assert_no_fence_in_text(message.content)


@pytest.mark.asyncio
async def test_generate_a2ui_truncated_fence_on_length_stays_as_prose() -> None:
    """A length-stopped turn keeps the unfinished fence as text instead of dropping it."""
    ai, pm = setup()
    raw = 'Here is the weather:\n```a2ui\n[{"createSurface":'
    pm.responses = [
        ModelResponse(
            finish_reason=FinishReason.LENGTH,
            finish_message='max tokens',
            message=Message(role=Role.MODEL, content=[text_part(raw)]),
        )
    ]

    response = await ai.generate(model='programmableModel', prompt=WEATHER_PROMPT, use=[Surfaces()])
    message = assert_finished_message(response, finish_reason=FinishReason.LENGTH)
    assert response.finish_message == 'max tokens'
    assert_no_a2ui_parts(message.content)
    assert '```a2ui' in joined_text(message.content)
    assert 'createSurface' in joined_text(message.content)


def test_a2ui_catalog_string_is_the_lookup_id() -> None:
    """catalog= is the registry id generate and the Developer UI look up."""
    assert Surfaces(catalog='basic').config.catalog == 'basic'


def test_a2ui_accepts_camel_surface_id() -> None:
    """surfaceId= is the same knob as surface_id=."""
    assert Surfaces(surfaceId='sfc').config.surface_id == 'sfc'


def test_a2ui_rejects_unknown_version() -> None:
    """A typo version is rejected so it cannot stamp envelopes the renderer will drop."""
    with pytest.raises(ValidationError):
        Surfaces(version='v9')
