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

"""L2 pins: the next model request sees text, never ``application/a2ui+json``."""

from __future__ import annotations

import pytest
from genkit_a2ui import A2UI_MIME_TYPE, Surfaces
from helpers import (
    A2UI_FENCE,
    BASIC_CATALOG_ID,
    WEATHER_PROMPT,
    a2ui_data_part,
    assert_finished_message,
    create_surface_ids,
    joined_text,
    model_ok,
    request_has_a2ui_part,
    request_history_messages,
    request_joined_text,
    setup,
    text_part,
    weather_fence,
)

from genkit import Message, Part
from genkit._core._typing import DataPart, Role


@pytest.mark.asyncio
async def test_generate_a2ui_sends_click_to_model_as_text() -> None:
    """A button click reaches the model as text, not as an A2UI data part."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.USER,
                content=[
                    text_part('User interacted with the UI…'),
                    a2ui_data_part([
                        {
                            'action': {
                                'name': 'refresh',
                                'surfaceId': 's1',
                                'sourceComponentId': 'btn',
                                'timestamp': 't',
                                'context': {'city': 'Tokyo'},
                            }
                        }
                    ]),
                ],
            )
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    joined = request_joined_text(pm)
    assert 'UI action "refresh"' in joined
    assert 's1' in joined
    assert 'Tokyo' in joined
    for message in request_history_messages(pm):
        assert message.content


@pytest.mark.asyncio
async def test_generate_a2ui_replays_prior_surface_as_fence() -> None:
    """A prior surface is replayed as an a2ui fence with the same surface id."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.MODEL,
                content=[
                    text_part('Here you go:'),
                    a2ui_data_part([
                        {
                            'version': 'v0.9',
                            'createSurface': {
                                'surfaceId': 's1',
                                'catalogId': BASIC_CATALOG_ID,
                            },
                        },
                        {
                            'version': 'v0.9',
                            'updateComponents': {
                                'surfaceId': 's1',
                                'components': [{'id': 'root', 'component': 'Text', 'text': 'hi'}],
                            },
                        },
                    ]),
                ],
            ),
            Message(role=Role.USER, content=[text_part('thanks')]),
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    joined = request_joined_text(pm)
    assert A2UI_FENCE in joined
    assert 'createSurface' in joined
    assert 'Here you go:' in joined
    assert '[rendered UI surface]' not in joined
    assert 's1' in joined
    for message in request_history_messages(pm):
        assert message.content


@pytest.mark.asyncio
async def test_generate_a2ui_splits_replayed_surfaces_around_a_click() -> None:
    """Surfaces stay one fence until a click; the click is its own text line."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.USER,
                content=[
                    a2ui_data_part([
                        {'createSurface': {'surfaceId': 's1', 'catalogId': BASIC_CATALOG_ID}},
                        {
                            'updateComponents': {
                                'surfaceId': 's1',
                                'components': [{'id': 'root', 'component': 'Text', 'text': 'one'}],
                            }
                        },
                        {
                            'action': {
                                'name': 'refresh',
                                'surfaceId': 's1',
                                'sourceComponentId': 'btn',
                                'timestamp': 't',
                            }
                        },
                        {'createSurface': {'surfaceId': 's2', 'catalogId': BASIC_CATALOG_ID}},
                        {
                            'updateComponents': {
                                'surfaceId': 's2',
                                'components': [{'id': 'root', 'component': 'Text', 'text': 'two'}],
                            }
                        },
                    ])
                ],
            )
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    joined = request_joined_text(pm)
    assert joined.count(A2UI_FENCE) == 2
    assert 'UI action "refresh"' in joined
    assert joined.index(A2UI_FENCE) < joined.index('UI action')
    for message in request_history_messages(pm):
        assert message.content


@pytest.mark.asyncio
async def test_generate_a2ui_empty_ui_message_keeps_placeholder_before_model() -> None:
    """A message that is only an empty A2UI part is preserved with a [UI] placeholder."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(role=Role.MODEL, content=[a2ui_data_part([])]),
            Message(role=Role.USER, content=[text_part('hi')]),
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    messages = request_history_messages(pm)
    assert all(message.content for message in messages)
    assert any(joined_text(message.content) == '[UI]' for message in messages)
    assert any(joined_text(message.content) == 'hi' for message in messages)
    assert len([m for m in messages if m.role == Role.MODEL]) == 1


@pytest.mark.asyncio
async def test_generate_a2ui_keeps_neighbor_text_when_rewriting_ui() -> None:
    """Text on the same message as an A2UI part is kept when the part is rewritten."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.MODEL,
                content=[
                    text_part('Here you go:'),
                    a2ui_data_part([
                        {'createSurface': {'surfaceId': 's1', 'catalogId': BASIC_CATALOG_ID}},
                    ]),
                ],
            ),
            Message(role=Role.USER, content=[text_part('thanks')]),
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    joined = request_joined_text(pm)
    assert 'Here you go:' in joined
    assert A2UI_FENCE in joined
    for message in request_history_messages(pm):
        assert message.content


@pytest.mark.asyncio
async def test_generate_a2ui_new_surface_does_not_reuse_history_id() -> None:
    """A new card after a prior surface s1 does not reuse s1."""
    ai, pm = setup()
    pm.responses = [model_ok(weather_fence())]

    response = await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.MODEL,
                content=[
                    a2ui_data_part([
                        {
                            'createSurface': {
                                'surfaceId': 's1',
                                'catalogId': BASIC_CATALOG_ID,
                            }
                        }
                    ])
                ],
            ),
            Message(role=Role.USER, content=[text_part(WEATHER_PROMPT)]),
        ],
        use=[Surfaces(surface_id='sfc-new')],
    )
    message = assert_finished_message(response)
    ids = create_surface_ids(message.content)
    assert ids == ['sfc-new']
    assert 's1' not in ids


@pytest.mark.asyncio
async def test_generate_a2ui_drops_bare_a2ui_mime_part() -> None:
    """A data part with the A2UI mime type but no envelopes is not sent to the model."""
    ai, pm = setup()
    pm.responses = [model_ok()]

    await ai.generate(
        model='programmableModel',
        messages=[
            Message(
                role=Role.USER,
                content=[
                    text_part('hi'),
                    Part(DataPart(data={'garbage': True}, metadata={'mimeType': A2UI_MIME_TYPE})),
                ],
            )
        ],
        use=[Surfaces()],
    )

    assert not request_has_a2ui_part(pm)
    assert 'hi' in request_joined_text(pm)
    for message in request_history_messages(pm):
        assert message.content
