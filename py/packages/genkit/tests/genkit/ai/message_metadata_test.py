#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Message metadata survives output-instruction injection through generate."""

import pytest

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit.testing import define_scripted_model


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['system', 'user', 'pending'])
async def test_generate_preserves_metadata_when_injecting_instructions(target: str) -> None:
    """The model sees caller metadata for all three output-instruction targets."""
    ai = Genkit()
    model, _ = define_scripted_model(
        ai,
        responses=[ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('yes')]))],
    )
    metadata = {'request_id': target}
    chosen = Message(role=Role.USER, content=[Part.from_text('Choose yes or no.')], metadata=metadata)
    messages = [chosen]
    target_index = 0
    if target == 'system':
        chosen = Message(role=Role.SYSTEM, content=[Part.from_text('Answer the question.')], metadata=metadata)
        messages = [chosen, Message(role=Role.USER, content=[Part.from_text('Choose yes or no.')])]
    elif target == 'pending':
        chosen = Message(
            role=Role.USER,
            content=[Part.from_text('pending', metadata={'purpose': 'output', 'pending': True})],
            metadata=metadata,
        )
        messages = [Message(role=Role.SYSTEM, content=[Part.from_text('Answer the question.')]), chosen]
        target_index = 1
    original_messages = [message.model_dump() for message in messages]

    response = await ai.generate(
        model='scriptedModel',
        messages=messages,
        output_format='enum',
        output_schema={'type': 'string', 'enum': ['yes', 'no']},
    )

    assert response.output == 'yes'
    request = model.last_request
    assert request is not None
    delivered = request.messages[target_index]
    assert delivered.metadata == metadata
    assert any(
        part.metadata and part.metadata.get('purpose') == 'output' and not part.metadata.get('pending')
        for part in delivered.content
    )
    assert [message.model_dump() for message in messages] == original_messages
