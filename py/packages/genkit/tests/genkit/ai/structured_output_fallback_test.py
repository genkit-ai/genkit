# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Structured output instructions for models without native constraints."""

from typing import Literal

import pytest

from genkit import Genkit, Message, ModelResponse, Part
from genkit._core._action import ActionRunContext
from genkit._core._model import ModelRequest
from genkit._core._typing import ModelInfo, Role


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'capability, with_tools, instructions, constrained, expected_instructions, expected_constrained',
    [
        ('none', False, None, None, True, False),
        ('none', True, None, None, True, False),
        ('no-tools', True, None, None, True, False),
        ('no-tools', False, None, None, False, True),
        ('all', True, None, None, False, True),
        (None, True, None, None, False, True),
        ('none', True, False, None, False, False),
        ('none', True, 'Return the requested JSON.', None, True, False),
        ('none', True, True, None, True, False),
        ('none', True, None, False, False, False),
    ],
)
async def test_generate_structured_output_fallback(
    capability: Literal['all', 'none', 'no-tools'] | None,
    with_tools: bool,
    instructions: bool | str | None,
    constrained: bool | None,
    expected_instructions: bool,
    expected_constrained: bool,
) -> None:
    """Only unsupported native constraints need automatic schema instructions."""
    ai = Genkit()
    requests: list[ModelRequest] = []

    async def model_fn(request: ModelRequest, ctx: ActionRunContext) -> ModelResponse:
        requests.append(request)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"value": 42}')]))

    ai.define_model(
        name='test-model',
        fn=model_fn,
        info=ModelInfo.model_validate({'supports': {'constrained': capability}}),
    )

    @ai.tool(name='lookup')
    async def lookup() -> str:
        """Look up data."""
        return 'data'

    response = await ai.generate(
        model='test-model',
        prompt='Give a value.',
        tools=['lookup'] if with_tools else None,
        output_schema={'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']},
        output_instructions=instructions,
        output_constrained=constrained,
    )

    assert response.output == {'value': 42}
    assert len(requests) == 1
    request = requests[0]
    output_parts = [
        part
        for message in request.messages
        for part in message.content
        if (part.metadata or {}).get('purpose') == 'output'
    ]
    assert len(output_parts) == int(expected_instructions)
    if output_parts:
        if isinstance(instructions, str):
            assert output_parts[0].text == instructions
        else:
            assert 'Output should be in JSON format and conform to the following schema' in (output_parts[0].text or '')
            assert 'value' in (output_parts[0].text or '')
    assert request.output_constrained is expected_constrained
    assert bool(request.tools) is with_tools
