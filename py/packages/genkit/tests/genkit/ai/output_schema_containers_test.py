#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for container types and non-BaseModel schemas in output_schema."""

from enum import Enum

import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse
from genkit._ai._testing import ProgrammableModel, define_programmable_model
from genkit._core._typing import FinishReason, Part, Role, TextPart


class MenuItem(BaseModel):
    name: str
    price: float


class StatusEnum(str, Enum):
    ACTIVE = 'active'
    INACTIVE = 'inactive'


@pytest.fixture
def setup_test() -> tuple[Genkit, ProgrammableModel]:
    """Setup test Genkit instance with programmable model."""
    ai = Genkit(model='programmableModel')
    pm, _ = define_programmable_model(ai, name='programmableModel')
    return (ai, pm)


@pytest.mark.asyncio
async def test_output_schema_list_of_models(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Verify output_schema=list[MenuItem] parses and validates into list of Pydantic models."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[
                    Part(TextPart(text='[{"name": "Espresso", "price": 3.50}, {"name": "Croissant", "price": 4.25}]'))
                ],
            ),
        )
    )

    response = await ai.generate(
        prompt='Give me a menu',
        output_schema=list[MenuItem],
    )

    assert response.finish_reason == FinishReason.STOP
    assert isinstance(response.output, list)
    assert len(response.output) == 2
    assert isinstance(response.output[0], MenuItem)
    assert response.output[0].name == 'Espresso'
    assert response.output[0].price == 3.50
    assert isinstance(response.output[1], MenuItem)
    assert response.output[1].name == 'Croissant'
    assert response.output[1].price == 4.25


@pytest.mark.asyncio
async def test_output_schema_list_of_primitives(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Verify output_schema=list[str] parses and validates string items."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(TextPart(text='["python", "typescript", "go"]'))],
            ),
        )
    )

    response = await ai.generate(
        prompt='List languages',
        output_schema=list[str],
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.output == ['python', 'typescript', 'go']


@pytest.mark.asyncio
async def test_output_schema_enum(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Verify output_schema=StatusEnum parses and validates enum value."""
    ai, pm = setup_test

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(TextPart(text='"active"'))],
            ),
        )
    )

    response = await ai.generate(
        prompt='Get status',
        output_schema=StatusEnum,
    )

    assert response.finish_reason == FinishReason.STOP
    assert response.output == StatusEnum.ACTIVE


@pytest.mark.asyncio
async def test_output_schema_list_validation_failure(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Verify invalid items in list[MenuItem] mark response as FAILED."""
    ai, pm = setup_test

    # Price is not a valid float
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(TextPart(text='[{"name": "Espresso", "price": "not_a_number"}]'))],
            ),
        )
    )

    response = await ai.generate(
        prompt='Give me a menu',
        output_schema=list[MenuItem],
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.output is None


@pytest.mark.asyncio
async def test_prompt_with_container_output_schema(
    setup_test: tuple[Genkit, ProgrammableModel],
) -> None:
    """Verify define_prompt works seamlessly with container output_schema."""
    ai, pm = setup_test

    prompt = ai.define_prompt(
        name='menuPrompt',
        prompt='List menu items',
        output_schema=list[MenuItem],
    )

    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(
                role=Role.MODEL,
                content=[Part(TextPart(text='[{"name": "Tea", "price": 2.50}]'))],
            ),
        )
    )

    response = await prompt()
    assert response.finish_reason == FinishReason.STOP
    assert isinstance(response.output, list)
    assert len(response.output) == 1
    assert isinstance(response.output[0], MenuItem)
    assert response.output[0].name == 'Tea'
