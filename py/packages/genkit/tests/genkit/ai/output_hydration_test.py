#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""output_schema hydrates every type pydantic validates, not just BaseModel.

``ai.generate(output_schema=T)`` types ``.output`` as ``T``; these tests pin
that runtime returns a ``T`` too, for responses, stream chunks, and prompts.
"""

import dataclasses
import datetime
import enum
import uuid
from decimal import Decimal
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import TypedDict

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part
from genkit._core._partial import construct_partial, hydrate_partial
from genkit._core._typing import Role
from genkit.testing import define_scripted_model


class InvoiceItem(BaseModel):
    sku: str
    quantity: int


class InvoiceList(RootModel[list[InvoiceItem]]):
    pass


@dataclasses.dataclass
class Vendor:
    vendor_name: str
    risk_score: int


@pydantic_dataclass
class PydanticVendor:
    vendor_name: str
    risk_score: int


class VendorDict(TypedDict):
    vendor_name: str
    risk_score: int


class Triage(enum.Enum):
    APPROVE = 'APPROVE'
    ESCALATE = 'ESCALATE'


class TriageStr(str, enum.Enum):  # noqa: UP042  # enum.StrEnum is 3.11+
    APPROVE = 'APPROVE'
    ESCALATE = 'ESCALATE'


def _ai_with_reply(text: str, chunks: list[str] | None = None) -> Genkit:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))]
    if chunks is not None:
        pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(c)]) for c in chunks]]
    return ai


VENDOR_JSON = '{"vendor_name": "Acme", "risk_score": 12}'
UID = '550e8400-e29b-41d4-a716-446655440000'

CASES: list[tuple[str, Any, str, Any]] = [
    ('list_of_models', list[InvoiceItem], '[{"sku": "CH-1", "quantity": 2}]', [InvoiceItem(sku='CH-1', quantity=2)]),
    ('root_model', InvoiceList, '[{"sku": "CH-1", "quantity": 2}]', InvoiceList([InvoiceItem(sku='CH-1', quantity=2)])),
    ('dict', dict[str, int], '{"a": 1}', {'a': 1}),
    ('literal', Literal['APPROVE', 'ESCALATE'], '"ESCALATE"', 'ESCALATE'),
    ('annotated', Annotated[int, Field(gt=0)], '7', 7),
    ('dataclass', Vendor, VENDOR_JSON, Vendor(vendor_name='Acme', risk_score=12)),
    ('pydantic_dataclass', PydanticVendor, VENDOR_JSON, PydanticVendor(vendor_name='Acme', risk_score=12)),
    ('typed_dict', VendorDict, VENDOR_JSON, {'vendor_name': 'Acme', 'risk_score': 12}),
    ('enum', Triage, '"ESCALATE"', Triage.ESCALATE),
    ('str_enum', TriageStr, '"ESCALATE"', TriageStr.ESCALATE),
    ('decimal', Decimal, '14250.75', Decimal('14250.75')),
    ('uuid', uuid.UUID, f'"{UID}"', uuid.UUID(UID)),
    (
        'datetime',
        datetime.datetime,
        '"2026-10-15T14:30:00Z"',
        datetime.datetime(2026, 10, 15, 14, 30, tzinfo=datetime.timezone.utc),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('schema', 'reply', 'expected'), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
async def test_generate_hydrates_output_schema(schema: Any, reply: str, expected: Any) -> None:  # noqa: ANN401
    response = await _ai_with_reply(reply).generate(prompt='hi', output_schema=schema)

    assert response.error is None
    assert response.output == expected
    assert type(response.output) is type(expected)


@pytest.mark.asyncio
async def test_decimal_keeps_its_lexeme() -> None:
    response = await _ai_with_reply('{"amount": 0.1}').generate(prompt='hi', output_schema=dict[str, Decimal])

    assert response.output == {'amount': Decimal('0.1')}


@pytest.mark.asyncio
async def test_wrong_shape_is_invalid_output_for_non_model_schema() -> None:
    response = await _ai_with_reply('[{"sku": "CH-1"}]').generate(prompt='hi', output_schema=list[InvoiceItem])

    assert response.output is None
    assert response.error is not None
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}


@pytest.mark.asyncio
async def test_enum_format_returns_member() -> None:
    response = await _ai_with_reply('ESCALATE').generate(prompt='hi', output_schema=TriageStr, output_format='enum')

    assert response.output is TriageStr.ESCALATE


@pytest.mark.asyncio
async def test_array_format_with_list_schema_hydrates_chunks_and_response() -> None:
    ai = _ai_with_reply(
        '[{"sku": "CH-1", "quantity": 2}, {"sku": "DK-2", "quantity": 5}]',
        chunks=['[{"sku": "CH-1", "quantity": 2}, ', '{"sku": "DK-2", "quantity": 5}]'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=list[InvoiceItem], output_format='array')
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [[InvoiceItem(sku='CH-1', quantity=2)], [InvoiceItem(sku='DK-2', quantity=5)]]
    assert response.output == [InvoiceItem(sku='CH-1', quantity=2), InvoiceItem(sku='DK-2', quantity=5)]


@pytest.mark.asyncio
async def test_stream_chunks_of_list_schema_are_models() -> None:
    ai = _ai_with_reply(
        '[{"sku": "CH-1", "quantity": 2}]',
        chunks=['[{"sku": "CH-1", "qu', 'antity": 2}]'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=list[InvoiceItem])
    outputs = [chunk.output async for chunk in stream.stream]

    # Open: the half-written item is an InvoiceItem with holes.
    first = outputs[0]
    assert first is not None
    assert isinstance(first[0], InvoiceItem)
    assert first[0].sku == 'CH-1'
    assert first[0].quantity is None
    # Closed: validated like the final response.
    assert outputs[1] == [InvoiceItem(sku='CH-1', quantity=2)]
    assert (await stream.response).output == outputs[1]


@pytest.mark.asyncio
async def test_define_prompt_accepts_generic_alias_schema() -> None:
    ai = _ai_with_reply('[{"sku": "CH-1", "quantity": 2}]')
    prompt = ai.define_prompt(name='invoice', prompt='List items.', output_schema=list[InvoiceItem])

    response = await prompt()

    assert response.output == [InvoiceItem(sku='CH-1', quantity=2)]


@pytest.mark.asyncio
async def test_looked_up_prompt_keeps_caller_generic_alias_schema() -> None:
    ai = _ai_with_reply('[{"sku": "CH-1", "quantity": 2}]')
    ai.define_prompt(name='invoice', prompt='List items.', output_schema={'type': 'array'})

    prompt = ai.prompt('invoice', output_schema=list[InvoiceItem])
    response = await prompt()

    assert response.output == [InvoiceItem(sku='CH-1', quantity=2)]


class Dish(BaseModel):
    name: str

    @field_validator('name')
    @classmethod
    def _capitalized(cls, v: str) -> str:
        if not v[:1].isupper():
            raise ValueError('must be capitalized')
        return v


def test_construct_partial_closed_raises_instead_of_model_construct() -> None:
    with pytest.raises(ValidationError):
        construct_partial(schema_type=Dish, data={'name': 'salmon'}, closed=True)


def test_construct_partial_open_still_builds() -> None:
    out = construct_partial(schema_type=Dish, data={'name': 'salm'})

    assert isinstance(out, Dish)
    assert out.name == 'salm'


def test_hydrate_partial_enum_prefix_is_none_until_whole() -> None:
    assert hydrate_partial(schema=TriageStr, data='ESC', closed=False) is None
    assert hydrate_partial(schema=TriageStr, data='ESCALATE', closed=False) is TriageStr.ESCALATE


def test_hydrate_partial_dataclass_stays_raw_until_closed() -> None:
    assert hydrate_partial(schema=Vendor, data={'vendor_name': 'Ac'}, closed=False) == {'vendor_name': 'Ac'}
    assert hydrate_partial(schema=Vendor, data={'vendor_name': 'Acme', 'risk_score': 1}, closed=True) == Vendor(
        vendor_name='Acme', risk_score=1
    )
