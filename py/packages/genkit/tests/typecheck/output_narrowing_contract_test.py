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

"""Type-checker contract for ``.output`` narrowing (PR2-15).

With ``output_schema=T``, ``response.output`` types as ``T``: no ``None``
guard and no ``cast()`` on the happy path. Stream chunks stay ``T | None``
because early chunks have nothing to parse. With no schema or a JSON-schema
dict, ``.output`` types as ``Any`` so untyped callers can index it.

pyright, pyrefly, and ty check this file (see ``[tool.pyright]``); the tests
also run so each static type matches the runtime value.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit.testing import define_scripted_model, define_static_response_model


class Order(BaseModel):
    dish: str
    quantity: int


@dataclass
class Allergen:
    name: str


class Spice(Enum):
    MILD = 'mild'
    HOT = 'hot'


def _scripted(ai: Genkit, text: str, chunks: list[str] | None = None) -> None:
    define_scripted_model(
        ai,
        name='kitchen',
        responses=[ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))],
        chunks=[[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(c)]) for c in chunks]] if chunks else None,
    )


@pytest.mark.asyncio
async def test_generate_output_is_schema_type() -> None:
    """``output_schema=Order`` gives ``Order``, usable without a guard."""
    ai = Genkit()
    _scripted(ai, '{"dish": "tartine", "quantity": 2}')

    res = await ai.generate(model='kitchen', prompt='order', output_schema=Order)

    assert_type(res, ModelResponse[Order])
    assert_type(res.output, Order)
    total: int = res.output.quantity * 3
    assert total == 6


@pytest.mark.asyncio
async def test_generate_output_with_config_dict_is_schema_type() -> None:
    """The ``config={...}`` overload narrows the same way."""
    ai = Genkit()
    _scripted(ai, '{"dish": "soup", "quantity": 1}')

    res = await ai.generate(model='kitchen', prompt='order', config={'temperature': 0.1}, output_schema=Order)

    assert_type(res.output, Order)
    assert res.output.dish == 'soup'


@pytest.mark.asyncio
async def test_generate_output_non_model_schemas() -> None:
    """``list[T]``, dataclasses, and ``Enum`` narrow to the hydrated type."""
    ai = Genkit()
    _, _ = define_static_response_model(ai, '[{"dish": "tartine", "quantity": 1}]', name='orders')
    _, _ = define_static_response_model(ai, '{"name": "peanut"}', name='allergen')
    _, _ = define_static_response_model(ai, '"hot"', name='spice')

    orders = await ai.generate(model='orders', prompt='x', output_schema=list[Order])
    assert_type(orders.output, list[Order])
    assert orders.output[0].dish == 'tartine'

    allergen = await ai.generate(model='allergen', prompt='x', output_schema=Allergen)
    assert_type(allergen.output, Allergen)
    assert allergen.output.name == 'peanut'

    spice = await ai.generate(model='spice', prompt='x', output_schema=Spice)
    assert_type(spice.output, Spice)
    assert spice.output is Spice.HOT


@pytest.mark.asyncio
async def test_untyped_output_is_any() -> None:
    """No schema or a dict schema types as ``Any``, so indexing type-checks."""
    ai = Genkit()
    _, _ = define_static_response_model(ai, '{"dish": "tartine"}', name='menu')

    res = await ai.generate(model='menu', prompt='x')
    assert_type(res.output, Any)

    res_dict = await ai.generate(model='menu', prompt='x', output_schema={'type': 'object'})
    assert_type(res_dict.output, Any)
    assert res_dict.output['dish'] == 'tartine'


@pytest.mark.asyncio
async def test_stream_final_output_narrows_chunks_stay_optional() -> None:
    """Chunks are ``Order | None``; the awaited response is ``Order``."""
    ai = Genkit()
    _scripted(ai, '{"dish": "tartine", "quantity": 2}', chunks=['{"dish": "tar', 'tine", "quantity": 2}'])

    stream = ai.generate_stream(model='kitchen', prompt='order', output_schema=Order)
    seen: list[Order | None] = []
    async for chunk in stream.stream:
        assert_type(chunk, ModelResponseChunk[Order])
        assert_type(chunk.output, Order | None)
        seen.append(chunk.output)
    final = await stream.response

    assert_type(final.output, Order)
    assert final.output.quantity == 2
    last = seen[-1]
    assert last is not None and last.dish == 'tartine'


@pytest.mark.asyncio
async def test_prompt_output_narrows() -> None:
    """``define_prompt`` / ``ai.prompt`` with ``output_schema`` narrow on call and stream."""
    ai = Genkit()
    _, _ = define_static_response_model(ai, '{"dish": "tartine", "quantity": 2}', name='kitchen')
    order_prompt = ai.define_prompt(name='order', model='kitchen', prompt='order', output_schema=Order)

    res = await order_prompt()
    assert_type(res.output, Order)
    assert res.output.dish == 'tartine'

    looked_up = ai.prompt('order', output_schema=Order)
    looked_up_res = await looked_up()
    assert_type(looked_up_res.output, Order)
    assert looked_up_res.output.quantity == 2

    stream = order_prompt.stream()
    async for chunk in stream.stream:
        assert_type(chunk.output, Order | None)
    final = await stream.response
    assert_type(final.output, Order)
    assert final.output.dish == 'tartine'


def test_bare_model_response_output_is_object() -> None:
    """Unparameterized ``ModelResponse`` defaults ``OutputT`` to ``object``."""

    def read(res: ModelResponse) -> object:
        assert_type(res.output, object)
        return res.output

    assert read(ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('hi')]))) is None
