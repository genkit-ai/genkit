# Copyright 2025 Google LLC
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

"""Unit tests for multi-parameter tool functions and dynamic schema generation."""

from typing import Any, cast

import pytest
from pydantic import BaseModel

from genkit import Genkit, ToolRunContext


@pytest.mark.asyncio
async def test_multi_param_tool_schema_and_invocation() -> None:
    """Tools with multiple native parameters infer schema and unpack kwargs."""
    ai = Genkit()

    @ai.tool(name='get_weather')
    async def get_weather(city: str, unit: str = 'celsius') -> str:
        """Get the current weather for a city.

        Args:
            city: The target city name.
            unit: Temperature unit (celsius or fahrenheit).
        """
        return f'{city}: 22° {unit}'

    # Verify inferred input schema
    schema = cast(dict[str, Any], get_weather.input_schema)
    assert schema is not None
    props = cast(dict[str, Any], schema.get('properties', {}))
    assert 'city' in props
    assert props['city'].get('type') == 'string'
    assert props['city'].get('description') == 'The target city name.'
    assert 'unit' in props
    assert props['unit'].get('type') == 'string'
    assert props['unit'].get('default') == 'celsius'
    assert schema.get('required') == ['city']

    # Invocation via dict (model request payload)
    res1 = await get_weather({'city': 'Tokyo', 'unit': 'celsius'})
    assert res1 == 'Tokyo: 22° celsius'

    # Invocation via direct keyword arguments
    res2 = await get_weather(city='London', unit='fahrenheit')
    assert res2 == 'London: 22° fahrenheit'

    # Invocation with default argument omitted
    res3 = await get_weather(city='Berlin')
    assert res3 == 'Berlin: 22° celsius'


@pytest.mark.asyncio
async def test_multi_param_tool_with_context() -> None:
    """Tools with multi-params and ToolRunContext exclude ctx from wire schema and inject it."""
    ai = Genkit()

    @ai.tool(name='transfer_funds')
    async def transfer_funds(account_id: str, amount: float, ctx: ToolRunContext) -> str:
        """Transfer funds to an account.

        Args:
            account_id: Destination account.
            amount: Amount in USD.
        """
        assert isinstance(ctx, ToolRunContext)
        return f'Transferred ${amount:.2f} to {account_id}'

    # Verify ctx is not in the wire schema
    schema = cast(dict[str, Any], transfer_funds.input_schema)
    assert schema is not None
    props = cast(dict[str, Any], schema.get('properties', {}))
    assert 'account_id' in props
    assert 'amount' in props
    assert 'ctx' not in props
    assert schema.get('required') == ['account_id', 'amount']

    # Invocation
    res = await transfer_funds({'account_id': 'acc-999', 'amount': 150.0})
    assert res == 'Transferred $150.00 to acc-999'


@pytest.mark.asyncio
async def test_single_model_backward_compatibility() -> None:
    """Existing tools using a single Pydantic model continue to work identically."""
    ai = Genkit()

    class OrderInput(BaseModel):
        item: str
        quantity: int = 1

    @ai.tool(name='place_order')
    async def place_order(order: OrderInput) -> str:
        return f'Ordered {order.quantity}x {order.item}'

    schema = cast(dict[str, Any], place_order.input_schema)
    assert schema is not None
    assert 'properties' in schema
    props = cast(dict[str, Any], schema['properties'])
    assert 'item' in props

    res = await place_order(OrderInput(item='Latte', quantity=2))
    assert res == 'Ordered 2x Latte'
