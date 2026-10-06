# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Calling a flow: `context=` on the plain call, and no input vs an explicit None."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from genkit import ActionRunContext, Genkit, GenkitError
from genkit._core._reflection import create_reflection_asgi_app


@pytest.mark.asyncio
async def test_await_flow_with_context_keyword_reaches_ctx_context() -> None:
    """`await f('ada', context={'auth': {...}})` shows that auth on `ctx.context`."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str, ctx: ActionRunContext) -> str:
        return f'hello {name} as {ctx.context["auth"]["uid"]}'

    assert await greet('ada', context={'auth': {'uid': 'u1'}}) == 'hello ada as u1'


@pytest.mark.asyncio
async def test_await_flow_without_context_keyword_sees_empty_context() -> None:
    """`await f('ada')` runs with an empty `ctx.context`."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str, ctx: ActionRunContext) -> str:
        return f'hello {name} with {ctx.context!r}'

    assert await greet('ada') == 'hello ada with {}'


@pytest.mark.asyncio
async def test_await_flow_with_context_and_no_input_uses_default_and_gets_context() -> None:
    """`await greet(context=...)` runs with the Python default and still sees the context."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        context = ai.current_context() or {}
        return f'hello {name} as {context["auth"]["uid"]}'

    assert await greet(context={'auth': {'uid': 'u1'}}) == 'hello world as u1'


@pytest.mark.asyncio
async def test_await_flow_with_no_input_uses_python_default() -> None:
    """`await greet()` runs with `name='world'`."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    assert await greet() == 'hello world'


@pytest.mark.asyncio
async def test_await_flow_with_explicit_none_and_str_default_raises_invalid_argument() -> None:
    """`await greet(None)` on `name: str = 'world'` raises INVALID_ARGUMENT instead of using the default."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    with pytest.raises(GenkitError) as exc:
        await greet(None)

    assert exc.value.status == 'INVALID_ARGUMENT'
    assert "Invalid input for action 'greet'" in str(exc.value)


@pytest.mark.asyncio
async def test_await_flow_with_explicit_none_and_optional_input_receives_none() -> None:
    """`await greet(None)` on `name: str | None = 'world'` runs with `name=None`."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str | None = 'world') -> str:
        return f'hello {name}'

    assert await greet(None) == 'hello None'
    assert await greet() == 'hello world'


@pytest.mark.asyncio
async def test_await_flow_with_explicit_none_and_no_default_raises_invalid_argument() -> None:
    """`await greet(None)` on a required `name: str` raises INVALID_ARGUMENT."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str) -> str:
        return f'hello {name}'

    with pytest.raises(GenkitError) as exc:
        await greet(None)

    assert exc.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_await_flow_with_no_input_and_no_default_raises_input_required() -> None:
    """`await greet()` on a required `name: str` raises INVALID_ARGUMENT saying input is required."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str) -> str:
        return f'hello {name}'

    with pytest.raises(GenkitError) as exc:
        await greet()

    assert exc.value.status == 'INVALID_ARGUMENT'
    assert "Action 'greet' requires input but none was provided" in str(exc.value)


class Item(BaseModel):
    dish: str
    qty: int


class Order(BaseModel):
    table: int
    items: list[Item]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('call_input', 'want'),
    [
        pytest.param(
            None,
            "INVALID_ARGUMENT: Invalid input for action 'place': "
            'Input should be a valid dictionary or instance of Order, got None',
            id='None',
        ),
        pytest.param(
            {'table': 4, 'items': [{'dish': 'pad thai'}]},
            "INVALID_ARGUMENT: Invalid input for action 'place': items[0].qty: Field required",
            id='nested field missing',
        ),
    ],
)
async def test_flow_invalid_model_input_names_the_field_once(call_input: object, want: str) -> None:
    """`place(order: Order)` given bad input says which field is wrong, without Pydantic's dump or a repeat."""
    ai = Genkit()

    @ai.flow()
    async def place(order: Order) -> str:
        return order.items[0].dish

    with pytest.raises(GenkitError) as exc:
        await place(call_input)  # type: ignore[arg-type]

    assert str(exc.value) == want


@pytest.mark.asyncio
async def test_flow_with_no_input_and_required_model_says_input_is_required() -> None:
    """`await place()` on a required `order: Order` says input is missing, not Pydantic's `got None`."""
    ai = Genkit()

    @ai.flow()
    async def place(order: Order) -> str:
        return order.items[0].dish

    with pytest.raises(GenkitError) as exc:
        await place()

    assert str(exc.value) == "INVALID_ARGUMENT: Action 'place' requires input but none was provided."


@pytest.mark.asyncio
async def test_flow_run_with_explicit_none_and_str_default_raises_invalid_argument() -> None:
    """`greet.run(None)` follows the same rule as `await greet(None)`."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    with pytest.raises(GenkitError) as exc:
        await greet.run(None)

    assert exc.value.status == 'INVALID_ARGUMENT'
    assert (await greet.run()).response == 'hello world'


@pytest.mark.asyncio
async def test_flow_stream_with_no_input_uses_python_default() -> None:
    """`greet.stream().response` resolves with the Python default."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    assert await greet.stream().response == 'hello world'


@asynccontextmanager
async def _dev_ui_client(ai: Genkit) -> AsyncIterator[AsyncClient]:
    client = AsyncClient(transport=ASGITransport(app=create_reflection_asgi_app(ai.registry)), base_url='http://test')
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_dev_ui_run_without_input_uses_python_default() -> None:
    """A Dev UI runAction with no `input` runs the defaulted flow with its default."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    async with _dev_ui_client(ai) as client:
        response = await client.post('/api/runAction', json={'key': '/flow/greet'})

    assert response.status_code == 200
    assert response.json()['result'] == 'hello world'


@pytest.mark.asyncio
async def test_dev_ui_run_with_null_input_uses_python_default() -> None:
    """A Dev UI runAction with `"input": null` behaves like no input."""
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    async with _dev_ui_client(ai) as client:
        response = await client.post('/api/runAction', json={'key': '/flow/greet', 'input': None})

    assert response.status_code == 200
    assert response.json()['result'] == 'hello world'
