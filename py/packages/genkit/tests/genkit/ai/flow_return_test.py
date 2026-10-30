# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A flow's return value is checked against its return annotation."""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, ConfigDict, ValidationError

from genkit import Genkit, GenkitError
from genkit._core._error import RuntimeErrorReason
from genkit._core._reflection import create_reflection_asgi_app


class Receipt(BaseModel):
    account: str
    amount: int


def _assert_invalid_output(error: GenkitError, flow_name: str) -> None:
    assert error.status == 'INTERNAL'
    assert error.reason is RuntimeErrorReason.INVALID_OUTPUT
    assert isinstance(error.cause, ValidationError)
    assert f"Flow '{flow_name}' returned a value that doesn't match its return annotation" in str(error)


@pytest.mark.asyncio
async def test_flow_returning_matching_dict_gives_caller_the_model() -> None:
    """`-> Receipt` returning `{'account': 'acme', 'amount': 1}` returns a `Receipt` instance."""
    ai = Genkit()

    @ai.flow()
    async def charge(name: str) -> Receipt:
        return {'account': name, 'amount': 1}  # type: ignore[return-value]

    result = await charge('acme')

    assert result == Receipt(account='acme', amount=1)
    assert result.account == 'acme'


@pytest.mark.asyncio
async def test_flow_returning_model_instance_returns_same_instance() -> None:
    """Returning a `Receipt` returns that object unchanged."""
    ai = Genkit()
    receipt = Receipt(account='acme', amount=1)

    @ai.flow()
    async def charge(name: str) -> Receipt:
        return receipt

    assert await charge('acme') is receipt


@pytest.mark.asyncio
async def test_flow_returning_wrong_shape_raises_invalid_output_with_validation_cause() -> None:
    """`-> Receipt` returning `{'account': 'acme'}` raises INTERNAL / INVALID_OUTPUT with a ValidationError cause."""
    ai = Genkit()

    @ai.flow()
    async def charge(name: str) -> Receipt:
        return {'account': name}  # type: ignore[return-value]

    with pytest.raises(GenkitError) as exc:
        await charge('acme')

    _assert_invalid_output(exc.value, 'charge')
    assert str(exc.value) == (
        "INTERNAL: Flow 'charge' returned a value that doesn't match its return annotation: amount: Field required"
    )


@pytest.mark.asyncio
async def test_flow_returning_int_for_str_annotation_raises_invalid_output() -> None:
    """`-> str` returning `5` raises INVALID_OUTPUT."""
    ai = Genkit()

    @ai.flow()
    async def label(name: str) -> str:
        return 5  # type: ignore[return-value]

    with pytest.raises(GenkitError) as exc:
        await label('acme')

    _assert_invalid_output(exc.value, 'label')


@pytest.mark.asyncio
async def test_flow_annotated_none_returning_value_raises_invalid_output() -> None:
    """`-> None` returning `'x'` raises INVALID_OUTPUT."""
    ai = Genkit()

    @ai.flow()
    async def notify(name: str) -> None:
        return 'x'  # type: ignore[return-value]

    with pytest.raises(GenkitError) as exc:
        await notify('acme')

    _assert_invalid_output(exc.value, 'notify')


@pytest.mark.asyncio
async def test_flow_without_return_annotation_returns_value_unchecked() -> None:
    """No annotation: whatever the body returns comes back."""
    ai = Genkit()

    @ai.flow()
    async def anything(name: str):  # noqa: ANN202
        return {'account': name}

    assert await anything('acme') == {'account': 'acme'}


@pytest.mark.asyncio
async def test_flow_that_raises_keeps_its_own_error() -> None:
    """A body that raises `KeyError` surfaces that failure, not INVALID_OUTPUT."""
    ai = Genkit()

    @ai.flow()
    async def charge(name: str) -> Receipt:
        raise KeyError(name)

    with pytest.raises(KeyError) as exc:
        await charge('acme')

    assert exc.value.args == ('acme',)


@pytest.mark.asyncio
async def test_streamed_flow_response_is_the_model() -> None:
    """`flow.stream(x).response` resolves to a `Receipt` when the body returned a dict."""
    ai = Genkit()

    @ai.flow()
    async def charge(name: str) -> Receipt:
        return {'account': name, 'amount': 1}  # type: ignore[return-value]

    assert await charge.stream('acme').response == Receipt(account='acme', amount=1)


class TabReceipt(BaseModel):
    table: int
    note: str | None = None


class StrictReceipt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    table: int
    note: str | None = None


class Secret(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)
    password: int


@pytest.mark.asyncio
async def test_flow_returning_dict_with_extra_keys_gives_model_without_them() -> None:
    """`-> Receipt` returning `{'table': 4, 'tip_cents': 300}` gives `Receipt(table=4, note=None)`."""
    ai = Genkit()

    @ai.flow()
    async def close_tab(table: int) -> TabReceipt:
        return {'table': table, 'tip_cents': 300}  # type: ignore[return-value]

    result = await close_tab(4)

    assert result == TabReceipt(table=4, note=None)
    assert not hasattr(result, 'tip_cents')


@pytest.mark.asyncio
async def test_flow_returning_extra_keys_for_forbid_model_raises_invalid_output() -> None:
    """`-> StrictReceipt` returning an extra key raises INTERNAL / INVALID_OUTPUT."""
    ai = Genkit()

    @ai.flow()
    async def close_tab(table: int) -> StrictReceipt:
        return {'table': table, 'tip_cents': 300}  # type: ignore[return-value]

    with pytest.raises(GenkitError) as exc:
        await close_tab(4)

    _assert_invalid_output(exc.value, 'close_tab')
    assert 'tip_cents' in str(exc.value)
    assert 'Extra inputs are not permitted' in str(exc.value)


@pytest.mark.asyncio
async def test_flow_return_with_hidden_input_model_omits_value_from_error() -> None:
    """A hidden-input return model does not print the rejected value in `str(e)`."""
    ai = Genkit()

    @ai.flow()
    async def login(_name: str) -> Secret:
        return {'password': 'hunter2-secret'}  # type: ignore[return-value]

    with pytest.raises(GenkitError) as exc:
        await login('ada')

    assert 'hunter2' not in str(exc.value)
    assert 'got ' not in str(exc.value)


@pytest.mark.asyncio
async def test_tool_returning_wrong_shape_is_not_checked() -> None:
    """A `@ai.tool` `-> int` returning `'x'` gives back `'x'`."""
    ai = Genkit()

    @ai.tool()
    async def echo(_name: str) -> int:
        return 'x'  # type: ignore[return-value]

    result = await echo('ada')
    assert result.output == 'x'


@pytest.mark.asyncio
async def test_dev_ui_run_with_bad_return_reports_internal_invalid_output(hex_ids: None) -> None:
    """A Dev UI runAction of a bad-return flow gets INTERNAL, reason INVALID_OUTPUT, and the real message."""
    ai = Genkit()

    @ai.flow()
    async def charge(name: str) -> Receipt:
        return {'account': name}  # type: ignore[return-value]

    client = AsyncClient(transport=ASGITransport(app=create_reflection_asgi_app(ai._registry)), base_url='http://test')
    try:
        response = await client.post('/api/runAction', json={'key': '/flow/charge', 'input': 'acme'})
    finally:
        await client.aclose()

    error: dict[str, Any] = response.json()['error']
    assert error['code'] == 13  # INTERNAL
    assert error['details']['reason'] == 'INVALID_OUTPUT'
    assert "Flow 'charge' returned a value that doesn't match its return annotation" in error['message']
    assert error['details']['traceId'] == response.headers['x-genkit-trace-id']
