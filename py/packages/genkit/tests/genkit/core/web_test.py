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

"""Tests for serving flows over HTTP with genkit.web."""

import datetime
import uuid

import pytest
from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

import genkit
import genkit.web
from genkit import FlowContext, Genkit, PublicError
from genkit.plugin_api import Action
from genkit.web import error_json, error_sse, read_body, result_json, to_sse, wants_stream

WRAP_MESSAGE = 'Flow request must be wrapped in {"data": ...}'


class _MenuItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    unit_price: float
    ordered_at: datetime.datetime
    order_id: uuid.UUID
    note: str | None = None


def _greet_flow() -> Action:
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    return greet


def test_request_context_types_live_on_genkit_web_only() -> None:
    """genkit.web exports the HTTP request/response wire surface, not the root package."""
    assert sorted(genkit.web.__all__) == [
        'ContextProvider',
        'RequestData',
        'StreamHandle',
        'error_body',
        'error_json',
        'error_sse',
        'error_status',
        'read_body',
        'result_json',
        'to_sse',
        'wants_stream',
    ]
    web_only = {'ContextProvider', 'RequestData', 'error_json', 'error_sse', 'result_json'}
    assert not web_only & set(genkit.__all__)
    assert not hasattr(genkit, 'RequestData')
    assert not hasattr(genkit, 'result_json')


@pytest.mark.asyncio
async def test_read_body_data_envelope_returns_the_input() -> None:
    """flow.run(input=read_body({"data": "bob"})) runs the flow with "bob"."""
    flow_input = read_body({'data': 'bob'})

    response = await _greet_flow().run(input=flow_input)

    assert flow_input == 'bob'
    assert response.response == 'hello bob'


@pytest.mark.asyncio
async def test_read_body_empty_object_means_no_input() -> None:
    """flow.run(input=read_body({})) runs a flow with an optional input on its default."""
    response = await _greet_flow().run(input=read_body({}))

    assert response.response == 'hello world'


@pytest.mark.asyncio
async def test_read_body_null_data_means_no_input() -> None:
    """flow.run(input=read_body({"data": null})) runs a flow with an optional input on its default."""
    response = await _greet_flow().run(input=read_body({'data': None}))

    assert response.response == 'hello world'


def test_read_body_input_key_is_rejected() -> None:
    """read_body({"input": 1}) is a 400 INVALID_ARGUMENT telling the caller to wrap in data."""
    with pytest.raises(PublicError) as excinfo:
        read_body({'input': 1})

    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert excinfo.value.message == WRAP_MESSAGE


@pytest.mark.parametrize('body', [[1], 'x', None, 7, {'foo': 'bar'}])
def test_read_body_non_object_is_rejected(body: object) -> None:
    """read_body of a list, string, null, number, or unwrapped object is a 400 INVALID_ARGUMENT."""
    with pytest.raises(PublicError) as excinfo:
        read_body(body)

    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert excinfo.value.message == WRAP_MESSAGE


def test_wants_stream_on_event_stream_accept_or_stream_true() -> None:
    """wants_stream is True for an event-stream Accept or ?stream=true, otherwise False."""
    assert wants_stream(accept='text/event-stream', stream=None) is True
    assert wants_stream(accept='text/event-stream, */*', stream=None) is True
    assert wants_stream(accept=None, stream='true') is True
    assert wants_stream(accept='application/json', stream=None) is False
    assert wants_stream(accept=None, stream='false') is False
    assert wants_stream(accept=None, stream=None) is False


def test_result_json_encodes_nested_models_by_alias_without_none_fields() -> None:
    """result_json emits compact camelCase JSON for nested models while preserving plain-dict nulls."""
    order_id = uuid.UUID('12345678-1234-5678-1234-567812345678')
    item = _MenuItem(
        unit_price=9.5,
        ordered_at=datetime.datetime(2026, 3, 1, 19, 30),
        order_id=order_id,
        note=None,
    )

    assert result_json([item]) == (
        '{"result":[{"unitPrice":9.5,"orderedAt":"2026-03-01T19:30:00",'
        '"orderId":"12345678-1234-5678-1234-567812345678"}]}'
    )
    assert result_json({'z': 1, 'a': None}) == '{"result":{"z":1,"a":null}}'
    assert result_json(None) == '{"result":null}'


def test_error_json_and_error_sse_preserve_public_errors_and_redact_internal_ones() -> None:
    """error_json and error_sse expose PublicError text and redact internal exceptions."""
    public = PublicError('NOT_FOUND', 'no order 99', details={'table': 4})
    secret = RuntimeError('postgres://admin:secret@db.internal/orders')

    assert error_json(public) == '{"message":"no order 99","status":"NOT_FOUND","details":{"table":4}}'
    assert (
        error_sse(public) == 'data: {"error":{"message":"no order 99","status":"NOT_FOUND","details":{"table":4}}}\n\n'
    )
    assert error_json(secret) == '{"message":"Internal Error","status":"INTERNAL"}'
    assert error_sse(secret) == 'data: {"error":{"message":"Internal Error","status":"INTERNAL"}}\n\n'


@pytest.mark.asyncio
async def test_web_encoders_emit_unescaped_utf8() -> None:
    """Non-ASCII UTF-8 strings serialize unescaped across JSON and SSE encoders."""
    ai = Genkit()

    @ai.flow()
    async def daily_special(ctx: FlowContext[str]) -> str:
        ctx.send_chunk('café')
        return 'crème brûlée'

    err = PublicError('INVALID_ARGUMENT', 'café')

    assert result_json('café') == '{"result":"café"}'
    assert error_json(err) == '{"message":"café","status":"INVALID_ARGUMENT"}'
    assert error_sse(err) == 'data: {"error":{"message":"café","status":"INVALID_ARGUMENT"}}\n\n'
    assert [line async for line in to_sse(daily_special.stream())] == [
        'data: {"message":"café"}\n\n',
        'data: {"result":"crème brûlée"}\n\n',
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_float', [float('nan'), float('inf'), float('-inf')])
async def test_web_encoders_send_non_finite_floats_as_null(bad_float: float) -> None:
    """Out-of-range floats go out as null, like tool outputs and JS JSON.stringify, not a 500."""
    assert result_json(bad_float) == '{"result":null}'

    ai = Genkit()

    @ai.flow()
    async def compute_score() -> float:
        return bad_float

    lines = [line async for line in to_sse(compute_score.stream())]

    assert lines == ['data: {"result":null}\n\n']
