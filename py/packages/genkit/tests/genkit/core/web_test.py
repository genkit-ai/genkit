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

"""Tests for reading a served flow's HTTP request with genkit.web."""

import pytest

from genkit import Genkit, PublicError
from genkit.plugin_api import Action
from genkit.web import read_body, wants_stream

WRAP_MESSAGE = 'Flow request must be wrapped in {"data": ...}'


def _greet_flow() -> Action:
    ai = Genkit()

    @ai.flow()
    async def greet(name: str = 'world') -> str:
        return f'hello {name}'

    return greet


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
    assert excinfo.value.original_message == WRAP_MESSAGE


@pytest.mark.parametrize('body', [[1], 'x', None, 7, {'foo': 'bar'}])
def test_read_body_non_object_is_rejected(body: object) -> None:
    """read_body of a list, string, null, number, or unwrapped object is a 400 INVALID_ARGUMENT."""
    with pytest.raises(PublicError) as excinfo:
        read_body(body)

    assert excinfo.value.status == 'INVALID_ARGUMENT'
    assert excinfo.value.original_message == WRAP_MESSAGE


def test_wants_stream_on_event_stream_accept_or_stream_true() -> None:
    """wants_stream is True for an event-stream Accept or ?stream=true, otherwise False."""
    assert wants_stream(accept='text/event-stream', stream=None) is True
    assert wants_stream(accept='text/event-stream, */*', stream=None) is True
    assert wants_stream(accept=None, stream='true') is True
    assert wants_stream(accept='application/json', stream=None) is False
    assert wants_stream(accept=None, stream='false') is False
    assert wants_stream(accept=None, stream=None) is False
