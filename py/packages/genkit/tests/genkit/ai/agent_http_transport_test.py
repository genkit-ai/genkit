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

"""Unit tests for HTTP agent transport stream/error parsing."""

import json

import pytest

from genkit._ai._agents._transports._http import parse_stream_line, stream_error_from_payload
from genkit._core._error import GenkitError


def test_parse_stream_line_plain_json() -> None:
    data = parse_stream_line('{"result": {"finishReason": "stop"}}')
    assert data == {'result': {'finishReason': 'stop'}}


def test_parse_stream_line_sse_data_prefix() -> None:
    data = parse_stream_line('data: {"message": {"modelChunk": {"role": "model"}}}')
    assert data == {'message': {'modelChunk': {'role': 'model'}}}


def test_parse_stream_line_sse_error_prefix() -> None:
    data = parse_stream_line('error: {"error": {"status": "INTERNAL", "message": "boom"}}')
    assert data == {'error': {'status': 'INTERNAL', 'message': 'boom'}}


def test_stream_error_from_payload_callable() -> None:
    err = stream_error_from_payload({'error': {'status': 'UNAVAILABLE', 'message': 'down'}})
    assert isinstance(err, GenkitError)
    assert err.status == 'UNAVAILABLE'
    assert err.original_message == 'down'


def test_stream_error_from_payload_reflection() -> None:
    err = stream_error_from_payload({'error': {'code': 13, 'message': 'boom'}})
    assert err.status == 'INTERNAL'
    assert err.original_message == 'boom'


def test_stream_error_from_payload_fastapi_wrapper() -> None:
    wrapped = {'error': {'error': {'status': 'INTERNAL', 'message': 'wrapped'}}}
    err = stream_error_from_payload(wrapped)
    assert err.original_message == 'wrapped'


def test_parse_stream_line_empty_returns_none() -> None:
    assert parse_stream_line('') is None
    assert parse_stream_line('   ') is None


def test_parse_stream_line_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_stream_line('data: not-json')
