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

"""HttpAgentTransport posts the callable/flow envelope ({data, init})."""

from __future__ import annotations

from typing import Any
from unittest import mock

import pytest

from genkit._ai._agents._transports._http import HttpAgentTransport
from genkit._core._typing import AgentInit, AgentInput, MessageData, Part, TextPart

URL = 'http://example.test/weatherAgent'
RESULT_LINE = 'data: {"result": {"finishReason": "stop", "message": {"role": "model", "content": [{"text": "ok"}]}}}'


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> Any:
        self.calls.append({'url': url, 'json': json, 'headers': headers})

        class Resp:
            status_code = 200

            async def aread(self) -> bytes:
                return b''

            async def aiter_lines(self):
                yield RESULT_LINE

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        return Resp()

    async def post(self, url: str, *, json: dict[str, Any]) -> Any:
        self.calls.append({'url': url, 'json': json})

        class Resp:
            status_code = 200
            content = b'{"result": null}'
            text = '{"result": null}'

            def json(self) -> dict[str, Any]:
                return {'result': None}

        return Resp()


@pytest.mark.asyncio
async def test_run_turn_posts_data_init_envelope_with_accept_header() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server')
    with mock.patch(
        'genkit._ai._agents._transports._http.get_cached_client',
        return_value=client,
    ):
        stream, output = await transport.run_turn(
            agent_input=AgentInput(message=MessageData(role='user', content=[Part(root=TextPart(text='hi'))])),
            init=AgentInit(snapshot_id='snap-1'),
        )
        async for _ in stream:
            pass
        await output

    call = client.calls[0]
    assert call['url'] == URL
    assert call['headers'] == {'Accept': 'text/event-stream', 'Content-Type': 'application/json'}
    assert set(call['json']) == {'data', 'init'}
    assert call['json']['init'] == {'snapshotId': 'snap-1'}


@pytest.mark.asyncio
async def test_get_snapshot_posts_data_envelope() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server')
    with mock.patch(
        'genkit._ai._agents._transports._http.get_cached_client',
        return_value=client,
    ):
        await transport.get_snapshot(snapshot_id='snap-1')

    assert client.calls[0]['json'] == {'data': {'snapshotId': 'snap-1'}}
