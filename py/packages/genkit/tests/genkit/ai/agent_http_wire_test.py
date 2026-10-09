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

import httpx
import pytest

from genkit import Part
from genkit._ai._agents._transports._http import HttpAgentTransport
from genkit._core._model import AgentInit, AgentInput, Message

URL = 'http://example.test/weatherAgent'
RESULT_LINE = 'data: {"result": {"finishReason": "stop", "message": {"role": "model", "content": [{"text": "ok"}]}}}'


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def stream(
        self, method: str, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: Any = None
    ) -> Any:
        self.calls.append({'url': url, 'json': json, 'headers': headers, 'timeout': timeout})

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

    async def post(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str] | None = None, timeout: Any = None
    ) -> Any:
        self.calls.append({'url': url, 'json': json, 'headers': headers or {}, 'timeout': timeout})

        class Resp:
            status_code = 200
            content = b'{"result": null}'
            text = '{"result": null}'

            def json(self) -> dict[str, Any]:
                return {'result': None}

        return Resp()


async def _run_turn(transport: HttpAgentTransport) -> None:
    stream, output = await transport.run_turn(
        agent_input=AgentInput(message=Message(role='user', content=[Part.from_text('hi')])),
        init=AgentInit(snapshot_id='snap-1'),
    )
    async for _ in stream:
        pass
    await output


@pytest.mark.asyncio
async def test_run_turn_posts_data_init_envelope_with_accept_header() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server')
    with mock.patch(
        'genkit._ai._agents._transports._http._agent_client',
        return_value=client,
    ):
        await _run_turn(transport)

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
        'genkit._ai._agents._transports._http._agent_client',
        return_value=client,
    ):
        await transport.get_snapshot(snapshot_id='snap-1')

    assert client.calls[0]['json'] == {'data': {'snapshotId': 'snap-1'}}


@pytest.mark.asyncio
async def test_static_headers_on_turn_and_snapshot() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(
        url=URL,
        state_management='server',
        headers={'Authorization': 'Bearer static'},
    )
    with mock.patch(
        'genkit._ai._agents._transports._http._agent_client',
        return_value=client,
    ):
        await _run_turn(transport)
        await transport.get_snapshot(snapshot_id='snap-1')

    assert client.calls[0]['headers'] == {
        'Authorization': 'Bearer static',
        'Accept': 'text/event-stream',
        'Content-Type': 'application/json',
    }
    assert client.calls[1]['headers'] == {'Authorization': 'Bearer static'}


@pytest.mark.asyncio
async def test_sync_callable_headers_resolved_per_request() -> None:
    client = FakeClient()
    tokens = iter(['tok-1', 'tok-2'])
    transport = HttpAgentTransport(
        url=URL,
        state_management='server',
        headers=lambda: {'Authorization': f'Bearer {next(tokens)}'},
    )
    with mock.patch(
        'genkit._ai._agents._transports._http._agent_client',
        return_value=client,
    ):
        await _run_turn(transport)
        await transport.get_snapshot(snapshot_id='snap-1')

    assert client.calls[0]['headers']['Authorization'] == 'Bearer tok-1'
    assert client.calls[1]['headers']['Authorization'] == 'Bearer tok-2'


@pytest.mark.asyncio
async def test_async_callable_headers_resolved_per_request() -> None:
    client = FakeClient()
    n = {'i': 0}

    async def refresh() -> dict[str, str]:
        n['i'] += 1
        return {'Authorization': f'Bearer async-{n["i"]}'}

    transport = HttpAgentTransport(
        url=URL,
        state_management='server',
        headers=refresh,
    )
    with mock.patch(
        'genkit._ai._agents._transports._http._agent_client',
        return_value=client,
    ):
        await _run_turn(transport)
        await transport.abort_snapshot('snap-1')

    assert client.calls[0]['headers']['Authorization'] == 'Bearer async-1'
    assert client.calls[1]['headers']['Authorization'] == 'Bearer async-2'


@pytest.mark.asyncio
async def test_timeout_defaults_to_60s_reads_and_10s_connect() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server')
    with mock.patch('genkit._ai._agents._transports._http._agent_client', return_value=client):
        await _run_turn(transport)
        await transport.get_snapshot(snapshot_id='snap-1')

    for call in client.calls:
        assert call['timeout'] == httpx.Timeout(60.0, connect=10.0)


@pytest.mark.asyncio
async def test_timeout_reaches_turn_and_one_shot_requests() -> None:
    """A long research turn can go quiet for minutes between chunks."""
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server', timeout=600.0)
    with mock.patch('genkit._ai._agents._transports._http._agent_client', return_value=client):
        await _run_turn(transport)
        await transport.abort_snapshot('snap-1')

    assert [call['timeout'] for call in client.calls] == [httpx.Timeout(600.0, connect=10.0)] * 2


@pytest.mark.asyncio
async def test_timeout_none_waits_indefinitely_but_still_bounds_connect() -> None:
    client = FakeClient()
    transport = HttpAgentTransport(url=URL, state_management='server', timeout=None)
    with mock.patch('genkit._ai._agents._transports._http._agent_client', return_value=client):
        await _run_turn(transport)

    assert client.calls[0]['timeout'] == httpx.Timeout(None, connect=10.0)
