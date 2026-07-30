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

"""Integration tests for HttpAgentTransport against a flow-shaped HTTP server."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route
from uvicorn import Config, Server

from genkit.agent import (
    AgentClient,
    AgentFinishReason,
    HttpAgentTransport,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


async def _agent_endpoint(request: Request) -> StreamingResponse:
    """Minimal expressHandler-shaped agent endpoint: {data, init} + SSE."""
    body = await request.json()
    assert 'data' in body, 'request must use the flow {"data": ...} envelope'
    assert 'input' not in body
    assert 'key' not in body
    assert 'text/event-stream' in request.headers.get('accept', '')

    text = ''
    data = body.get('data') or {}
    message = data.get('message') or {}
    content = message.get('content') or []
    if content:
        text = content[0].get('text', '')

    async def event_stream() -> AsyncIterator[str]:
        result = {
            'finishReason': 'stop',
            'message': {
                'role': 'model',
                'content': [{'text': f'Echo: {text}'}],
            },
        }
        yield f'data: {json.dumps({"result": result})}\n\n'

    return StreamingResponse(event_stream(), media_type='text/event-stream')


@pytest.mark.asyncio
async def test_http_transport_flow_envelope_integration() -> None:
    port = _find_free_port()
    app = Starlette(routes=[Route('/weatherAgent', _agent_endpoint, methods=['POST'])])
    config = Config(app=app, host='127.0.0.1', port=port, log_level='error')
    server = Server(config)
    task = asyncio.create_task(server.serve())

    try:
        for _ in range(50):
            if server.started:
                break
            await asyncio.sleep(0.05)
        assert server.started

        transport = HttpAgentTransport(
            url=f'http://127.0.0.1:{port}/weatherAgent',
            state_management='server',
        )
        client = AgentClient(transport)
        chat = client.chat()
        res = await chat.send('Hello Genkit!')
        assert res.text == 'Echo: Hello Genkit!'
        assert res.finish_reason == AgentFinishReason.STOP
    finally:
        server.should_exit = True
        await task
