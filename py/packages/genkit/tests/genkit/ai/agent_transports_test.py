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

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any
from unittest import mock

import pytest
import websockets

from genkit import ActionRunContext, Genkit
from genkit._core._environment import GENKIT_ENV, GenkitEnvironment
from genkit._core._reflection import ServerSpec
from genkit._core._typing import MessageData, Part, TextPart
from genkit.agent import (
    AgentClient,
    AgentFinishReason,
    AgentInput,
    AgentResult,
    HttpAgentTransport,
    InMemoryLatestStateStore,
    SessionRunner,
    SnapshotStatus,
    TurnResult,
    WebSocketAgentTransport,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# HTTP Transport Integration Test (Real Starlette / ReflectionServer)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_transport_integration() -> None:
    port = _find_free_port()

    # Configure Dev environment so Genkit starts the reflection server
    with mock.patch.dict(os.environ, {GENKIT_ENV: GenkitEnvironment.DEV}):
        ai = Genkit(
            reflection_server_spec=ServerSpec(scheme='http', host='127.0.0.1', port=port),
            plugins=[],
        )

        # Register a simple custom agent
        async def echo_agent(sess: SessionRunner, ctx: ActionRunContext) -> AgentResult:
            async def handle_turn(inp: AgentInput) -> TurnResult | None:
                text = inp.message.content[0].root.text if inp.message else ''
                await sess.add_messages(MessageData(role='model', content=[Part(root=TextPart(text=f'Echo: {text}'))]))
                return TurnResult(finish_reason=AgentFinishReason.STOP)

            await sess.run(handle_turn)
            return await sess.result()

        ai.define_custom_agent(name='echoAgent', fn=echo_agent, store=InMemoryLatestStateStore())

        # Wait for the Starlette reflection server to become active in the background
        # (Using private attr for test synchronization, as in reflection_server_test.py)
        assert ai._reflection_ready.wait(timeout=5), 'Reflection server never became ready'  # type: ignore[reportPrivateUsage]

        try:
            # 1. Instantiate the HTTP transport talking to the reflection endpoint
            url = f'http://127.0.0.1:{port}/api/runAction'
            transport = HttpAgentTransport(url=url, agent_name='/agent/echoAgent')
            client = AgentClient(transport)

            # 2. Run a turn and check the stream and output!
            chat = client.chat()
            turn = chat.send('Hello Genkit!')

            chunks = []
            async for chunk in turn.stream:
                if chunk.text:
                    chunks.append(chunk.text)

            res = await turn.output
            assert res.text == 'Echo: Hello Genkit!'
            assert res.finish_reason == AgentFinishReason.STOP
            await chat.close()
        finally:
            # Shutdown Genkit and reflection server
            pass


# ---------------------------------------------------------------------------
# WebSocket Transport Integration Test (Real local WebSocket server)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_transport_integration() -> None:
    ws_port = _find_free_port()
    ws_url = f'ws://127.0.0.1:{ws_port}'

    connections_count = 0
    server_received_messages = []

    # 1. Define a mock WebSocket server that implements the Genkit agent protocol
    async def mock_agent_server(websocket: Any) -> None:
        nonlocal connections_count
        connections_count += 1
        try:
            async for message in websocket:
                server_received_messages.append(message)
                data = json.loads(message)
                # Parse inputs
                text = data.get('input', {}).get('message', {}).get('content', [{}])[0].get('text', '')

                # Stream a chunk back
                await websocket.send(
                    json.dumps({
                        'chunk': {
                            'modelChunk': {
                                'role': 'model',
                                'content': [{'text': f'Streaming: {text}'}],
                            }
                        }
                    })
                )
                await asyncio.sleep(0.1)

                # Send the final output to resolve the future
                await websocket.send(
                    json.dumps({
                        'output': {
                            'message': {
                                'role': 'model',
                                'content': [{'text': f'Final: {text}'}],
                            },
                            'finishReason': 'stop',
                        }
                    })
                )
        except websockets.exceptions.ConnectionClosed:
            pass

    # Start the local WebSocket server in the background
    async with websockets.serve(mock_agent_server, '127.0.0.1', ws_port):
        # 2. Instantiate the WebSocket transport and client
        transport = WebSocketAgentTransport(url=ws_url)
        client = AgentClient(transport)

        # 3. Open a session and run a turn!
        chat = client.chat()
        turn = chat.send('Hi WebSocket!')

        chunks = []
        async for chunk in turn.stream:
            if chunk.text:
                chunks.append(chunk.text)

        res = await turn.output
        assert res.text == 'Final: Hi WebSocket!'
        assert res.finish_reason == AgentFinishReason.STOP

        # Verify stream chunks
        assert len(chunks) > 0
        assert 'Streaming: Hi WebSocket!' in chunks[0]

        # Verify server connection
        assert connections_count == 1
        assert len(server_received_messages) == 1

        # 4. Test Reconnection: Close socket on the session's transport, then run a second turn
        await chat._transport.close()
        turn2 = chat.send('Reconnected!')
        res2 = await turn2.output
        assert res2.text == 'Final: Reconnected!'
        # Connection count should increment since it reconnected!
        assert connections_count == 2

        await chat.close()


@pytest.mark.asyncio
async def test_websocket_transport_metadata_queries() -> None:
    """Verifies that WebSocket transport derives HTTP URLs and performs metadata queries correctly."""
    # 1. Instantiate the WebSocket transport with a dummy URL
    transport = WebSocketAgentTransport(url='ws://example.com/agent/my_agent')

    # Verify auto-derived URLs
    assert transport.state_url == 'http://example.com/agent/my_agent/state'
    assert transport.abort_url == 'http://example.com/agent/my_agent/abort'

    # Mock the cached httpx AsyncClient
    with mock.patch('genkit._ai._agents._transports._websocket.get_cached_client') as mock_get_client:
        mock_client = mock.AsyncMock()
        mock_get_client.return_value = mock_client

        # Mock get_snapshot success response
        mock_client.post.return_value = mock.MagicMock(
            status_code=200,
            json=lambda: {
                'snapshotId': 'snap_123',
                'createdAt': '2026-06-23T19:56:49Z',
                'status': 'completed',
                'state': {'messages': [], 'custom': {'score': 100}, 'artifacts': []},
            },
        )

        # Call get_snapshot
        snapshot = await transport.get_snapshot('snap_123')
        assert snapshot is not None
        assert snapshot.snapshot_id == 'snap_123'
        assert snapshot.state.custom == {'score': 100}
        mock_client.post.assert_called_once_with(
            'http://example.com/agent/my_agent/state', json={'snapshotId': 'snap_123'}
        )

        mock_client.post.reset_mock()

        # Mock abort_snapshot success response
        mock_client.post.return_value = mock.MagicMock(status_code=200, json=lambda: {'status': 'aborted'})

        # Call abort_snapshot
        status = await transport.abort_snapshot('snap_123')
        assert status == SnapshotStatus.ABORTED
        mock_client.post.assert_called_once_with(
            'http://example.com/agent/my_agent/abort', json={'snapshotId': 'snap_123'}
        )
