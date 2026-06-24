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

"""Integration tests for the HTTP Agent Transport."""

import os
import socket
from unittest import mock

import pytest

from genkit import ActionRunContext, Genkit
from genkit._core._typing import MessageData, Part, TextPart
from genkit._core._environment import GENKIT_ENV, GenkitEnvironment
from genkit._core._reflection import ServerSpec
from genkit.agent import (
    AgentClient,
    AgentFinishReason,
    AgentInput,
    AgentResult,
    HttpAgentTransport,
    InMemoryLatestStateStore,
    SessionRunner,
    TurnResult,
)


def _find_free_port() -> int:
    """Finds an unused port on the local system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_http_transport_integration() -> None:
    """Tests HttpAgentTransport end-to-end against a local Starlette reflection server."""
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
