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

"""HTTP agent transport for client-side communication over stateless HTTP POST requests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable
from typing import Any, TypeVar, cast

from genkit._ai._agents._client import AgentTransport
from genkit._core._channel import CloseableQueue
from genkit._core._error import GenkitError
from genkit._core._http_client import get_cached_client
from genkit._core._typing import AgentInit, AgentInput, AgentOutput, AgentStreamChunk, SessionSnapshot, SnapshotStatus

StateT = TypeVar('StateT')
StreamT = TypeVar('StreamT')


class HttpAgentTransport(AgentTransport[StateT, StreamT]):
    """Client-side agent transport that talks to a remote agent over HTTP."""

    def __init__(
        self,
        url: str,
        agent_name: str | None = None,
        state_url: str | None = None,
        abort_url: str | None = None,
    ) -> None:
        """Initializes the HTTP transport.

        Args:
            url: The HTTP endpoint URL (e.g. 'http://localhost:3400/api/runAction').
            agent_name: Optional name/key of the agent, required if calling a reflection server.
            state_url: Optional URL to fetch session state.
            abort_url: Optional URL to abort/cancel session turns.
        """
        self.url = url
        self.agent_name = agent_name
        self.state_url = state_url or f'{url}/state'
        self.abort_url = abort_url or f'{url}/abort'

    async def run_turn(
        self,
        agent_input: AgentInput,
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Runs a single turn over HTTP using a streaming POST request."""
        client = get_cached_client('agent_transport')

        # Construct the REST payload
        payload: dict[str, Any] = {
            'input': agent_input.model_dump(by_alias=True, exclude_none=True),
            'init': init.model_dump(by_alias=True, exclude_none=True),
        }
        if self.agent_name:
            payload['key'] = self.agent_name

        output_future: asyncio.Future[AgentOutput] = asyncio.Future()
        stream_queue = CloseableQueue[AgentStreamChunk | BaseException]()

        async def fetch_stream() -> None:
            try:
                # We append ?stream=true to signal streaming mode to the Genkit server
                request_url = f'{self.url}?stream=true' if 'stream=true' not in self.url else self.url

                async with client.stream(
                    'POST',
                    request_url,
                    json=payload,
                    headers={'accept': 'text/event-stream'},
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise RuntimeError(
                            f'HTTP request failed ({response.status_code}): {body.decode(errors="ignore")}'
                        )

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        data = json.loads(line)
                        # Check if this is a stream chunk or the final result
                        if 'result' in data:
                            # The final response containing the output
                            output_val = AgentOutput.model_validate(data['result'])
                            if not output_future.done():
                                output_future.set_result(output_val)
                            stream_queue.close()
                            break
                        elif 'error' in data:
                            # Server returned an execution error
                            raise RuntimeError(f'Agent execution error: {data["error"]}')
                        else:
                            # It's an intermediate stream chunk (AgentStreamChunk)
                            chunk = AgentStreamChunk.model_validate(data)
                            stream_queue.put_nowait(chunk)
                    else:
                        # Premature end of HTTP stream without finding 'result'
                        err = GenkitError(
                            status='INTERNAL',
                            message='HTTP stream ended prematurely before agent turn completed',
                        )
                        if not output_future.done():
                            output_future.set_exception(err)
                        stream_queue.put_nowait(err)
            except BaseException as e:
                if not output_future.done():
                    output_future.set_exception(e)
                stream_queue.put_nowait(e)

        request_task = asyncio.create_task(fetch_stream())

        # Handle abort event
        event = abort_event
        if event is not None:

            async def watch_abort(ev: asyncio.Event = event) -> None:
                await ev.wait()
                request_task.cancel()
                if not output_future.done():
                    output_future.set_exception(asyncio.CancelledError())

            abort_task = asyncio.create_task(watch_abort())
            output_future.add_done_callback(lambda _: abort_task.cancel())

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            async for chunk in stream_queue:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return stream_generator(), output_future

    async def get_snapshot(self, snapshot_id: str) -> SessionSnapshot | None:
        """Retrieves a session snapshot from the server store."""
        client = get_cached_client('agent_transport')
        response = await client.post(self.state_url, json={'snapshotId': snapshot_id})
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return SessionSnapshot.model_validate(response.json())

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        client = get_cached_client('agent_transport')
        response = await client.post(self.abort_url, json={'snapshotId': snapshot_id})
        response.raise_for_status()
        data = response.json()
        status_val = data.get('status') if isinstance(data, dict) else None
        return cast(SnapshotStatus, SnapshotStatus(status_val)) if status_val else None

    async def close(self) -> None:
        """Cleanly closes the transport. No-op for HTTP."""
        pass
