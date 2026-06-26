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

from genkit._ai._agents._client import AgentClient, AgentTransport
from genkit._ai._agents._snapshot import parse_snapshot_lookup_kw
from genkit._ai._agents._types import StateManagement
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
        get_snapshot_url: str | None = None,
        abort_url: str | None = None,
        *,
        state_management: StateManagement,
    ) -> None:
        """Initializes the HTTP transport.

        Args:
            url: Agent turn endpoint (e.g. ``/api/myAgent``).
            agent_name: Registry action key when calling reflection ``/api/runAction``
                (e.g. ``/agent/myAgent``). Unused for dedicated agent HTTP routes.
            get_snapshot_url: ``getSnapshot`` route. Defaults to ``{url}/getSnapshot``.
            abort_url: ``abort`` route. Defaults to ``{url}/abort``.
            state_management: Declares server- vs client-managed state.
        """
        self.url = url
        self.agent_name = agent_name
        self.get_snapshot_url = get_snapshot_url or f'{url}/getSnapshot'
        self.abort_url = abort_url or f'{url}/abort'
        self.state_management: StateManagement = state_management

    async def _post_json(self, url: str, input_val: dict[str, Any]) -> Any:  # noqa: ANN401
        """POST JSON to a one-shot action endpoint and return the parsed body."""
        client = get_cached_client('agent_transport')
        response = await client.post(url, json=input_val)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        if not response.content:
            return None
        body = response.json()
        if isinstance(body, dict) and 'result' in body:
            return body['result']
        return body

    def _lookup_payload(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, str]:
        snapshot_id, session_id = parse_snapshot_lookup_kw(snapshot_id=snapshot_id, session_id=session_id)
        if snapshot_id is not None:
            return {'snapshotId': snapshot_id}
        assert session_id is not None
        return {'sessionId': session_id}

    async def run_turn(
        self,
        agent_input: AgentInput,
        init: AgentInit,
        abort_event: asyncio.Event | None = None,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Runs a single turn over HTTP using a streaming POST request."""
        client = get_cached_client('agent_transport')

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
                        if 'result' in data:
                            output_val = AgentOutput.model_validate(data['result'])
                            if not output_future.done():
                                output_future.set_result(output_val)
                            stream_queue.close()
                            break
                        if 'error' in data:
                            raise RuntimeError(f'Agent execution error: {data["error"]}')
                        chunk = AgentStreamChunk.model_validate(data)
                        stream_queue.put_nowait(chunk)
                    else:
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

        # Aborting a turn is a client-side detach: the caller stops listening,
        # but we leave the streaming request running so the server turn finishes
        # and persists. So we ignore abort_event here and let fetch_stream drain
        # to completion on its own.
        asyncio.create_task(fetch_stream())

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            async for chunk in stream_queue:
                if isinstance(chunk, BaseException):
                    raise chunk
                yield chunk

        return stream_generator(), output_future

    async def get_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        session_id: str | None = None,
    ) -> SessionSnapshot | None:
        """Retrieves a session snapshot from the server."""
        result = await self._post_json(
            self.get_snapshot_url,
            self._lookup_payload(snapshot_id=snapshot_id, session_id=session_id),
        )
        if result is None:
            return None
        return SessionSnapshot.model_validate(result)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        result = await self._post_json(self.abort_url, {'snapshotId': snapshot_id})
        if not isinstance(result, dict):
            return None
        status_val = result.get('status')
        return cast(SnapshotStatus, SnapshotStatus(status_val)) if status_val else None

    async def close(self) -> None:
        """Close the underlying bidi connection."""
        pass


def remote_agent(
    url: str,
    *,
    agent_name: str | None = None,
    get_snapshot_url: str | None = None,
    abort_url: str | None = None,
    state_management: StateManagement,
) -> AgentClient[Any, Any]:
    """Create a remote agent client over HTTP (JS ``remoteAgent`` equivalent)."""
    transport = HttpAgentTransport(
        url=url,
        agent_name=agent_name,
        get_snapshot_url=get_snapshot_url,
        abort_url=abort_url,
        state_management=state_management,
    )
    return AgentClient(transport)
