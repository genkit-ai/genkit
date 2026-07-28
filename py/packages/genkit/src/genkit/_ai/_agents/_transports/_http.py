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
import inspect
import json
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import Any

from pydantic import BaseModel
from typing_extensions import TypeVar as TypeVarExt

from genkit._ai._agents._client import (
    AgentClient,
    AgentTransport,
    error_from_exception,
    error_from_http,
    error_from_wire,
)
from genkit._ai._agents._snapshot import parse_snapshot_lookup_kw
from genkit._ai._agents._types import StateManagement
from genkit._core._channel import CloseableQueue
from genkit._core._error import GenkitError
from genkit._core._http_client import get_cached_client
from genkit._core._typing import (
    AgentAbortResponse,
    AgentInit,
    AgentInput,
    AgentOutput,
    AgentStreamChunk,
    SessionSnapshot,
    SnapshotStatus,
)

StateT = TypeVarExt('StateT', bound=BaseModel, default=Any)

# Auth usually rides on HTTP headers, not the agent envelope. Static dict for a
# fixed key; callable when a token needs refreshing between requests.
HeadersProvider = dict[str, str] | Callable[[], dict[str, str] | Awaitable[dict[str, str]]]


def parse_stream_line(line: str) -> dict[str, Any] | None:
    """Parse one SSE stream line into a JSON object.

    The protocol uses ``data: {...}`` (including errors as
    ``data: {"error": ...}``). The JS server currently emits failures with an
    ``error:`` prefix instead, so we accept that here too — but ``data:`` is
    what clients should expect.
    """
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith('data:'):
        stripped = stripped[5:].strip()
    elif stripped.startswith('error:'):
        # JS server emits this; protocol expects data: {"error": ...}.
        stripped = stripped[6:].strip()
    if not stripped:
        return None
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise GenkitError(status='INTERNAL', message=f'unexpected stream payload: {parsed!r}')
    return parsed


def stream_error_from_payload(data: dict[str, Any]) -> GenkitError:
    """Extract a GenkitError from a streamed error event."""
    error = data.get('error')
    if error is None:
        raise GenkitError(status='INTERNAL', message=f'stream event missing error field: {data!r}')
    # FastAPI wraps callable errors as {"error": {"error": {...}}}.
    if isinstance(error, dict) and 'error' in error:
        error = error['error']
    return error_from_wire(error)


class HttpAgentTransport(AgentTransport[StateT]):
    """Client-side agent transport that talks to a remote agent over HTTP."""

    def __init__(
        self,
        url: str,
        *,
        get_snapshot_url: str | None = None,
        abort_url: str | None = None,
        headers: HeadersProvider | None = None,
        state_management: StateManagement,
    ) -> None:
        """Initializes the HTTP transport.

        Args:
            url: Agent turn endpoint (e.g. ``/api/myAgent``).
            get_snapshot_url: ``getSnapshot`` route. Defaults to ``{url}/getSnapshot``.
            abort_url: ``abort`` route. Defaults to ``{url}/abort``.
            headers: Static headers, or a function called per request (sync or async).
            state_management: Declares server- vs client-managed state.
        """
        self.url = url
        self.get_snapshot_url = get_snapshot_url or f'{url}/getSnapshot'
        self.abort_url = abort_url or f'{url}/abort'
        self.headers = headers
        self.state_management: StateManagement = state_management
        self._background_tasks: set[asyncio.Task[Any]] = set()

    async def _resolve_headers(self) -> dict[str, str]:
        """Resolve caller headers for this request."""
        if self.headers is None:
            return {}
        if callable(self.headers):
            resolved = self.headers()
            if inspect.isawaitable(resolved):
                resolved = await resolved
            return dict(resolved)
        return dict(self.headers)

    async def _post_json(self, *, url: str, input_val: dict[str, Any]) -> Any:  # noqa: ANN401
        """POST JSON to a one-shot action endpoint and return the parsed body."""
        client = get_cached_client('agent_transport')
        # Same callable/flow envelope as run_turn: handlers expect {"data": ...}.
        response = await client.post(
            url,
            json={'data': input_val},
            headers=await self._resolve_headers(),
        )
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            body = response.text
            raise error_from_http(status_code=response.status_code, body=body)
        if not response.content:
            return None
        body = response.json()
        if isinstance(body, dict) and 'error' in body:
            raise error_from_wire(body['error'])
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
        *,
        agent_input: AgentInput,
        init: AgentInit,
    ) -> tuple[AsyncIterable[AgentStreamChunk], Awaitable[AgentOutput]]:
        """Runs a single turn over HTTP using a streaming POST request."""
        client = get_cached_client('agent_transport')

        # Callable/flow envelope used by expressHandler and FastAPI/Flask/Django
        # handlers: {"data": <AgentInput>, "init": <AgentInit>}. Streaming is
        # negotiated with Accept only (not ?stream=true).
        payload: dict[str, Any] = {
            'data': agent_input.model_dump(by_alias=True, exclude_none=True),
            'init': init.model_dump(by_alias=True, exclude_none=True),
        }

        output_future: asyncio.Future[AgentOutput] = asyncio.Future()
        stream_queue = CloseableQueue[AgentStreamChunk | Exception]()

        async def fetch_stream() -> None:
            try:
                # Accept/Content-Type win so a caller header can't break streaming.
                headers = {
                    **(await self._resolve_headers()),
                    'Accept': 'text/event-stream',
                    'Content-Type': 'application/json',
                }
                async with client.stream(
                    'POST',
                    self.url,
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors='ignore')
                        raise error_from_http(status_code=response.status_code, body=body)

                    async for line in response.aiter_lines():
                        data = parse_stream_line(line)
                        if data is None:
                            continue

                        if 'result' in data:
                            output_val = AgentOutput.model_validate(data['result'])
                            if not output_future.done():
                                output_future.set_result(output_val)
                            break
                        if 'error' in data:
                            raise stream_error_from_payload(data)

                        chunk_payload = data['message'] if 'message' in data else data
                        chunk = AgentStreamChunk.model_validate(chunk_payload)
                        stream_queue.put_nowait(chunk)
                    else:
                        err = GenkitError(
                            status='INTERNAL',
                            message='HTTP stream ended prematurely before agent turn completed',
                        )
                        if not output_future.done():
                            output_future.set_exception(err)
                        stream_queue.put_nowait(err)
            except Exception as e:
                err = e if isinstance(e, GenkitError) else error_from_exception(e)
                if not output_future.done():
                    output_future.set_exception(err)
                stream_queue.put_nowait(err)
            finally:
                # Wakes the stream consumer once buffered chunks drain, so the
                # generator ends cleanly on every path, not just the success one.
                stream_queue.close()

        # Aborting a turn is a client-side detach: the caller stops listening,
        # but we leave the streaming request running so the server turn finishes
        # and persists. Halting server-side work is a separate operation
        # (abort_snapshot), not part of running a turn.
        task = asyncio.create_task(fetch_stream())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        async def stream_generator() -> AsyncIterator[AgentStreamChunk]:
            async for chunk in stream_queue:
                if isinstance(chunk, Exception):
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
            url=self.get_snapshot_url,
            input_val=self._lookup_payload(snapshot_id=snapshot_id, session_id=session_id),
        )
        if result is None:
            return None
        return SessionSnapshot.model_validate(result)

    async def abort_snapshot(self, snapshot_id: str) -> SnapshotStatus | None:
        """Aborts the specified snapshot on the server."""
        result = await self._post_json(url=self.abort_url, input_val={'snapshotId': snapshot_id})
        if result is None:
            return None
        return AgentAbortResponse.model_validate(result).status


def remote_agent(
    url: str,
    *,
    get_snapshot_url: str | None = None,
    abort_url: str | None = None,
    headers: HeadersProvider | None = None,
    state_management: StateManagement,
    state_schema: type[StateT] | None = None,
) -> AgentClient[StateT]:
    """Create a remote agent client over HTTP."""
    transport: HttpAgentTransport[StateT] = HttpAgentTransport(
        url=url,
        get_snapshot_url=get_snapshot_url,
        abort_url=abort_url,
        headers=headers,
        state_management=state_management,
    )
    return AgentClient(transport, state_schema=state_schema)
