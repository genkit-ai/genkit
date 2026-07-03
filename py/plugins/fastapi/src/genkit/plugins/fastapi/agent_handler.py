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

"""Serve a Genkit agent over FastAPI with the routes the client speaks.

``remote_agent`` / ``AgentChat`` talk to an agent over three HTTP routes — run a
turn, read a snapshot, abort a snapshot. ``serve_agent`` builds an ``APIRouter``
with those routes from an ``Agent``, so apps don't hand-write the streaming and
snapshot plumbing every time — you mount it with ``app.include_router`` like any
other router. Auth and any per-request context come from a ``context_provider``,
so the agent turn sees the signed-in user the same way the rest of your app does.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from genkit.plugin_api import ContextProvider, RequestData

from .handler import _FastAPIRequestData

if TYPE_CHECKING:
    from genkit.agent import Agent, AgentInit, AgentInput

# Newline-delimited JSON: one record per line, which is the frame the client's
# HTTP transport reads back off the wire.
NDJSON_MEDIA_TYPE = 'application/x-ndjson'
_JSON_SEPARATORS = (',', ':')


def _line(obj: object) -> str:
    return json.dumps(obj, separators=_JSON_SEPARATORS) + '\n'


class _WebSocketRequestData(RequestData):
    """Adapts a WebSocket handshake to the ``RequestData`` a ``context_provider`` expects.

    Auth happens once at the handshake, so the provider reads the same headers it
    would on an HTTP request — the socket then carries many turns under that one
    resolved context.
    """

    def __init__(self, websocket: WebSocket) -> None:
        super().__init__(request=websocket)
        self.method = 'WEBSOCKET'
        self.headers = {k.lower(): v for k, v in websocket.headers.items()}
        self.input = None


def _parse_turn_body(body: dict[str, Any], *, session_id: str | None) -> tuple[AgentInput, AgentInit]:
    """Read a turn request as the agent wire format, or the ``{"message": "..."}`` shorthand.

    The shorthand keeps a plain ``curl`` one-liner working; the client always
    sends the full ``{"input": ..., "init": ...}`` shape.
    """
    from genkit.agent import AgentInit, AgentInput

    if 'message' in body:
        from genkit import Message, Part, Role, TextPart

        agent_input = AgentInput(
            message=Message(role=Role.USER, content=[Part(root=TextPart(text=str(body['message'])))]),
        )
        return agent_input, AgentInit(session_id=session_id)

    agent_input = AgentInput.model_validate(body.get('input') or {})
    init = AgentInit.model_validate(body.get('init') or {})
    if session_id and not init.session_id:
        init = init.model_copy(update={'session_id': session_id})
    return agent_input, init


def serve_agent(
    agent: Agent,
    *,
    base_path: str | None = None,
    bidi: bool = False,
    context_provider: ContextProvider | None = None,
) -> APIRouter:
    """Build an ``APIRouter`` serving ``agent`` over the routes its client speaks.

    With the default ``bidi=False``, registers, relative to ``base_path``:

    - ``POST {base_path}`` — run one turn, streaming NDJSON chunks then a final
      ``{"result": <AgentOutput>}`` (or ``{"error": ...}``).
    - ``POST {base_path}/getSnapshot`` — read a stored snapshot by id or session.
    - ``POST {base_path}/abort`` — abort a running snapshot.

    With ``bidi=True`` (experimental), the turn ``POST`` becomes a **WebSocket** at
    ``{base_path}``: auth runs once at the handshake, then the one connection carries
    many turns (client sends a turn frame, server streams that turn's NDJSON frames
    back). ``getSnapshot`` / ``abort`` stay HTTP, so a dropped socket reconnects and
    resumes from the stored snapshot. Same router either way — see the WebSocket
    example on the ``jh-fastapi-websocket-example`` branch.

    Mount the returned router like any other, so FastAPI's own ``prefix`` /
    ``dependencies`` / ``tags`` do the framework-level work::

        app.include_router(serve_agent(weather_agent), prefix='/api')
        # turn route at POST /api/weatherAgent

    Args:
        agent: The agent to serve (from ``define_agent`` / ``define_prompt_agent``).
        base_path: Path for the turn route; the snapshot and abort routes hang off
            it. Defaults to ``/<agent name>`` (e.g. a ``weatherAgent`` lands at
            ``/weatherAgent``), so serving several agents under one prefix just works.
            Pass ``''`` to mount at the router root.
        bidi: Serve the turn route as a persistent WebSocket instead of a streaming
            POST. Experimental — the WebSocket agent transport is not in core yet.
        context_provider: Reads the request and returns the context dict the turn
            runs with — the place to authenticate and attach the user. Raise to
            reject the request before the turn starts.

    Returns:
        An ``APIRouter`` with the turn route (POST or WebSocket) plus snapshot/abort.
    """
    if base_path is None:
        base_path = f'/{agent.name}'

    router = APIRouter()

    async def _resolve(request_data: RequestData) -> dict[str, object] | None:
        if context_provider is None:
            return None
        result = context_provider(request_data)
        if inspect.isawaitable(result):
            result = await result
        return cast(dict[str, object], result) if isinstance(result, dict) else None

    async def _resolve_context(request: Request, body: dict[str, Any]) -> dict[str, object] | None:
        return await _resolve(_FastAPIRequestData(request, body))

    async def _run_turn(
        body: dict[str, Any],
        query_session_id: str | None,
        context: dict[str, object] | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Drive one turn to completion, yielding chunk frames then a final result/error frame."""
        agent_input, init = _parse_turn_body(body, session_id=query_session_id)
        # Copy so a per-turn session id doesn't leak across turns on a shared socket.
        turn_context = dict(context) if context is not None else None
        if turn_context is not None and init.session_id and 'session_id' not in turn_context:
            turn_context['session_id'] = init.session_id

        conn = await agent.stream_bidi(init, context=turn_context)
        await conn.send(agent_input)
        await conn.close()
        try:
            async for chunk in conn.receive():
                yield chunk.model_dump(by_alias=True, exclude_none=True)
            output = await conn.output()
            yield {'result': output.model_dump(by_alias=True, exclude_none=True)}
        except Exception as exc:  # noqa: BLE001
            yield {'error': str(exc)}

    if bidi:

        @router.websocket(base_path)
        async def turn_socket(websocket: WebSocket) -> None:
            await websocket.accept()
            # Authenticate once at the handshake; reject by closing the socket.
            try:
                context = await _resolve(_WebSocketRequestData(websocket))
            except Exception:  # noqa: BLE001
                await websocket.close(code=1008)  # policy violation
                return
            try:
                while True:
                    body = json.loads(await websocket.receive_text())
                    query_session_id = body.get('sessionId') or body.get('session_id')
                    async for frame in _run_turn(body, query_session_id, context):
                        await websocket.send_text(_line(frame))
            except WebSocketDisconnect:
                return

    else:

        @router.post(base_path)
        async def run_turn(request: Request) -> StreamingResponse:
            body = await request.json()
            query_session_id = request.query_params.get('session_id') or request.query_params.get('thread_id')
            # Resolve context (and run auth) before streaming, so a rejection surfaces
            # as a normal error response instead of mid-stream.
            context = await _resolve_context(request, body)

            async def event_stream() -> AsyncIterator[str]:
                async for frame in _run_turn(body, query_session_id, context):
                    yield _line(frame)

            return StreamingResponse(event_stream(), media_type=NDJSON_MEDIA_TYPE)

    @router.post(f'{base_path}/getSnapshot')
    async def get_snapshot(request: Request) -> Response:
        body = await request.json()
        await _resolve_context(request, body)
        snapshot = await agent.get_snapshot(
            snapshot_id=body.get('snapshotId'),
            session_id=body.get('sessionId'),
        )
        if snapshot is None:
            return Response(status_code=404)
        return Response(
            content=json.dumps(
                {'result': snapshot.model_dump(by_alias=True, exclude_none=True)},
                separators=_JSON_SEPARATORS,
            ),
            media_type='application/json',
        )

    @router.post(f'{base_path}/abort')
    async def abort(request: Request) -> dict[str, object]:
        body = await request.json()
        await _resolve_context(request, body)
        status = await agent.abort(body['snapshotId'])
        return {'result': {'snapshotId': body['snapshotId'], 'status': str(status) if status else None}}

    return router
