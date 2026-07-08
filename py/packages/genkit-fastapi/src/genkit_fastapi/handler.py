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

"""Genkit FastAPI handler for serving flows and agents as HTTP endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, TypeVar, cast

from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from genkit import Action, ActionKind, Genkit, GenkitError
from genkit._ai._agents._snapshot import parse_abort_input, parse_snapshot_lookup_input
from genkit.agent import Agent, SessionSnapshot
from genkit.plugin_api import ContextProvider, RequestData, get_callable_json

# Compact JSON (no spaces) for smaller wire payload.
JSON_SEPARATORS = (',', ':')

StateT = TypeVar('StateT')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')
ChunkT = TypeVar('ChunkT')
InitT = TypeVar('InitT')


def to_dict(obj: Any) -> Any:  # noqa: ANN401
    """Convert object to dict if it's a Pydantic model, otherwise return as-is."""
    return obj.model_dump(by_alias=True, exclude_none=True) if isinstance(obj, BaseModel) else obj


class FastAPIRequestData(RequestData):
    """Wraps FastAPI request data for Genkit context."""

    def __init__(self, request: Request, body: dict[str, Any] | None) -> None:
        """Initialize request data wrapper."""
        super().__init__(request=request)
        self.method = request.method
        self.headers = {k.lower(): v for k, v in request.headers.items()}
        self.input = body.get('data') if body else None


def json_error_response(error: Exception, status_code: int = 400) -> Response:
    """Build a compact JSON error response from an exception."""
    ex = error.cause if isinstance(error, GenkitError) else error
    return Response(
        status_code=status_code,
        content=json.dumps(get_callable_json(ex), separators=JSON_SEPARATORS),
        media_type='application/json',
    )


def extract_action_input(body: dict[str, Any]) -> object:
    """Extract action input payload from supported wire formats."""
    if 'data' in body:
        return body['data']
    if 'input' in body:
        return body['input']
    if 'message' in body:
        return {'message': {'role': 'user', 'content': [{'text': str(body['message'])}]}}
    if 'snapshotId' in body or 'sessionId' in body:
        return body
    raise GenkitError(
        status='INVALID_ARGUMENT',
        message='Action request must be wrapped in {"data": ...} object',
    )


def resolve_session_init(body: dict[str, Any], query_params: Mapping[str, str]) -> object:
    """Resolve per-run init data, injecting session_id from query parameters if present."""
    init = body.get('init')
    query_session_id = query_params.get('session_id') or query_params.get('thread_id')
    if not query_session_id:
        return init
    if isinstance(init, dict) and not init.get('session_id') and not init.get('sessionId'):
        return {**init, 'session_id': query_session_id}
    if init is None:
        return {'session_id': query_session_id}
    return init


def wants_stream(request: Request) -> bool:
    """Check if the client requested an event stream or NDJSON stream."""
    accept = request.headers.get('accept', '')
    return 'text/event-stream' in accept or request.query_params.get('stream') == 'true'


def format_stream_chunk(chunk: object) -> str:
    """Format a stream chunk for SSE."""
    msg_json = json.dumps({'message': to_dict(chunk)}, separators=JSON_SEPARATORS)
    return f'data: {msg_json}\n\n'


def format_stream_result(result: object) -> str:
    """Format the final stream result for SSE."""
    res_json = json.dumps({'result': to_dict(result)}, separators=JSON_SEPARATORS)
    return f'data: {res_json}\n\n'


def format_stream_error(error: Exception) -> str:
    """Format a stream error for SSE."""
    ex = error.cause if isinstance(error, GenkitError) else error
    return f'error: {json.dumps({"error": get_callable_json(ex)}, separators=JSON_SEPARATORS)}\n\n'


def genkit_fastapi_handler(
    ai: Genkit | None = None,
    context_provider: ContextProvider | None = None,
) -> Callable[
    [Callable[[], Action[InputT, OutputT, ChunkT, InitT]] | Action[InputT, OutputT, ChunkT, InitT]],
    Callable[[Request], Awaitable[Response | dict[str, Any]]],
]:
    """Decorator for serving Genkit actions (flows, agents, tools, etc.) via FastAPI.

    Example (decorator on flow directly):
        ```python
        @app.post('/chat', response_model=None)
        @genkit_fastapi_handler(ai)
        @ai.flow()
        async def chat(prompt: str) -> str:
            response = await ai.generate(prompt=prompt)
            return response.text
        ```

    Example (wrapper when flow is defined later; must be async):
        ```python
        @app.post('/chat', response_model=None)
        @genkit_fastapi_handler(ai)
        async def chat():
            return my_flow


        @ai.flow()
        async def my_flow(prompt: str) -> str: ...
        ```

    Args:
        ai: Optional Genkit instance.
        context_provider: Optional function to extract context from the request.

    Returns:
        A decorator that wraps an Action or a function returning an Action.
    """

    def decorator(
        fn: Callable[[], Action[InputT, OutputT, ChunkT, InitT]] | Action[InputT, OutputT, ChunkT, InitT],
    ) -> Callable[[Request], Awaitable[Response | dict[str, Any]]]:
        async def handler(request: Request) -> Response | dict[str, Any]:
            if isinstance(fn, Action):
                action = fn
            else:
                result = fn()
                if not asyncio.iscoroutine(result):
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message='genkit_fastapi_handler wrapper must be async when action is defined elsewhere',
                    )
                action = await result
            if not isinstance(action, Action):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message='genkit_fastapi_handler must wrap an Action or an async function returning an Action',
                )

            body = await request.json()
            if not isinstance(body, dict):
                return json_error_response(
                    GenkitError(
                        status='INVALID_ARGUMENT',
                        message='Action request must be a JSON object',
                    )
                )

            try:
                input_data = extract_action_input(body)
            except GenkitError as err:
                return json_error_response(err)

            request_data = FastAPIRequestData(request, body)
            action_context: dict[str, object] | None = None

            if context_provider:
                context = context_provider(request_data)
                if asyncio.iscoroutine(context):
                    context = await context
                if isinstance(context, dict):
                    action_context = context

            init = resolve_session_init(body, request.query_params)
            action_obj = cast(Action[Any, Any, Any, Any], action)

            if wants_stream(request):

                async def event_stream() -> AsyncIterator[str]:
                    try:
                        stream_response = action_obj.stream(input_data, context=action_context, init=init)
                        async for chunk in stream_response.stream:
                            yield format_stream_chunk(chunk)
                        result = await stream_response.response
                        yield format_stream_result(result)
                    except Exception as e:
                        yield format_stream_error(e)

                return StreamingResponse(event_stream(), media_type='text/event-stream')

            try:
                response = await action_obj.run(input_data, context=action_context, init=init)
                if response.response is None and action_obj.kind == ActionKind.AGENT_SNAPSHOT:
                    return Response(status_code=404)
                return {'result': to_dict(response.response)}
            except Exception as e:
                return json_error_response(e, status_code=500)

        return handler

    return decorator


def serve_flow(
    flow: Action[InputT, OutputT, ChunkT, InitT],
    *,
    base_path: str | None = None,
    context_provider: ContextProvider | None = None,
    ai: Genkit | None = None,
) -> APIRouter:
    """Build an APIRouter serving a single flow over HTTP.

    Mount the returned router like any other, so FastAPI's own prefix / dependencies handle wiring::

        app.include_router(serve_flow(chat_flow), prefix='/api')

    Args:
        flow: The flow action to serve.
        base_path: Route path. Defaults to /<flow name>.
        context_provider: Reads the request and returns the context dict.
        ai: Optional Genkit instance.

    Returns:
        An APIRouter with the single flow route registered.
    """
    resolved_base_path = f'/{flow.name}' if base_path is None else base_path
    router = APIRouter()
    handler = genkit_fastapi_handler(ai, context_provider=context_provider)(flow)
    router.post(resolved_base_path, response_model=None)(handler)
    return router


def serve_agent(
    agent: Agent[StateT],
    *,
    base_path: str | None = None,
    context_provider: ContextProvider | None = None,
    ai: Genkit | None = None,
) -> APIRouter:
    """Build an APIRouter serving an agent and its snapshot/abort endpoints over HTTP.

    Mount the returned router like any other::

        app.include_router(serve_agent(weather_agent), prefix='/api')

    Args:
        agent: The agent to serve.
        base_path: Route path. Defaults to /<agent name>.
        context_provider: Reads the request and returns the context dict.
        ai: Optional Genkit instance.

    Returns:
        An APIRouter with the turn route plus snapshot/abort endpoints.
    """
    resolved_base_path = f'/{agent.name}' if base_path is None else base_path
    router = APIRouter()

    turn_handler = genkit_fastapi_handler(ai, context_provider=context_provider)(agent)
    router.post(resolved_base_path, response_model=None)(turn_handler)

    if agent.store is not None:

        async def snapshot_fn(input_val: dict[str, Any] | str | None = None) -> SessionSnapshot | None:
            sid, sess_id = parse_snapshot_lookup_input(input_val)
            return await agent.get_snapshot_data(snapshot_id=sid, session_id=sess_id)

        async def abort_fn(input_val: dict[str, Any] | str | None = None) -> dict[str, object]:
            snapshot_id = parse_abort_input(input_val)
            status = await agent.abort_snapshot_data(snapshot_id)
            return {'snapshotId': snapshot_id, 'status': str(status) if status else None}

        snapshot_action = Action(
            kind=ActionKind.AGENT_SNAPSHOT,
            name=f'{agent.name}_snapshot',
            fn=snapshot_fn,
            description=f'Gets snapshot data for {agent.name}',
        )
        abort_action = Action(
            kind=ActionKind.AGENT_ABORT,
            name=f'{agent.name}_abort',
            fn=abort_fn,
            description=f'Aborts {agent.name} agent by snapshotId',
        )

        snapshot_handler = genkit_fastapi_handler(ai, context_provider=context_provider)(snapshot_action)
        abort_handler = genkit_fastapi_handler(ai, context_provider=context_provider)(abort_action)
        router.post(f'{resolved_base_path}/getSnapshot', response_model=None)(snapshot_handler)
        router.post(f'{resolved_base_path}/abort', response_model=None)(abort_handler)

    return router
